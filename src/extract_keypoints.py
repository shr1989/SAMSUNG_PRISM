import argparse
import csv
import json
import logging
import os
import time
import warnings
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from src import config

log = logging.getLogger("extract_keypoints")

STATUS_FIELDS = [
    "video_id",
    "status",
    "frames",
    "pose_rate",
    "left_rate",
    "right_rate",
    "seconds",
    "error",
]


@dataclass
class ExtractionResult:
    video_id: str
    status: str
    frames: int = 0
    pose_rate: float = 0.0
    left_rate: float = 0.0
    right_rate: float = 0.0
    seconds: float = 0.0
    error: str = ""


def landmarks_to_array(landmarks, count: int) -> np.ndarray:
    if landmarks is None:
        return np.full((count, config.COORDS), np.nan, dtype=np.float32)
    return np.array([(p.x, p.y, p.z) for p in landmarks.landmark], dtype=np.float32)


def results_to_frame(results) -> np.ndarray:
    return np.concatenate(
        [
            landmarks_to_array(results.pose_landmarks, config.NUM_POSE),
            landmarks_to_array(results.left_hand_landmarks, config.NUM_HAND),
            landmarks_to_array(results.right_hand_landmarks, config.NUM_HAND),
        ]
    )


def iter_frames(path: Path, start: int, end: int, stride: int = 1) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for offset in range(end - start):
            if offset % stride:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def create_holistic(model_complexity: int = config.HOLISTIC_MODEL_COMPLEXITY):
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=config.HOLISTIC_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.HOLISTIC_MIN_TRACKING_CONFIDENCE,
    )


def extract_video(holistic, path: Path, start: int, end: int, stride: int = 1) -> np.ndarray:
    frames = [
        results_to_frame(holistic.process(rgb)) for rgb in iter_frames(path, start, end, stride)
    ]
    if not frames:
        raise ValueError("no decodable frames")
    return np.stack(frames)


def detection_rates(keypoints: np.ndarray) -> tuple[float, float, float]:
    present = ~np.isnan(keypoints[..., 0])
    return tuple(
        float(present[:, part].all(axis=1).mean())
        for part in (
            slice(0, config.NUM_POSE),
            config.LEFT_HAND_SLICE,
            config.RIGHT_HAND_SLICE,
        )
    )


def save_atomic(array: np.ndarray, target: Path) -> None:
    tmp = target.with_name(f".{target.stem}.tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, target)


_model_complexity = config.HOLISTIC_MODEL_COMPLEXITY
_frame_stride = config.FRAME_STRIDE


def _init_worker(model_complexity: int, frame_stride: int) -> None:
    global _model_complexity, _frame_stride
    _model_complexity = model_complexity
    _frame_stride = frame_stride
    warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype", category=UserWarning)
    cv2.setNumThreads(1)


def _process(task: tuple[str, int, int, str]) -> ExtractionResult:
    video_id, start, end, out_dir = task
    began = time.perf_counter()
    try:
        # A fresh graph per video prevents tracking state leaking between clips.
        with create_holistic(_model_complexity) as holistic:
            keypoints = extract_video(
                holistic, config.VIDEO_DIR / f"{video_id}.mp4", start, end, _frame_stride
            )
        save_atomic(keypoints, Path(out_dir) / f"{video_id}.npy")
        pose, left, right = detection_rates(keypoints)
        return ExtractionResult(
            video_id, "ok", len(keypoints), pose, left, right, time.perf_counter() - began
        )
    except Exception as exc:  # recorded per video and reported
        return ExtractionResult(
            video_id, "failed", seconds=time.perf_counter() - began, error=repr(exc)
        )


def read_status(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=STATUS_FIELDS)
    return pd.read_csv(path, dtype={"video_id": str}, keep_default_na=False).drop_duplicates(
        "video_id", keep="last"
    )


def select_videos(manifest: pd.DataFrame, limit: int | None, split: str | None) -> pd.DataFrame:
    if split:
        manifest = manifest[manifest["split"] == split]
    if limit:
        manifest = manifest.sample(n=min(limit, len(manifest)), random_state=config.SEED)
    return manifest


def pending_tasks(
    selected: pd.DataFrame, status: pd.DataFrame, out_dir: Path, retry_failed: bool
) -> list[tuple[str, int, int, str]]:
    failed = set(status.loc[status["status"] == "failed", "video_id"])
    succeeded = set(status.loc[status["status"] == "ok", "video_id"])
    tasks = []
    for row in selected.itertuples():
        # An array without an "ok" status row was written after the run was interrupted.
        done = row.video_id in succeeded and (out_dir / f"{row.video_id}.npy").exists()
        if done or (row.video_id in failed and not retry_failed):
            continue
        tasks.append((row.video_id, int(row.clip_start), int(row.clip_end), str(out_dir)))
    return tasks


def process_memory_mb() -> float:
    proc = psutil.Process()
    rss = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return rss / 2**20


@dataclass
class RunStats:
    peak_rss_mb: float
    mean_system_cpu_percent: float
    frames: int
    videos: int


def run(
    tasks: list,
    workers: int,
    status_path: Path,
    model_complexity: int = config.HOLISTIC_MODEL_COMPLEXITY,
    frame_stride: int = config.FRAME_STRIDE,
    max_minutes: float | None = None,
) -> RunStats:
    peak_mb = process_memory_mb()
    frames = done = 0
    began = time.perf_counter()
    psutil.cpu_percent(None)
    new_file = not status_path.exists()
    with (
        status_path.open("a", newline="") as handle,
        Pool(workers, _init_worker, (model_complexity, frame_stride)) as pool,
        tqdm(total=len(tasks), unit="video", mininterval=30, ascii=True) as progress,
    ):
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        if new_file:
            writer.writeheader()
        for done, result in enumerate(pool.imap_unordered(_process, tasks), start=1):
            progress.update()
            writer.writerow(asdict(result))
            handle.flush()
            frames += result.frames
            peak_mb = max(peak_mb, process_memory_mb())
            if result.status == "failed":
                log.warning("failed %s: %s", result.video_id, result.error)
            if max_minutes and (time.perf_counter() - began) / 60 > max_minutes:
                # Leaving the Pool context terminates in-flight workers; resume picks them up.
                log.info("time limit of %.0f min reached after %d videos", max_minutes, done)
                break
            if done % 500 == 0 or done == len(tasks):
                rate = 60 * done / (time.perf_counter() - began)
                log.info(
                    "%d/%d videos, %.1f videos/min, ETA %.1f min, peak RSS %.0f MB",
                    done,
                    len(tasks),
                    rate,
                    (len(tasks) - done) / rate,
                    peak_mb,
                )
    return RunStats(peak_mb, psutil.cpu_percent(None), frames, done)


def summarise(status: pd.DataFrame, video_ids: set[str], stats: RunStats, wall: float) -> dict:
    subset = status[status["video_id"].isin(video_ids)]
    ok = subset[subset["status"] == "ok"].astype(
        {c: float for c in ("frames", "pose_rate", "left_rate", "right_rate", "seconds")}
    )
    total_frames = max(ok["frames"].sum(), 1)

    def frame_weighted(column: str) -> float:
        return round(float((ok[column] * ok["frames"]).sum() / total_frames), 4)

    return {
        "videos_selected": len(video_ids),
        "succeeded": int(len(ok)),
        "failed": int((subset["status"] == "failed").sum()),
        "failures": subset.loc[subset["status"] == "failed", ["video_id", "error"]].to_dict(
            "records"
        ),
        "frames_processed": int(ok["frames"].sum()),
        "pose_detection_rate": frame_weighted("pose_rate"),
        "left_hand_detection_rate": frame_weighted("left_rate"),
        "right_hand_detection_rate": frame_weighted("right_rate"),
        "videos_without_any_hand": int(((ok["left_rate"] == 0) & (ok["right_rate"] == 0)).sum()),
        "mean_seconds_per_video_per_worker": round(float(ok["seconds"].mean()), 3),
        "mean_ms_per_frame_per_worker": round(1000 * float(ok["seconds"].sum() / total_frames), 2),
        "wall_seconds_this_run": round(wall, 2),
        "videos_this_run": stats.videos,
        "videos_per_minute_this_run": round(60 * stats.videos / max(wall, 1e-9), 1),
        "frames_per_second_this_run": round(stats.frames / max(wall, 1e-9), 1),
        "mean_system_cpu_percent_this_run": stats.mean_system_cpu_percent,
        "peak_rss_mb_this_run": round(stats.peak_rss_mb, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=config.SPLITS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=config.KEYPOINT_DIR)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--report", type=Path, default=config.REPORT_DIR / "extraction_report.json")
    parser.add_argument("--manifest", type=Path, default=config.TRAINING_MANIFEST_CSV)
    parser.add_argument(
        "--model-complexity", type=int, choices=(0, 1, 2), default=config.HOLISTIC_MODEL_COMPLEXITY
    )
    parser.add_argument("--frame-stride", type=int, default=config.FRAME_STRIDE)
    parser.add_argument("--max-minutes", type=float)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault("GLOG_minloglevel", "2")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    status_path = args.out_dir / config.EXTRACTION_STATUS_CSV.name

    manifest = pd.read_csv(args.manifest, dtype={"video_id": str})
    selected = select_videos(manifest, args.limit, args.split)
    tasks = pending_tasks(selected, read_status(status_path), args.out_dir, args.retry_failed)
    log.info("%d selected, %d pending, %d workers", len(selected), len(tasks), args.workers)

    began = time.perf_counter()
    stats = (
        run(
            tasks,
            args.workers,
            status_path,
            args.model_complexity,
            args.frame_stride,
            args.max_minutes,
        )
        if tasks
        else RunStats(process_memory_mb(), 0.0, 0, 0)
    )
    report = summarise(
        read_status(status_path), set(selected["video_id"]), stats, time.perf_counter() - began
    )
    report["pending_at_start"] = len(tasks)
    report["config"] = {
        "manifest": str(args.manifest.name),
        "workers": args.workers,
        "model_complexity": args.model_complexity,
        "frame_stride": args.frame_stride,
    }
    args.report.write_text(json.dumps(report, indent=2))
    log.info(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
