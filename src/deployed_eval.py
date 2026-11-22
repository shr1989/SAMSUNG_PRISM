import argparse
import json
import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import config
from src.extract_keypoints import create_holistic, iter_frames, results_to_frame
from src.inference import Recognizer

log = logging.getLogger("deployed_eval")

OCCLUDED_FRACTION = 0.3
BOX_PADDING = 0.06
MIN_HALF_SIZE = 0.06
HANDS = {
    "left": (config.LEFT_HAND_SLICE, config.LEFT_WRIST),
    "right": (config.RIGHT_HAND_SLICE, config.RIGHT_WRIST),
}

_recognizer: Recognizer | None = None


def _init() -> None:
    global _recognizer
    _recognizer = Recognizer()


def dominant_hand(reference: np.ndarray) -> str:
    detected = {
        name: int((~np.isnan(reference[:, part, 0]).all(1)).sum())
        for name, (part, _) in HANDS.items()
    }
    return max(("right", "left"), key=lambda name: detected[name])


def hand_box(
    reference: np.ndarray, index: int, hand: str
) -> tuple[float, float, float, float] | None:
    part, pose_wrist = HANDS[hand]
    order = sorted(range(len(reference)), key=lambda i: abs(i - index))
    for i in order:
        points = reference[i, part, :2]
        if not np.isnan(points).all():
            low, high = np.nanmin(points, 0), np.nanmax(points, 0)
            break
        wrist = reference[i, pose_wrist, :2]
        if not np.isnan(wrist).any():
            low = high = wrist
            break
    else:
        return None
    centre, half = (low + high) / 2, np.maximum((high - low) / 2 + BOX_PADDING, MIN_HALF_SIZE)
    return (*(centre - half), *(centre + half))


def occlude(frame: np.ndarray, box) -> np.ndarray:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    out = frame.copy()
    out[max(0, int(y0 * height)) : int(y1 * height), max(0, int(x0 * width)) : int(x1 * width)] = 0
    return out


def extract(frames: list[np.ndarray]) -> np.ndarray:
    with create_holistic() as holistic:
        return np.stack([results_to_frame(holistic.process(frame)) for frame in frames])


def evaluate_video(task: tuple[str, int, int, int, int]) -> dict:
    video_id, label, start, end, seed = task
    reference = np.load(config.KEYPOINT_DIR / f"{video_id}.npy")
    frames = list(
        iter_frames(config.VIDEO_DIR / f"{video_id}.mp4", start, end, config.FRAME_STRIDE)
    )
    frames = frames[: len(reference)]
    hand = dominant_hand(reference)
    span = max(1, round(OCCLUDED_FRACTION * len(frames)))
    first = np.random.default_rng(seed).integers(0, len(frames) - span + 1)
    hidden = range(first, first + span)
    occluded_frames = [
        occlude(f, box) if i in hidden and (box := hand_box(reference, i, hand)) else f
        for i, f in enumerate(frames)
    ]
    result = {"video_id": video_id, "label": label, "hand": hand, "frames": len(frames)}
    for name, clip in (
        ("stored", reference),
        ("clean", extract(frames)),
        ("occluded", extract(occluded_frames)),
    ):
        predictions, _ = _recognizer.classify(clip)
        labels = [p["label"] for p in predictions]
        result[f"{name}_top1"] = labels[0] == label
        result[f"{name}_top5"] = label in labels
        result[f"{name}_pred"] = labels[0]
        if name != "stored":
            part = HANDS[hand][0]
            seen = ~np.isnan(clip[list(hidden), part, 0]).all(1)
            result[f"{name}_hand_detected_in_window"] = float(seen.mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest = pd.read_csv(config.FINAL_MANIFEST_CSV, dtype={"video_id": str})
    test = manifest[manifest["split"] == "test"].sample(n=args.videos, random_state=config.SEED)
    tasks = [
        (r.video_id, int(r.label), int(r.clip_start), int(r.clip_end), config.SEED + i)
        for i, r in enumerate(test.itertuples())
    ]
    with Pool(args.workers, _init) as pool:
        rows = pd.DataFrame(pool.map(evaluate_video, tasks, chunksize=4))
    summary = {
        "videos": len(rows),
        "method": (
            f"deployed path (video -> MediaPipe -> preprocess -> FP32 ONNX); occluded variant "
            f"blacks out the dominant hand's region in {OCCLUDED_FRACTION:.0%} contiguous "
            "frames before MediaPipe"
        ),
        "agreement_clean_vs_stored_top1": float((rows["clean_pred"] == rows["stored_pred"]).mean()),
    }
    for name in ("stored", "clean", "occluded"):
        summary[f"{name}_top1"] = float(rows[f"{name}_top1"].mean())
        summary[f"{name}_top5"] = float(rows[f"{name}_top5"].mean())
    summary["occlusion_top1_drop_pp"] = 100 * (summary["clean_top1"] - summary["occluded_top1"])
    summary["occlusion_top5_drop_pp"] = 100 * (summary["clean_top5"] - summary["occluded_top5"])
    for name in ("clean", "occluded"):
        summary[f"{name}_hand_detected_in_window"] = float(
            rows[f"{name}_hand_detected_in_window"].mean()
        )
    rows.to_csv(config.REPORT_DIR / "deployed_eval_videos.csv", index=False)
    (config.REPORT_DIR / "deployed_eval.json").write_text(json.dumps(summary, indent=2))
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
