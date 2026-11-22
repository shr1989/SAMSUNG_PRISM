import argparse
import json
import logging
import time
from collections.abc import Iterator

import cv2
import numpy as np
import pandas as pd
import psutil

from src import config
from src.energy import EnergyMeter, thermal_state
from src.inference import DEFAULT_MODEL
from src.streaming import SOURCE_FPS, SegmenterConfig, StreamingRecognizer

log = logging.getLogger("live_benchmark")


def replay_frames(videos: pd.DataFrame, gap_frames: int) -> Iterator[tuple[np.ndarray, int]]:
    """Endless camera-like stream: test clips at 25 fps separated by blank (no person) frames."""
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    while True:
        for row in videos.itertuples():
            cap = cv2.VideoCapture(str(config.VIDEO_DIR / f"{row.video_id}.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, row.clip_start)
            for _ in range(row.clip_end - row.clip_start):
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), row.label
            cap.release()
            for _ in range(gap_frames):
                yield blank, -1


def percentiles(values) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {}
    return {f"p{q}": float(np.percentile(values, q)) for q in (50, 95, 99)} | {
        "mean": float(values.mean()),
        "max": float(values.max()),
        "n": int(len(values)),
    }


def run(minutes: float, gap_s: float, model_path) -> dict:
    manifest = pd.read_csv(config.FINAL_MANIFEST_CSV, dtype={"video_id": str})
    videos = manifest[manifest["split"] == "test"].sample(frac=1.0, random_state=config.SEED)
    stream = StreamingRecognizer(model_path)
    frames = replay_frames(videos, int(gap_s * SOURCE_FPS))
    process = psutil.Process()
    samples, predictions, minutes_log = [], [], []
    meter, minute_meter = EnergyMeter(), EnergyMeter()
    cpu_start = sum(process.cpu_times()[:2])
    start = time.perf_counter()
    next_minute, frame_index, overruns, segment_label = 60.0, 0, 0, -1
    duration = minutes * 60
    try:
        while (elapsed := time.perf_counter() - start) < duration:
            target = frame_index / SOURCE_FPS
            if target > elapsed:
                time.sleep(target - elapsed)
            elif elapsed - target > 1 / SOURCE_FPS:
                overruns += 1
            rgb, label = next(frames)
            was_signing = stream.signing
            now = time.perf_counter() - start
            prediction = stream.process(rgb, now)
            if stream.last_sample_time == now:
                samples.append({"t": now, "landmarks_ms": stream.last_landmarks_ms})
            if stream.signing and not was_signing:
                segment_label = label
            if prediction is not None:
                predictions.append(
                    {
                        "t": now,
                        "true": segment_label,
                        "pred": prediction.top5[0]["label"],
                        "in_top5": segment_label in [p["label"] for p in prediction.top5],
                        "confidence": prediction.confidence,
                        "committed": prediction.committed,
                        "samples": prediction.segment_samples,
                    }
                    | prediction.latency_ms
                )
            frame_index += 1
            if now >= next_minute:
                minutes_log.append(
                    {
                        "minute": int(next_minute // 60),
                        "avg_power_w": minute_meter.average_watts(),
                        "thermal": thermal_state(),
                        "overruns_so_far": overruns,
                    }
                )
                minute_meter.reset()
                next_minute += 60
    finally:
        stream.close()
    wall = time.perf_counter() - start
    return {
        "samples": pd.DataFrame(samples),
        "predictions": pd.DataFrame(predictions),
        "minutes": minutes_log,
        "summary": {
            "duration_s": wall,
            "frames": frame_index,
            "frames_behind_realtime": overruns,
            "average_power_w": meter.average_watts(),
            "cpu_utilisation_percent_of_one_core": 100
            * (sum(process.cpu_times()[:2]) - cpu_start)
            / wall,
        },
    }


def drift(frame: pd.DataFrame, column: str, warmup_s: float = 30.0) -> dict:
    # Compares the first and last full minutes after warm-up.
    early = frame[(frame["t"] >= warmup_s) & (frame["t"] < warmup_s + 60)][column]
    late = frame[frame["t"] >= frame["t"].max() - 60][column]
    result = {"first_minute": percentiles(early), "last_minute": percentiles(late)}
    for q in ("p50", "p95"):
        result[f"{q}_drift_percent"] = 100 * (
            result["last_minute"][q] / result["first_minute"][q] - 1
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--gap-s", type=float, default=0.6)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--name", default="live_benchmark")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = run(args.minutes, args.gap_s, args.model)
    samples, predictions = result["samples"], result["predictions"]
    labelled = predictions[predictions["true"] >= 0]
    report = {
        "method": "test clips replayed at 25 fps (real-time pacing) through StreamingRecognizer",
        "model": str(args.model),
        "summary": result["summary"],
        "per_sample_landmarks_ms": percentiles(samples["landmarks_ms"]),
        "prediction_latency_ms": {
            key: percentiles(predictions[key])
            for key in ("landmarks_ms", "preprocess_ms", "model_ms", "end_to_end_ms")
        },
        "decision_delay_ms": 1000
        * SegmenterConfig().end_samples
        * config.FRAME_STRIDE
        / SOURCE_FPS,
        "predictions": int(len(predictions)),
        "streaming_top1": float((labelled["pred"] == labelled["true"]).mean()),
        "streaming_top5": float(labelled["in_top5"].mean()),
        "landmarks_drift": drift(samples, "landmarks_ms"),
        "end_to_end_drift": drift(predictions, "end_to_end_ms"),
        "per_minute": result["minutes"],
    }
    samples.to_csv(config.REPORT_DIR / f"{args.name}_samples.csv", index=False)
    predictions.to_csv(config.REPORT_DIR / f"{args.name}_predictions.csv", index=False)
    (config.REPORT_DIR / f"{args.name}.json").write_text(json.dumps(report, indent=2))
    log.info(json.dumps({k: v for k, v in report.items() if k != "per_minute"}, indent=2))


if __name__ == "__main__":
    main()
