import argparse
import json
import logging
import platform

import numpy as np
import pandas as pd

from src import config
from src.inference import Recognizer

log = logging.getLogger("benchmark")


def percentiles(values: list[float]) -> dict:
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    models = {
        "onnx_fp32": config.EXPORT_DIR / "model_fp32.onnx",
        "onnx_int8": config.EXPORT_DIR / "model_int8.onnx",
    }
    recognizers = {name: Recognizer(path) for name, path in models.items() if path.exists()}
    manifest = pd.read_csv(config.MANIFEST_CSV, dtype={"video_id": str})
    test = manifest[manifest["split"] == "test"].sample(
        n=args.videos + args.warmup, random_state=config.SEED
    )
    timings: dict[str, list[float]] = {"landmarks_ms": []}
    extractor = next(iter(recognizers.values()))
    for index, video_id in enumerate(test["video_id"]):
        keypoints, landmarks_ms = extractor.extract(config.VIDEO_DIR / f"{video_id}.mp4")
        for name, recognizer in recognizers.items():
            _, timing = recognizer.classify(keypoints)
            if index >= args.warmup:
                for key, value in timing.items():
                    timings.setdefault(f"{name}_{key}", []).append(value)
                timings.setdefault(f"{name}_end_to_end_ms", []).append(
                    landmarks_ms + sum(timing.values())
                )
        if index >= args.warmup:
            timings["landmarks_ms"].append(landmarks_ms)
    report = {
        "videos": args.videos,
        "warmup": args.warmup,
        "frame_stride": config.FRAME_STRIDE,
        "platform": f"{platform.system()} {platform.machine()} {platform.processor()}",
        "latency_ms": {key: percentiles(values) for key, values in timings.items()},
    }
    (config.REPORT_DIR / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
