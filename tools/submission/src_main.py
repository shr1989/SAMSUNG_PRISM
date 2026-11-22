"""Submission entry point: predict(video_path) -> JSON-serialisable dict.

Pipeline (identical to training and evaluation): video -> MediaPipe Holistic (every 2nd frame)
-> (T, 75, 3) keypoints -> src.preprocess.preprocess -> FP32 ONNX 1D-CNN -> top-5 words.
"""

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2

from src import config
from src.extract_keypoints import create_holistic, extract_video
from src.inference import DEFAULT_MODEL, Recognizer

INT8_MODEL = config.EXPORT_DIR / "model_int8.onnx"
MODELS = {"fp32": DEFAULT_MODEL, "int8": INT8_MODEL}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


@lru_cache(maxsize=2)
def _recognizer(model_path: str) -> Recognizer:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    if not config.PREPROCESS_CONFIG_JSON.is_file():
        raise FileNotFoundError(f"preprocessing config not found: {config.PREPROCESS_CONFIG_JSON}")
    return Recognizer(path)


def _resolve_model(model: str | Path) -> Path:
    return MODELS.get(str(model).lower(), Path(model)).resolve()


def _frame_range(path: Path, start_frame: int | None, end_frame: int | None) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot decode video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if total <= 0:
        raise ValueError(f"video reports no frames: {path}")
    start = 0 if start_frame is None else int(start_frame)
    end = total if end_frame is None else min(int(end_frame), total)
    if not 0 <= start < end:
        raise ValueError(f"invalid frame range [{start_frame}, {end_frame}) for {total} frames")
    return start, end


def predict(
    video_path: str | Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
    model: str | Path = "fp32",
) -> dict:
    """Recognise the single ASL sign in a video.

    Args:
        video_path: path to a video file containing one isolated sign.
        start_frame, end_frame: optional frame bounds [start, end) (e.g. WLASL clip bounds);
            the whole video is used by default.
        model: "fp32" (default, deployed model), "int8", or a path to an ONNX file.

    Returns:
        {"prediction": UPPERCASE word, "confidence": float, "top5": [{"word", "label",
        "confidence"}, ...], "latency_ms": {...}, "frames_used": int, "model": str}

    Raises:
        FileNotFoundError: the video or model file does not exist.
        ValueError: the file is not a decodable video or the frame range is invalid.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {path}")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        expected = sorted(VIDEO_SUFFIXES)
        raise ValueError(f"unsupported video type {path.suffix!r}; expected {expected}")
    start, end = _frame_range(path, start_frame, end_frame)
    recognizer = _recognizer(str(_resolve_model(model)))

    began = time.perf_counter()
    with create_holistic() as holistic:
        keypoints = extract_video(holistic, path, start, end, recognizer.cfg.frame_stride)
    landmarks_ms = 1000 * (time.perf_counter() - began)
    top5, timing = recognizer.classify(keypoints)
    latency = {"landmarks_ms": landmarks_ms} | timing
    latency["end_to_end_ms"] = sum(latency.values())
    return {
        "prediction": top5[0]["word"],
        "confidence": top5[0]["confidence"],
        "top5": top5,
        "latency_ms": latency,
        "frames_used": int(len(keypoints)),
        "model": recognizer.model_path.name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recognise one ASL sign in a video (offline).")
    parser.add_argument("video", type=Path)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--model", default="fp32", help='"fp32", "int8" or an ONNX path')
    args = parser.parse_args(argv)
    try:
        result = predict(args.video, args.start_frame, args.end_frame, args.model)
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
