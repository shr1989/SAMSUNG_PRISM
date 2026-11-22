import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from src import config
from src.extract_keypoints import create_holistic, extract_video
from src.preprocess import PreprocessConfig, preprocess

DEFAULT_MODEL = config.EXPORT_DIR / "model_fp32.onnx"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum()


@dataclass
class Recognizer:
    model_path: Path = DEFAULT_MODEL
    preprocess_config: Path = config.PREPROCESS_CONFIG_JSON
    top_k: int = 5

    def __post_init__(self) -> None:
        self.cfg = PreprocessConfig.load(self.preprocess_config)
        options = ort.SessionOptions()
        # One thread: the model takes well under 1 ms, and idle-spinning worker threads cost power.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), options, providers=["CPUExecutionProvider"]
        )
        frames = self.session.get_inputs()[0].shape[1]
        if frames != self.cfg.num_frames:
            raise ValueError(
                f"model expects {frames} frames but preprocessing produces {self.cfg.num_frames}"
            )
        manifest = pd.read_csv(config.MANIFEST_CSV, usecols=["label", "gloss"])
        self.glosses = manifest.drop_duplicates("label").set_index("label")["gloss"].to_dict()

    def classify(self, keypoints: np.ndarray) -> tuple[list[dict], dict]:
        began = time.perf_counter()
        features = preprocess(keypoints, self.cfg)[None]
        prepared = time.perf_counter()
        logits = self.session.run(None, {"keypoints": features})[0][0]
        finished = time.perf_counter()
        probs = softmax(logits)
        best = np.argsort(probs)[::-1][: self.top_k]
        predictions = [
            {"word": self.glosses[int(i)].upper(), "label": int(i), "confidence": float(probs[i])}
            for i in best
        ]
        timing = {
            "preprocess_ms": 1000 * (prepared - began),
            "model_ms": 1000 * (finished - prepared),
        }
        return predictions, timing

    def extract(self, video_path: Path) -> tuple[np.ndarray, float]:
        began = time.perf_counter()
        frames = _frame_count(video_path)
        with create_holistic() as holistic:
            keypoints = extract_video(holistic, video_path, 0, frames, self.cfg.frame_stride)
        return keypoints, 1000 * (time.perf_counter() - began)

    def predict_video(self, video_path: Path) -> dict:
        keypoints, landmarks_ms = self.extract(video_path)
        predictions, timing = self.classify(keypoints)
        return {
            "word": predictions[0]["word"],
            "confidence": predictions[0]["confidence"],
            "top5": predictions,
            "latency_ms": {"landmarks_ms": landmarks_ms}
            | timing
            | {"end_to_end_ms": landmarks_ms + sum(timing.values())},
            "frames_used": int(len(keypoints)),
        }


def _frame_count(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise OSError(f"cannot open {video_path}")
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
