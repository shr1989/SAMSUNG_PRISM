import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src import config
from src.extract_keypoints import create_holistic, results_to_frame
from src.inference import DEFAULT_MODEL, Recognizer

SOURCE_FPS = 25.0


@dataclass
class SegmenterConfig:
    start_samples: int = 2
    end_samples: int = 4
    preroll_samples: int = 4
    min_samples: int = 6
    min_hand_samples: int = 4
    max_samples: int = 96
    threshold: float = 0.35
    repeat_block_s: float = 1.5


@dataclass
class Prediction:
    word: str
    confidence: float
    top5: list[dict]
    committed: bool
    segment_samples: int
    latency_ms: dict
    timestamp: float


@dataclass
class StreamingRecognizer:
    """Sign-by-sign recognition on a live frame stream using the offline preprocessing and model.

    Landmarks are sampled at the training rate (25 fps / FRAME_STRIDE). A sign starts when a hand
    is visible for `start_samples` samples and ends after `end_samples` samples without hands or at
    `max_samples`; the buffered keypoints (with pre-roll) are then classified exactly like a clip.
    """

    model_path: Path = DEFAULT_MODEL
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)

    def __post_init__(self) -> None:
        self.recognizer = Recognizer(self.model_path)
        self.interval = self.recognizer.cfg.frame_stride / SOURCE_FPS
        self.holistic = create_holistic()
        self.preroll: deque = deque(maxlen=self.segmenter.preroll_samples)
        self.segment: list[np.ndarray] = []
        self.present_run = self.absent_run = self.hand_samples = 0
        self.last_sample_time = -np.inf
        self.last_commit: tuple[str, float] = ("", -np.inf)
        self.captions: deque = deque(maxlen=8)
        self.latest_keypoints: np.ndarray | None = None
        self.last_landmarks_ms = 0.0

    def close(self) -> None:
        self.holistic.close()

    @property
    def signing(self) -> bool:
        return bool(self.segment)

    def process(self, rgb: np.ndarray, timestamp: float) -> Prediction | None:
        if timestamp - self.last_sample_time < self.interval - 1e-3:
            return None
        self.last_sample_time = timestamp
        began = time.perf_counter()
        keypoints = results_to_frame(self.holistic.process(rgb))
        self.last_landmarks_ms = 1000 * (time.perf_counter() - began)
        self.latest_keypoints = keypoints
        return self._update(keypoints, timestamp)

    def _update(self, keypoints: np.ndarray, timestamp: float) -> Prediction | None:
        hands = np.isnan(keypoints[config.NUM_POSE :, 0]).reshape(2, -1)
        hand_visible = bool((~hands.all(axis=1)).any())
        self.present_run = self.present_run + 1 if hand_visible else 0
        self.absent_run = 0 if hand_visible else self.absent_run + 1
        cfg = self.segmenter
        if not self.segment:
            if self.present_run >= cfg.start_samples:
                self.segment = [*self.preroll, keypoints]
                self.hand_samples = cfg.start_samples
            else:
                self.preroll.append(keypoints)
            return None
        self.segment.append(keypoints)
        self.hand_samples += hand_visible
        if self.absent_run >= cfg.end_samples or len(self.segment) >= cfg.max_samples:
            return self._finish(timestamp)
        return None

    def _finish(self, timestamp: float) -> Prediction | None:
        segment, self.segment = np.stack(self.segment), []
        self.preroll.clear()
        too_short = len(segment) < self.segmenter.min_samples
        if too_short or self.hand_samples < self.segmenter.min_hand_samples:
            return None
        predictions, timing = self.recognizer.classify(segment)
        best = predictions[0]
        word, last_time = self.last_commit
        committed = best["confidence"] >= self.segmenter.threshold and not (
            best["word"] == word and timestamp - last_time < self.segmenter.repeat_block_s
        )
        if committed:
            self.captions.append(best["word"])
            self.last_commit = (best["word"], timestamp)
        latency = {"landmarks_ms": self.last_landmarks_ms} | timing
        latency["end_to_end_ms"] = sum(latency.values())
        return Prediction(
            best["word"],
            best["confidence"],
            predictions,
            committed,
            len(segment),
            latency,
            timestamp,
        )
