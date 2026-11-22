import csv

import numpy as np
import pandas as pd

from src import config
from src.validate_keypoints import array_problems, expected_frames, validate

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


def clip(frames: int) -> np.ndarray:
    keypoints = np.random.default_rng(0).uniform(0.2, 0.8, (frames, 75, 3)).astype(np.float32)
    keypoints[:, config.LEFT_HAND_SLICE] = np.nan
    return keypoints


def test_expected_frames_with_stride():
    assert expected_frames(0, 10, 2) == 5 and expected_frames(3, 10, 2) == 4


def test_array_problems_detects_defects():
    good = clip(5)
    assert array_problems(good, 5, 5) == []
    partial = good.copy()
    partial[2, config.NUM_POSE + config.NUM_HAND + 3, 1] = np.nan
    assert array_problems(partial, 5, 5) == ["partially missing right_hand"]
    assert array_problems(good, 4, None) == ["5 frames, expected 1..4"]
    assert array_problems(good.astype(np.float64), 5, 5)[0].startswith("shape/dtype")
    infinite = good.copy()
    infinite[0, 0, 0] = np.inf
    assert array_problems(infinite, 5, 5) == ["infinite values"]


def test_validate_classifies_every_state(tmp_path):
    manifest = pd.DataFrame(
        {
            "video_id": ["a", "b", "c", "d", "e"],
            "split": ["train", "train", "val", "test", "test"],
            "label": [0, 1, 0, 1, 0],
            "clip_start": [0] * 5,
            "clip_end": [10] * 5,
        }
    )
    np.save(tmp_path / "a.npy", clip(5))
    (tmp_path / "c.npy").write_bytes(b"not a numpy file")
    np.save(tmp_path / "orphan.npy", clip(5))
    with (tmp_path / "extraction_status.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(STATUS_FIELDS)
        writer.writerow(["a", "ok", 5, 1, 0, 1, 1.0, ""])
        writer.writerow(["b", "failed", 0, 0, 0, 0, 1.0, "OSError('cannot open')"])
        writer.writerow(["c", "ok", 5, 1, 0, 1, 1.0, ""])
        writer.writerow(["d", "ok", 5, 1, 0, 1, 1.0, ""])
    result = validate(manifest, tmp_path)
    states = dict(zip(result["table"]["video_id"], result["table"]["state"], strict=True))
    assert states == {
        "a": "completed",
        "b": "failed",
        "c": "corrupted",
        "d": "corrupted",
        "e": "pending",
    }
    report = result["report"]
    assert report["counts"] == {"completed": 1, "pending": 1, "failed": 1, "corrupted": 2}
    assert report["arrays_without_ok_status"] == 1
