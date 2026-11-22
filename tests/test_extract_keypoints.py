from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from src import config
from src.extract_keypoints import (
    _process,
    create_holistic,
    detection_rates,
    extract_video,
    iter_frames,
    landmarks_to_array,
    pending_tasks,
    read_status,
    results_to_frame,
    run,
    save_atomic,
)

REAL_VIDEO = config.VIDEO_DIR / "00335.mp4"


def fake_landmarks(count: int, value: float):
    return SimpleNamespace(
        landmark=[SimpleNamespace(x=value, y=value + 0.1, z=value + 0.2) for _ in range(count)]
    )


def write_video(path, frames: int, size: int = 64):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (size, size))
    for i in range(frames):
        writer.write(np.full((size, size, 3), i, dtype=np.uint8))
    writer.release()


def test_missing_landmarks_are_nan():
    out = landmarks_to_array(None, config.NUM_HAND)
    assert out.shape == (config.NUM_HAND, 3) and np.isnan(out).all()


def test_landmark_ordering():
    results = SimpleNamespace(
        pose_landmarks=fake_landmarks(config.NUM_POSE, 0.1),
        left_hand_landmarks=None,
        right_hand_landmarks=fake_landmarks(config.NUM_HAND, 0.7),
    )
    frame = results_to_frame(results)
    assert frame.shape == (config.NUM_LANDMARKS, config.COORDS)
    np.testing.assert_allclose(frame[: config.NUM_POSE], [[0.1, 0.2, 0.3]] * config.NUM_POSE)
    assert np.isnan(frame[config.LEFT_HAND_SLICE]).all()
    np.testing.assert_allclose(frame[config.RIGHT_HAND_SLICE, 0], 0.7)


def test_detection_rates():
    keypoints = np.zeros((4, config.NUM_LANDMARKS, 3), dtype=np.float32)
    keypoints[:3, config.LEFT_HAND_SLICE] = np.nan
    assert detection_rates(keypoints) == (1.0, 0.25, 1.0)


def test_iter_frames_respects_clip_bounds(tmp_path):
    path = tmp_path / "clip.mp4"
    write_video(path, 12)
    assert len(list(iter_frames(path, 0, 12))) == 12
    assert len(list(iter_frames(path, 3, 8))) == 5
    assert len(list(iter_frames(path, 0, 50))) == 12


def test_iter_frames_unopenable(tmp_path):
    with pytest.raises(OSError):
        list(iter_frames(tmp_path / "missing.mp4", 0, 5))


def test_extract_blank_video_yields_nan(tmp_path):
    path = tmp_path / "blank.mp4"
    write_video(path, 6, size=128)
    with create_holistic() as holistic:
        keypoints = extract_video(holistic, path, 0, 6)
    assert keypoints.shape == (6, config.NUM_LANDMARKS, config.COORDS)
    assert np.isnan(keypoints).all()


@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="dataset not extracted")
def test_extract_real_video():
    with create_holistic() as holistic:
        keypoints = extract_video(holistic, REAL_VIDEO, 0, 10)
    assert keypoints.shape == (10, config.NUM_LANDMARKS, config.COORDS)
    assert keypoints.dtype == np.float32
    assert not np.isnan(keypoints[:, : config.NUM_POSE]).all()


def test_failed_video_is_recorded(tmp_path):
    result = _process(("does_not_exist", 0, 10, str(tmp_path)))
    assert result.status == "failed" and "cannot open" in result.error
    assert not list(tmp_path.iterdir())


def test_save_atomic(tmp_path):
    target = tmp_path / "a.npy"
    save_atomic(np.ones((2, 75, 3), dtype=np.float32), target)
    assert [p.name for p in tmp_path.iterdir()] == ["a.npy"]
    assert np.load(target).shape == (2, 75, 3)


def test_pending_tasks_resume(tmp_path):
    selected = pd.DataFrame(
        {
            "video_id": ["a", "b", "c", "d", "e"],
            "clip_start": [0, 0, 2, 0, 0],
            "clip_end": [5, 5, 9, 5, 5],
        }
    )
    for video_id in ("a", "d"):
        np.save(tmp_path / f"{video_id}.npy", np.zeros(1))
    status = pd.DataFrame(
        [
            {"video_id": "a", "status": "ok"},
            {"video_id": "b", "status": "failed"},
            {"video_id": "e", "status": "ok"},
        ]
    )
    assert [t[0] for t in pending_tasks(selected, status, tmp_path, False)] == ["c", "d", "e"]
    assert [t[0] for t in pending_tasks(selected, status, tmp_path, True)] == ["b", "c", "d", "e"]
    assert pending_tasks(selected, status, tmp_path, False)[0][1:3] == (2, 9)


@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="dataset not extracted")
def test_run_writes_status_and_resumes(tmp_path):
    status_path = tmp_path / "_status.csv"
    tasks = [("00335", 0, 8, str(tmp_path)), ("missing_id", 0, 8, str(tmp_path))]
    run(tasks, 1, status_path)
    status = read_status(status_path).set_index("video_id")
    assert status.loc["00335", "status"] == "ok"
    assert status.loc["missing_id", "status"] == "failed"
    assert (tmp_path / "00335.npy").exists()

    selected = pd.DataFrame(
        {"video_id": ["00335", "missing_id"], "clip_start": [0, 0], "clip_end": [8, 8]}
    )
    assert pending_tasks(selected, read_status(status_path), tmp_path, False) == []


def test_iter_frames_stride(tmp_path):
    path = tmp_path / "clip.mp4"
    write_video(path, 12)
    frames = list(iter_frames(path, 0, 12, stride=2))
    full = list(iter_frames(path, 0, 12))
    assert len(frames) == 6
    for strided, original in zip(frames, full[::2], strict=True):
        np.testing.assert_array_equal(strided, original)
    assert len(list(iter_frames(path, 3, 8, stride=2))) == 3
