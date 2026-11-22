from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import app
from src import config
from src.inference import DEFAULT_MODEL

needs_model = pytest.mark.skipif(
    not (DEFAULT_MODEL.exists() and config.PREPROCESS_CONFIG_JSON.exists()),
    reason="exported model or preprocessing config missing",
)
needs_videos = pytest.mark.skipif(
    not (config.VIDEO_DIR.is_dir() and config.FINAL_MANIFEST_CSV.exists()),
    reason="WLASL videos not available",
)


def write_video(path: Path, frames: int, fourcc: str = "mp4v", fps: float = 25.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (128, 128))
    for _ in range(frames):
        writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
    writer.release()
    return path


def test_rejects_missing_empty_and_wrong_type(tmp_path):
    with pytest.raises(app.VideoError, match="upload a video"):
        app.validate(None)
    with pytest.raises(app.VideoError, match="could not be found"):
        app.validate(str(tmp_path / "missing.mp4"))
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(app.VideoError, match="empty"):
        app.validate(str(empty))
    text = tmp_path / "notes.txt"
    text.write_text("hello")
    with pytest.raises(app.VideoError, match="Unsupported file type"):
        app.validate(str(text))
    fake = tmp_path / "fake.mp4"
    fake.write_text("not a video")
    with pytest.raises(app.VideoError, match="could not be opened"):
        app.validate(str(fake))


def test_rejects_too_long_video(tmp_path):
    with pytest.raises(app.VideoError, match="too long"):
        app.validate(str(write_video(tmp_path / "long.mp4", int(25 * (app.MAX_SECONDS + 2)))))


@needs_model
def test_rejects_too_short_and_no_person(tmp_path):
    with pytest.raises(app.VideoError, match="too short"):
        app.analyse(str(write_video(tmp_path / "short.mp4", 4)))
    with pytest.raises(app.VideoError, match="No person"):
        app.analyse(str(write_video(tmp_path / "blank.mp4", 40)))


@needs_model
@needs_videos
def test_matches_main_predict_and_reads_webm(tmp_path):
    import main

    manifest = pd.read_csv(config.FINAL_MANIFEST_CSV, dtype={"video_id": str})
    row = manifest[(manifest["split"] == "test") & (manifest["clip_start"] == 0)].iloc[0]
    video = config.VIDEO_DIR / f"{row.video_id}.mp4"
    result = app.analyse(str(video))
    reference = main.predict(video)
    assert [p["label"] for p in result.top5] == [p["label"] for p in reference["top5"]]
    assert np.allclose(
        [p["confidence"] for p in result.top5], [p["confidence"] for p in reference["top5"]]
    )
    assert set(result.timing_ms) == {"landmarks_ms", "preprocess_ms", "model_ms", "total_ms"}
    scores, card = app.run(str(video))[1], app.result_card(result)
    assert list(scores) == [p["word"] for p in result.top5] and result.top5[0]["word"] in card

    cap = cv2.VideoCapture(str(video))
    webm = tmp_path / "clip.webm"
    writer = cv2.VideoWriter(str(webm), cv2.VideoWriter_fourcc(*"VP80"), 25.0, (256, 256))
    while (frame := cap.read())[0]:
        writer.write(frame[1])
    writer.release()
    cap.release()
    if webm.stat().st_size == 0:
        pytest.skip("this OpenCV build cannot write WebM")
    assert len(app.analyse(str(webm)).top5) == 5
