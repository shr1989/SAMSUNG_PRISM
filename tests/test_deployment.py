import numpy as np
import pytest

from src import config
from src.deployed_eval import dominant_hand, hand_box, occlude
from src.energy import EnergyMeter, process_energy_joules
from src.inference import DEFAULT_MODEL
from src.streaming import SegmenterConfig, StreamingRecognizer

needs_model = pytest.mark.skipif(
    not (DEFAULT_MODEL.exists() and config.PREPROCESS_CONFIG_JSON.exists()),
    reason="exported model or preprocessing config missing",
)


def frame(hand: str | None) -> np.ndarray:
    keypoints = np.random.default_rng(0).uniform(0.3, 0.7, (75, 3)).astype(np.float32)
    keypoints[config.LEFT_HAND_SLICE] = np.nan
    keypoints[config.RIGHT_HAND_SLICE] = np.nan
    if hand:
        part = config.LEFT_HAND_SLICE if hand == "left" else config.RIGHT_HAND_SLICE
        keypoints[part] = np.random.default_rng(1).uniform(0.4, 0.6, (21, 3))
    return keypoints


@needs_model
def test_streaming_segments_one_sign_and_blocks_repeats():
    stream = StreamingRecognizer(segmenter=SegmenterConfig(threshold=0.0))
    try:
        events = []
        sequence = [None] * 3 + ["right"] * 10 + [None] * 4 + ["right"] * 10 + [None] * 4
        for t, hand in enumerate(sequence):
            events.append(stream._update(frame(hand), t * 0.08))
        predictions = [e for e in events if e is not None]
        assert len(predictions) == 2
        first, second = predictions
        assert first.segment_samples == 3 + 10 + 4
        assert first.committed and len(first.top5) == 5
        assert set(first.latency_ms) == {
            "landmarks_ms",
            "preprocess_ms",
            "model_ms",
            "end_to_end_ms",
        }
        assert second.word == first.word and not second.committed
    finally:
        stream.close()


@needs_model
def test_streaming_ignores_short_hand_flicker():
    stream = StreamingRecognizer()
    try:
        sequence = ["left"] * 2 + [None] * 4 + [None] * 10
        assert all(stream._update(frame(h), t * 0.08) is None for t, h in enumerate(sequence))
    finally:
        stream.close()


@needs_model
def test_streaming_samples_at_training_rate():
    stream = StreamingRecognizer()
    try:
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        sampled = 0
        for i in range(50):
            stream.process(blank, i / 25)
            sampled += stream.last_sample_time == i / 25
        assert sampled == 50 // config.FRAME_STRIDE
    finally:
        stream.close()


def test_dominant_hand_and_box():
    clip = np.stack([frame("left")] * 3 + [frame(None)] * 2)
    assert dominant_hand(clip) == "left"
    box = hand_box(clip, 4, "left")
    assert box is not None and box[0] < box[2] and box[1] < box[3]
    x0, y0, x1, y1 = box
    assert x1 - x0 >= 2 * 0.06 - 1e-6


def test_occlude_blacks_out_box_only():
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    out = occlude(image, (0.2, 0.2, 0.5, 0.5))
    assert (out[25:45, 25:45] == 0).all() and (out[70:, 70:] == 200).all()
    assert (image == 200).all()


def test_energy_meter_counts_cpu_work():
    meter = EnergyMeter()
    start = process_energy_joules()
    total = 0
    for i in range(2_000_000):
        total += i
    assert process_energy_joules() > start
    assert meter.average_watts() > 0
