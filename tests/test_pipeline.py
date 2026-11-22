import numpy as np
import pytest
import torch

from src import config
from src.augment import MIRROR_INDEX, augment, mirror, occlude, rotate
from src.dataset import SplitData, batches, check_disjoint
from src.model import NUM_FEATURES, build_model, frame_features
from src.preprocess import PreprocessConfig, fill_missing, normalise, preprocess, resample

SHAPE = (config.NUM_LANDMARKS, config.COORDS)


def synthetic_clip(frames: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clip = rng.uniform(0.3, 0.7, size=(frames, *SHAPE)).astype(np.float32)
    clip[:, config.LEFT_SHOULDER, :2] = [0.6, 0.4]
    clip[:, config.RIGHT_SHOULDER, :2] = [0.4, 0.4]
    return clip


def test_fill_missing_interpolates_short_interior_gap():
    clip = synthetic_clip(10)
    hand = config.RIGHT_HAND_SLICE
    clip[3:5, hand] = np.nan
    filled = fill_missing(clip, max_gap=4)
    expected = clip[2, hand] + (clip[5, hand] - clip[2, hand]) / 3
    np.testing.assert_allclose(filled[3, hand], expected, rtol=1e-5)


def test_fill_missing_leaves_long_and_leading_gaps():
    clip = synthetic_clip(12)
    hand = config.LEFT_HAND_SLICE
    clip[0:2, hand] = np.nan
    clip[4:10, hand] = np.nan
    filled = fill_missing(clip, max_gap=4)
    assert np.isnan(filled[0:2, hand]).all()
    assert np.isnan(filled[4:10, hand]).all()


def test_fill_missing_holds_short_trailing_gap():
    clip = synthetic_clip(8)
    hand = config.LEFT_HAND_SLICE
    clip[6:, hand] = np.nan
    np.testing.assert_array_equal(fill_missing(clip, 4)[7, hand], clip[5, hand])


def test_normalise_shoulder_frame():
    out = normalise(synthetic_clip())
    np.testing.assert_allclose(out[:, config.LEFT_SHOULDER, :2], [[0.5, 0.0]] * 20, atol=1e-5)
    np.testing.assert_allclose(out[:, config.RIGHT_SHOULDER, :2], [[-0.5, 0.0]] * 20, atol=1e-5)


def test_normalise_without_shoulders_is_finite():
    clip = synthetic_clip()
    clip[:, : config.NUM_POSE] = np.nan
    out = normalise(clip)
    assert np.isfinite(out[:, config.NUM_POSE :]).all()


def test_resample_preserves_order_and_endpoints():
    clip = np.arange(10, dtype=np.float32)[:, None, None] * np.ones((1, *SHAPE), np.float32)
    out = resample(clip, 32)
    assert out.shape == (32, *SHAPE)
    assert out[0, 0, 0] == 0 and out[-1, 0, 0] == 9
    assert (np.diff(out[:, 0, 0]) >= 0).all()


def test_resample_uses_nearest_when_neighbour_missing():
    clip = np.ones((4, *SHAPE), np.float32)
    clip[1] = np.nan
    out = resample(clip, 7)
    assert not np.isnan(out[0]).any() and np.isnan(out).any()


@pytest.mark.parametrize("frames", [1, 9, 30, 88])
def test_preprocess_shape_and_finite(frames):
    clip = synthetic_clip(frames)
    clip[::3, config.LEFT_HAND_SLICE] = np.nan
    out = preprocess(clip, PreprocessConfig())
    assert out.shape == (config.NUM_FRAMES, *SHAPE)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert (out[:, list(config.DROPPED_POSE)] == 0).all()


def test_preprocess_rejects_bad_shape():
    with pytest.raises(ValueError):
        preprocess(np.zeros((5, 70, 3), np.float32), PreprocessConfig())


def test_preprocess_config_roundtrip(tmp_path):
    cfg = PreprocessConfig(max_fill_gap=3)
    cfg.save(tmp_path / "p.json")
    assert PreprocessConfig.load(tmp_path / "p.json") == cfg


def batch_tensor(batch: int = 4) -> torch.Tensor:
    x = torch.randn(batch, config.NUM_FRAMES, *SHAPE)
    x[:, :10, config.LEFT_HAND_SLICE] = 0
    return x


def test_mirror_swaps_hands_and_negates_x():
    x = batch_tensor()
    out = mirror(x)
    torch.testing.assert_close(
        out[:, :, config.LEFT_HAND_SLICE, 0], -x[:, :, config.RIGHT_HAND_SLICE, 0]
    )
    torch.testing.assert_close(
        out[:, :, config.LEFT_SHOULDER, 1], x[:, :, config.RIGHT_SHOULDER, 1]
    )
    torch.testing.assert_close(mirror(out), x)
    assert sorted(MIRROR_INDEX) == list(range(config.NUM_LANDMARKS))


def test_rotation_preserves_distances_and_missing():
    x = batch_tensor()
    out = rotate(x, torch.full((4,), 25.0), torch.full((4,), -10.0))
    before = torch.linalg.vector_norm(x[:, :, 40] - x[:, :, 45], dim=-1)
    after = torch.linalg.vector_norm(out[:, :, 40] - out[:, :, 45], dim=-1)
    torch.testing.assert_close(before[:, 10:], after[:, 10:])
    assert (out[:, :10, config.LEFT_HAND_SLICE] == 0).all()


def test_augment_is_seeded_and_keeps_missing_zero():
    x = batch_tensor()
    a = augment(x, torch.Generator().manual_seed(1))
    b = augment(x, torch.Generator().manual_seed(1))
    assert a.shape == x.shape and torch.equal(a, b) and torch.isfinite(a).all()


def test_occlude_hides_expected_fraction():
    x = torch.randn(8, config.NUM_FRAMES, *SHAPE)
    out = occlude(x, 0.3, seed=0)
    hidden = (out[:, :, config.NUM_POSE :].abs().sum(-1) == 0).sum(dim=(1, 2))
    assert (hidden == round(0.3 * config.NUM_FRAMES) * config.NUM_HAND).all()


def test_frame_features_dimension_and_missing_hand():
    x = batch_tensor()
    f = frame_features(x)
    assert f.shape == (4, config.NUM_FRAMES, NUM_FEATURES)
    assert (f[:, :10, -2] == 0).all() and (f[:, 10:, -2] == 1).all()
    assert torch.isfinite(f).all()


@pytest.mark.parametrize("arch", ["cnn_gru", "cnn"])
def test_model_output_shape(arch):
    model = build_model(arch).eval()
    out = model(torch.randn(64, config.NUM_FRAMES, *SHAPE))
    assert out.shape == (64, config.NUM_CLASSES)


def test_batches_cover_all_and_disjoint_check():
    data = SplitData(torch.zeros(10, 1), torch.arange(10), np.array([str(i) for i in range(10)]))
    seen = torch.cat([y for _, y in batches(data, 3, True, torch.Generator().manual_seed(0))])
    assert sorted(seen.tolist()) == list(range(10))
    with pytest.raises(ValueError):
        check_disjoint(data, data)
