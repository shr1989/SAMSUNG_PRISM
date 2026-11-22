import numpy as np
import pytest
import torch

from src import config
from src.augment import MIRROR_INDEX, augment, mirror, occlude, rotate
from src.dataset import SplitData, batches, check_disjoint
from src.model import NUM_FEATURES_576, build_model, features_576, hand_upright, size_mb
from src.model_experimental import EXPERIMENTS, NUM_FEATURES_314, features_314
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


def test_features_576_composition_and_missing_hand():
    x = batch_tensor()
    f = features_576(x)
    assert NUM_FEATURES_576 == 225 + 126 + 225 == 576
    assert f.shape == (4, config.NUM_FRAMES, 576) and torch.isfinite(f).all()
    torch.testing.assert_close(f[..., :225], x.flatten(2))
    assert (f[:, :10, 225:288] == 0).all()
    left_velocity = f[:, :11, 351:576].view(4, 11, 75, 3)[:, :, config.LEFT_HAND_SLICE]
    assert (left_velocity == 0).all()


def test_hand_upright_aligns_middle_knuckle_upwards():
    hand = torch.zeros(1, 1, config.NUM_HAND, 3)
    hand[0, 0, 9] = torch.tensor([0.3, 0.4, 0.0])
    hand[0, 0, 5] = torch.tensor([0.6, 0.8, 0.0])
    out = hand_upright(hand).view(config.NUM_HAND, 3)
    torch.testing.assert_close(out[9], torch.tensor([0.0, -1.0, 0.0]))
    torch.testing.assert_close(out[5], torch.tensor([0.0, -2.0, 0.0]))


def test_hand_upright_missing_hand_is_zero():
    assert (hand_upright(torch.zeros(2, 3, config.NUM_HAND, 3)) == 0).all()


def test_features_314_dimension_and_flags():
    x = batch_tensor()
    f = features_314(x)
    assert f.shape == (4, config.NUM_FRAMES, NUM_FEATURES_314)
    assert (f[:, :10, -2] == 0).all() and (f[:, 10:, -2] == 1).all()


def test_tcnn_matches_specification():
    model = build_model("tcnn").eval()
    assert model.projection[0].in_features == 576 and model.projection[0].out_features == 192
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv1d)]
    assert [c.kernel_size[0] for c in convs] == [5, 5] and [c.stride[0] for c in convs] == [1, 2]
    assert not any(isinstance(m, (torch.nn.GRU, torch.nn.LSTM)) for m in model.modules())
    assert size_mb(model) < 20
    with pytest.raises(ValueError):
        build_model("cnn_gru")


@pytest.mark.parametrize("arch", ["tcnn", "cnn", "cnn_gru"])
def test_model_output_shape(arch):
    model = (EXPERIMENTS[arch]() if arch in EXPERIMENTS else build_model(arch)).eval()
    out = model(torch.randn(64, config.NUM_FRAMES, *SHAPE))
    assert out.shape == (64, config.NUM_CLASSES)


def test_batches_cover_all_and_disjoint_check():
    data = SplitData(torch.zeros(10, 1), torch.arange(10), np.array([str(i) for i in range(10)]))
    seen = torch.cat([y for _, y in batches(data, 3, True, torch.Generator().manual_seed(0))])
    assert sorted(seen.tolist()) == list(range(10))
    with pytest.raises(ValueError):
        check_disjoint(data, data)


def test_training_feature_stats_rejects_other_training_set(tmp_path, monkeypatch):
    from src import train as train_module
    from src.preprocess import ids_fingerprint

    data = SplitData(
        torch.randn(4, config.NUM_FRAMES, *SHAPE), torch.arange(4), np.array(["a", "b", "c", "d"])
    )
    path = tmp_path / "stats.npz"
    monkeypatch.setattr(config, "FEATURE_STATS_NPZ", path)
    np.savez(
        path,
        mean=np.zeros(576, np.float32),
        std=np.ones(576, np.float32),
        train_ids_sha256=ids_fingerprint(["x"]),
        train_clips=1,
    )
    with pytest.raises(ValueError, match="different training set"):
        train_module.training_feature_stats(build_model("tcnn"), data)
    np.savez(
        path,
        mean=np.full(576, 2, np.float32),
        std=np.ones(576, np.float32),
        train_ids_sha256=ids_fingerprint(data.video_ids),
        train_clips=4,
    )
    mean, _ = train_module.training_feature_stats(build_model("tcnn"), data)
    assert (mean == 2).all()


def test_restore_rng_accepts_states_loaded_onto_training_device(tmp_path):
    from src.train import restore_rng, select_device

    device = select_device()
    generator = torch.Generator().manual_seed(7)
    expected = torch.rand(3, generator=generator)
    generator.manual_seed(7)
    torch.save(
        {"generator": generator.get_state(), "torch_rng": torch.get_rng_state()},
        tmp_path / "state.pt",
    )
    state = torch.load(tmp_path / "state.pt", map_location=device, weights_only=False)
    restored = torch.Generator()
    restore_rng(state, restored)
    assert torch.equal(torch.rand(3, generator=restored), expected)
