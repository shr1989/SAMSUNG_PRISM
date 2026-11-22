import numpy as np
import pytest
import torch

from src import config
from src.export_quantize import export_onnx, quantize_int8, run_onnx, session
from src.inference import Recognizer
from src.model import build_model
from src.preprocess import PreprocessConfig

SHAPE = (config.NUM_FRAMES, config.NUM_LANDMARKS, config.COORDS)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = build_model(config.DEFAULT_ARCH).eval()
    n = model.num_features
    model.set_feature_stats(torch.rand(n), torch.rand(n) + 0.5)
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_onnx(model, path)
    return model, path


def test_onnx_matches_torch_with_dynamic_batch(exported):
    model, path = exported
    x = torch.randn(3, *SHAPE)
    with torch.no_grad():
        expected = model(x).numpy()
    actual = run_onnx(session(path), x.numpy())
    assert actual.shape == (3, config.NUM_CLASSES)
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_int8_model_runs(exported, tmp_path):
    _, path = exported
    int8 = tmp_path / "int8.onnx"
    calibration = np.random.default_rng(0).normal(size=(8, *SHAPE)).astype(np.float32)
    ops = quantize_int8(path, int8, calibration)
    assert ops["quantize_linear_nodes"] > 0
    assert ops["int8_compute_ops"].get("Conv") == 2 and ops["fully_int8_compute"]
    out = run_onnx(session(int8), calibration)
    assert out.shape == (8, config.NUM_CLASSES) and np.isfinite(out).all()


@pytest.mark.skipif(not config.MANIFEST_CSV.exists(), reason="manifest not generated")
def test_classify_returns_structured_predictions(exported, tmp_path):
    _, path = exported
    PreprocessConfig().save(tmp_path / "preprocess.json")
    recognizer = Recognizer(path, tmp_path / "preprocess.json")
    keypoints = np.random.default_rng(0).uniform(0.3, 0.7, (25, 75, 3)).astype(np.float32)
    predictions, timing = recognizer.classify(keypoints)
    assert len(predictions) == 5
    assert predictions[0]["word"].isupper()
    confidences = [p["confidence"] for p in predictions]
    assert confidences == sorted(confidences, reverse=True) and 0 < sum(confidences) <= 1
    assert set(timing) == {"preprocess_ms", "model_ms"}


def test_recognizer_rejects_frame_mismatch(exported, tmp_path):
    _, path = exported
    PreprocessConfig(num_frames=32).save(tmp_path / "preprocess.json")
    with pytest.raises(ValueError, match="frames"):
        Recognizer(path, tmp_path / "preprocess.json")
