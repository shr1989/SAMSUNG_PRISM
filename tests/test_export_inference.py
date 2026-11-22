import numpy as np
import pytest
import torch

from src import config
from src.export_quantize import export_onnx, quantize_int8, run_onnx, session
from src.inference import Recognizer
from src.model import NUM_FEATURES, build_model

SHAPE = (config.NUM_FRAMES, config.NUM_LANDMARKS, config.COORDS)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = build_model("cnn_gru").eval()
    model.set_feature_stats(torch.rand(NUM_FEATURES), torch.rand(NUM_FEATURES) + 0.5)
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
    out = run_onnx(session(int8), calibration)
    assert out.shape == (8, config.NUM_CLASSES) and np.isfinite(out).all()


@pytest.mark.skipif(not config.PREPROCESS_CONFIG_JSON.exists(), reason="preprocess not run")
def test_classify_returns_structured_predictions(exported):
    _, path = exported
    recognizer = Recognizer(path)
    keypoints = np.random.default_rng(0).uniform(0.3, 0.7, (25, 75, 3)).astype(np.float32)
    predictions, timing = recognizer.classify(keypoints)
    assert len(predictions) == 5
    assert predictions[0]["word"].isupper()
    confidences = [p["confidence"] for p in predictions]
    assert confidences == sorted(confidences, reverse=True) and 0 < sum(confidences) <= 1
    assert set(timing) == {"preprocess_ms", "model_ms"}
