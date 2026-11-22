import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from src import config
from src.dataset import load_split
from src.evaluate import load_checkpoint
from src.train import classification_metrics

log = logging.getLogger("export_quantize")

INPUT_NAME, OUTPUT_NAME = "keypoints", "logits"
CALIBRATION_CLIPS = 300


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    dummy = torch.zeros(1, config.NUM_FRAMES, config.NUM_LANDMARKS, config.COORDS)
    torch.onnx.export(
        model.cpu().eval(),
        (dummy,),
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=17,
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))


def session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_onnx(sess: ort.InferenceSession, features: np.ndarray, batch: int = 256) -> np.ndarray:
    outputs = [
        sess.run([OUTPUT_NAME], {INPUT_NAME: features[i : i + batch]})[0]
        for i in range(0, len(features), batch)
    ]
    return np.concatenate(outputs)


def latency_ms(sess: ort.InferenceSession, sample: np.ndarray, warmup=20, runs=200) -> dict:
    for _ in range(warmup):
        sess.run([OUTPUT_NAME], {INPUT_NAME: sample})
    times = []
    for _ in range(runs):
        began = time.perf_counter()
        sess.run([OUTPUT_NAME], {INPUT_NAME: sample})
        times.append(1000 * (time.perf_counter() - began))
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "mean": float(np.mean(times)),
        "warmup": warmup,
        "runs": runs,
    }


class TrainCalibration(CalibrationDataReader):
    def __init__(self, features: np.ndarray):
        self._iter = iter(features[:, None])

    def get_next(self):
        sample = next(self._iter, None)
        return None if sample is None else {INPUT_NAME: sample}


def quantize_int8(fp32_path: Path, int8_path: Path, calibration: np.ndarray) -> dict:
    prepared = int8_path.with_name("model_fp32_prep.onnx")
    quant_pre_process(str(fp32_path), str(prepared))
    quantize_static(
        str(prepared),
        str(int8_path),
        TrainCalibration(calibration),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    prepared.unlink()
    ops = Counter(node.op_type for node in onnx.load(str(int8_path)).graph.node)
    return {
        "quantize_linear_nodes": ops["QuantizeLinear"],
        "dequantize_linear_nodes": ops["DequantizeLinear"],
        "fp32_gru_nodes": ops["GRU"],
    }


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    return {
        "max_abs_logit_diff": float(np.abs(reference - candidate).max()),
        "argmax_agreement": float((reference.argmax(1) == candidate.argmax(1)).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=config.CHECKPOINT_DIR / "cnn_gru" / "best.pt"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.ensure_dirs()
    torch.set_num_threads(4)

    model = load_checkpoint(args.checkpoint, torch.device("cpu"))
    fp32_path, int8_path = (
        config.EXPORT_DIR / "model_fp32.onnx",
        config.EXPORT_DIR / "model_int8.onnx",
    )
    export_onnx(model, fp32_path)

    train = load_split("train")
    rng = np.random.default_rng(config.SEED)
    # Calibration draws from the training split only.
    calibration = train.features.numpy()[rng.choice(len(train), CALIBRATION_CLIPS, replace=False)]
    report = {"int8_quantization": quantize_int8(fp32_path, int8_path, calibration)}

    fp32, int8 = session(fp32_path), session(int8_path)
    sample = load_split("val").features.numpy()[:1]
    for split in ("val", "test"):
        data = load_split(split)
        x = data.features.numpy()
        with torch.no_grad():
            torch_logits = torch.cat(
                [model(data.features[i : i + 256]) for i in range(0, len(data), 256)]
            ).numpy()
        fp32_logits, int8_logits = run_onnx(fp32, x), run_onnx(int8, x)
        report[split] = {
            "torch_fp32": classification_metrics(torch.from_numpy(torch_logits), data.labels),
            "onnx_fp32": classification_metrics(torch.from_numpy(fp32_logits), data.labels),
            "onnx_int8": classification_metrics(torch.from_numpy(int8_logits), data.labels),
            "onnx_fp32_vs_torch": compare(torch_logits, fp32_logits),
            "onnx_int8_vs_fp32": compare(fp32_logits, int8_logits),
        }
    report["latency_batch1_ms"] = {
        "onnx_fp32": latency_ms(fp32, sample),
        "onnx_int8": latency_ms(int8, sample),
    }
    report["size_mb"] = {
        "onnx_fp32": fp32_path.stat().st_size / 2**20,
        "onnx_int8": int8_path.stat().st_size / 2**20,
    }
    report["runtime"] = {
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "hardware": "Apple M4 MacBook Air (CPU)",
    }
    report["input"] = {
        "name": INPUT_NAME,
        "shape": ["batch", config.NUM_FRAMES, config.NUM_LANDMARKS, config.COORDS],
    }
    (config.REPORT_DIR / "export_report.json").write_text(json.dumps(report, indent=2))
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
