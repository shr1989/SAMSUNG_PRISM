"""Read-only confidence-calibration analysis of the existing FP32 test predictions."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main(predictions: Path) -> None:
    records = [json.loads(line) for line in predictions.read_text().splitlines()]
    confidence = np.array([r["confidence"] for r in records])
    correct = np.array([r["correct_top1"] for r in records], dtype=float)
    edges = np.linspace(0, 1, 11)
    bins = []
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            gap = abs(confidence[mask].mean() - correct[mask].mean())
            ece += mask.mean() * gap
            bins.append(
                {
                    "range": [round(low, 1), round(high, 1)],
                    "clips": int(mask.sum()),
                    "mean_confidence": float(confidence[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    report = {
        "source": str(predictions.relative_to(ROOT))
        if predictions.is_relative_to(ROOT)
        else predictions.name,
        "clips": len(records),
        "top1_accuracy": float(correct.mean()),
        "mean_top1_confidence": float(confidence.mean()),
        "expected_calibration_error_10_bins": float(ece),
        "bins": bins,
        "note": "Analysis of existing test predictions only; nothing was tuned or recalibrated.",
    }
    out = ROOT / "results" / "reports" / "calibration_analysis.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "bins"}, indent=2))
    for b in bins:
        print(b)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
