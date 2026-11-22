"""Check a submission bundle against the original evaluation artifacts (read-only)."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".jsonl"}
IDENTICAL = [
    "models/exported/model_fp32.onnx",
    "models/exported/model_int8.onnx",
    "models/checkpoints/tcnn/best.pt",
    "data/processed/preprocess_config.json",
    "data/processed/feature_stats.npz",
    "data/processed/manifest.csv",
    "data/processed/final_manifest.csv",
] + [f"src/{p.name}" for p in (ROOT / "src").glob("*.py") if p.name != "model_experimental.py"]
DEV_ONLY = re.compile(r"^(pytest|ruff|ipykernel|jupyter|black|mypy)\b", re.MULTILINE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stored_predictions() -> dict[str, tuple[int, list[int], float]]:
    with np.load(ROOT / "data/processed/test.npz", allow_pickle=True) as data:
        features = data["features"].astype(np.float32)
        video_ids = [str(v) for v in data["video_ids"]]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / "models/exported/model_fp32.onnx"), options)
    logits = np.concatenate(
        [
            session.run(None, {"keypoints": features[i : i + 256]})[0]
            for i in range(0, len(features), 256)
        ]
    )
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    top5 = np.argsort(-probs, 1)[:, :5]
    return {
        v: (int(t[0]), t.tolist(), float(p[t[0]]))
        for v, t, p in zip(video_ids, top5, probs, strict=True)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    report: dict = {}

    report["identical_to_repo"] = {
        name: digest(bundle / name) == digest(ROOT / name)
        for name in IDENTICAL
        if name != "src/hud_demo.py"
    }
    leaks = []
    for path in bundle.rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            text = path.read_text(errors="ignore")
            if "/Users/" in text or str(ROOT) in text:
                leaks.append(str(path.relative_to(bundle)))
    report["files_with_absolute_local_paths"] = leaks
    report["dev_only_requirements"] = DEV_ONLY.findall((bundle / "requirements.txt").read_text())

    records = [json.loads(line) for line in (bundle / "predictions.jsonl").read_text().splitlines()]
    stored = stored_predictions()
    scored = [r for r in records if "error" not in r]
    labels = {r["video_id"]: r["label"] for r in records}
    top1_same = [r["top5"][0]["label"] == stored[r["video_id"]][0] for r in scored]
    top5_same = [[p["label"] for p in r["top5"]] == stored[r["video_id"]][1] for r in scored]
    conf_diff = [abs(r["confidence"] - stored[r["video_id"]][2]) for r in scored]
    report["predictions"] = {
        "records": len(records),
        "errors": len(records) - len(scored),
        "same_ids_as_evaluated_test_set": sorted(labels) == sorted(stored),
        "top1_agreement_with_stored_eval": float(np.mean(top1_same)),
        "top5_order_agreement": float(np.mean(top5_same)),
        "max_confidence_abs_diff": float(np.max(conf_diff)),
        "top1_from_predict": float(np.mean([r["correct_top1"] for r in scored])),
        "top1_from_stored_eval": float(np.mean([stored[v][0] == labels[v] for v in stored])),
        "disagreeing_video_ids": [
            r["video_id"] for r, same in zip(scored, top1_same, strict=True) if not same
        ],
    }
    print(json.dumps(report, indent=2))
    ok = (
        all(report["identical_to_repo"].values())
        and not leaks
        and not report["dev_only_requirements"]
        and report["predictions"]["errors"] == 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
