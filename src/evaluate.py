import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src import config
from src.augment import occlude, rotate
from src.dataset import SplitData, load_split
from src.model import build_model
from src.train import classification_metrics, predict_logits, select_device

log = logging.getLogger("evaluate")

OCCLUSION_FRACTION = 0.3
ROTATION = {"yaw_deg": 20.0, "roll_deg": 10.0}


def load_checkpoint(path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location=device, weights_only=False)
    model = build_model(state["arch"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def per_class_table(logits: torch.Tensor, labels: torch.Tensor, glosses: dict) -> pd.DataFrame:
    y_true, y_pred = labels.cpu().numpy(), logits.argmax(1).numpy()
    classes = np.arange(config.NUM_CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return pd.DataFrame(
        {
            "label": classes,
            "gloss": [glosses[c] for c in classes],
            "support": support,
            "correct": np.bincount(y_true[y_true == y_pred], minlength=config.NUM_CLASSES),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def top_confusions(logits: torch.Tensor, labels: torch.Tensor, glosses: dict) -> pd.DataFrame:
    y_true, y_pred = labels.cpu().numpy(), logits.argmax(1).numpy()
    wrong = y_true != y_pred
    pairs = pd.DataFrame({"true": y_true[wrong], "pred": y_pred[wrong]})
    counts = pairs.value_counts().rename("count").reset_index()
    counts["true_gloss"] = counts["true"].map(glosses)
    counts["pred_gloss"] = counts["pred"].map(glosses)
    return counts


def save_confusion(logits: torch.Tensor, labels: torch.Tensor, path: Path) -> None:
    matrix = confusion_matrix(
        labels.cpu().numpy(), logits.argmax(1).numpy(), labels=np.arange(config.NUM_CLASSES)
    )
    np.savez_compressed(path, confusion=matrix.astype(np.int32))


def variant(data: SplitData, features: torch.Tensor) -> SplitData:
    return SplitData(features, data.labels, data.video_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=config.CHECKPOINT_DIR / config.DEFAULT_ARCH / "best.pt"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = select_device()
    model = load_checkpoint(args.checkpoint, device)
    manifest = pd.read_csv(config.MANIFEST_CSV, dtype={"video_id": str})
    glosses = manifest.drop_duplicates("label").set_index("label")["gloss"].to_dict()

    rows, report = [], {"checkpoint": str(args.checkpoint)}
    for split in ("val", "test"):
        data = load_split(split).to(device)
        batch = len(data)
        variants = {
            "clean": data,
            "occluded": variant(
                data, occlude(data.features.cpu(), OCCLUSION_FRACTION, config.SEED).to(device)
            ),
            "rotated": variant(
                data,
                rotate(
                    data.features,
                    torch.full((batch,), ROTATION["yaw_deg"]),
                    torch.full((batch,), ROTATION["roll_deg"]),
                ),
            ),
        }
        results = {}
        for name, split_data in variants.items():
            logits = predict_logits(model, split_data)
            results[name] = classification_metrics(logits, split_data.labels)
            rows.append(
                {"split": split, "variant": name, "samples": len(split_data)} | results[name]
            )
            if split == "test" and name == "clean":
                per_class_table(logits, data.labels, glosses).to_csv(
                    config.REPORT_DIR / "test_per_class.csv", index=False
                )
                save_confusion(logits, data.labels, config.REPORT_DIR / "test_confusion.npz")
                top_confusions(logits, data.labels, glosses).head(200).to_csv(
                    config.REPORT_DIR / "test_top_confusions.csv", index=False
                )
        for name in ("occluded", "rotated"):
            results[f"{name}_top1_drop_pp"] = 100 * (
                results["clean"]["top1"] - results[name]["top1"]
            )
        report[split] = results

    per_class = pd.read_csv(config.REPORT_DIR / "test_per_class.csv")
    tested = per_class[per_class["support"] > 0]
    report["test_per_class_summary"] = {
        "classes_tested": int(len(tested)),
        "classes_with_any_correct": int((tested["correct"] > 0).sum()),
        "median_class_recall": float(tested["recall"].median()),
    }
    report["robustness_settings"] = {"occluded_hand_fraction": OCCLUSION_FRACTION} | ROTATION
    report["note"] = (
        "Official WLASL split is signer-dependent: 99.97% of test videos come "
        "from signers also present in training."
    )
    pd.DataFrame(rows).to_csv(config.RESULTS_CSV, index=False)
    (config.REPORT_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
