import argparse
import json
import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger("select_subset")

SPLIT_FRACTIONS = {"train": 0.5, "val": 0.0, "test": 1.0}
SPLIT_MINIMUMS = {"train": 3, "val": 1, "test": 1}


def signer_diverse_order(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # Round-robin over signers so earlier picks cover as many distinct signers as possible.
    shuffled = group.iloc[rng.permutation(len(group))]
    rank_within_signer = shuffled.groupby("signer_id").cumcount()
    return shuffled.assign(_rank=rank_within_signer.values).sort_values("_rank", kind="stable")


def quota(available: int, fraction: float, minimum: int) -> int:
    return min(available, max(minimum, int(available * fraction)))


def select(
    manifest: pd.DataFrame, fractions: dict[str, float], minimums: dict[str, int], seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chosen = []
    for (split, _), group in manifest.groupby(["split", "label"], sort=True):
        ordered = signer_diverse_order(group, rng)
        take = quota(len(group), fractions[split], minimums[split])
        chosen.append(ordered.head(take).drop(columns="_rank").assign(class_rank=range(take)))
    selected = pd.concat(chosen)
    # Extraction priority: every class's first sample before any class's second sample.
    order = rng.permutation(len(selected))
    selected = selected.assign(_tiebreak=order).sort_values(["class_rank", "_tiebreak"])
    return selected.drop(columns="_tiebreak").reset_index(drop=True).rename_axis("priority")


def shortfalls(
    manifest: pd.DataFrame, selected: pd.DataFrame, minimums: dict[str, int]
) -> list[dict]:
    rows = []
    for split, minimum in minimums.items():
        counts = (
            selected[selected["split"] == split]
            .groupby("label")
            .size()
            .reindex(range(config.NUM_CLASSES), fill_value=0)
        )
        for label, count in counts[counts < minimum].items():
            rows.append({"split": split, "label": int(label), "selected": int(count)})
    return rows


def summary(manifest: pd.DataFrame, selected: pd.DataFrame, minimums: dict) -> dict:
    report = {
        "strategy": {
            "fractions": SPLIT_FRACTIONS,
            "minimum_per_class": minimums,
            "method": (
                "per split and class: floor(n * fraction) clamped to [minimum, n], "
                "signer round-robin, seeded shuffle"
            ),
            "seed": config.SEED,
        }
    }
    for split in config.SPLITS:
        orig = manifest[manifest["split"] == split]
        sel = selected[selected["split"] == split]
        per_class = sel.groupby("label").size()
        report[split] = {
            "original": int(len(orig)),
            "selected": int(len(sel)),
            "classes": int(sel["label"].nunique()),
            "per_class_min": int(per_class.min()),
            "per_class_median": float(per_class.median()),
            "per_class_max": int(per_class.max()),
            "signers_original": int(orig["signer_id"].nunique()),
            "signers_selected": int(sel["signer_id"].nunique()),
        }
    report["total_selected"] = int(len(selected))
    report["classes_below_minimum"] = shortfalls(manifest, selected, minimums)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest = pd.read_csv(config.MANIFEST_CSV, dtype={"video_id": str})
    selected = select(manifest, SPLIT_FRACTIONS, SPLIT_MINIMUMS, args.seed)
    excluded = manifest[~manifest["video_id"].isin(selected["video_id"])].assign(
        reason="not_selected_subset_reduction"
    )
    selected.to_csv(config.REDUCED_MANIFEST_CSV)
    excluded[["video_id", "label", "gloss", "split", "signer_id", "reason"]].to_csv(
        config.REDUCED_EXCLUDED_CSV, index=False
    )
    report = summary(manifest, selected, SPLIT_MINIMUMS)
    (config.REPORT_DIR / "selection_report.json").write_text(json.dumps(report, indent=2))
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
