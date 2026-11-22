import argparse
import json
import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger("select_subset")

SPLIT_FRACTIONS = {"train": 0.5, "val": 0.0, "test": 1.0}
SPLIT_MINIMUMS = {"train": 3, "val": 1, "test": 1}
FINAL_PER_CLASS = {"train": 3, "val": 1, "test": 1}


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


def select_final(reduced: pd.DataFrame, per_class: dict[str, int]) -> pd.DataFrame:
    # First k signer-diverse clips per split and class, using the reduced manifest's fixed ranks.
    keep = reduced["class_rank"] < reduced["split"].map(per_class)
    return reduced[keep].sort_values("priority").reset_index(drop=True)


def final_exclusions(manifest: pd.DataFrame, reduced: pd.DataFrame, final: pd.DataFrame):
    excluded = manifest[~manifest["video_id"].isin(final["video_id"])].copy()
    in_reduced = excluded["video_id"].isin(reduced["video_id"])
    excluded["reason"] = np.where(
        in_reduced, "over_final_per_class_quota", "not_selected_subset_reduction"
    )
    return excluded[["video_id", "label", "gloss", "split", "signer_id", "reason"]]


def write_final(manifest: pd.DataFrame) -> dict:
    reduced = pd.read_csv(config.REDUCED_MANIFEST_CSV, dtype={"video_id": str})
    final = select_final(reduced, FINAL_PER_CLASS)
    expected = config.NUM_CLASSES * sum(FINAL_PER_CLASS.values())
    if len(final) != expected:
        raise ValueError(f"final subset has {len(final)} videos, expected {expected}")
    final.to_csv(config.FINAL_MANIFEST_CSV, index=False)
    excluded = final_exclusions(manifest, reduced, final)
    excluded.to_csv(config.FINAL_EXCLUDED_CSV, index=False)
    report = summary(manifest, final, FINAL_PER_CLASS)
    report["strategy"] = {
        "source": config.REDUCED_MANIFEST_CSV.name,
        "per_class": FINAL_PER_CLASS,
        "method": "first k clips per split and class by the reduced manifest's class_rank "
        "(signer round-robin, seed 42); official splits unchanged",
    }
    report["excluded_reasons"] = excluded["reason"].value_counts().to_dict()
    report["excluded_by_split"] = excluded.groupby("split").size().to_dict()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--final", action="store_true", help="derive the 10,000-video subset")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.final:
        manifest = pd.read_csv(config.MANIFEST_CSV, dtype={"video_id": str})
        report = write_final(manifest)
        (config.REPORT_DIR / "final_selection_report.json").write_text(json.dumps(report, indent=2))
        log.info(json.dumps(report, indent=2))
        return

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
