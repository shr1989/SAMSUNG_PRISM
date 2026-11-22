import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

from src import config

log = logging.getLogger("inspect_dataset")


def load_class_list(path: Path) -> dict[int, str]:
    classes = {}
    for line in path.read_text().splitlines():
        if line.strip():
            index, gloss = line.split("\t", 1)
            classes[int(index)] = gloss.strip()
    return classes


def load_master(path: Path) -> tuple[pd.DataFrame, list[str]]:
    entries = json.loads(path.read_text())
    rows = []
    for gloss_index, entry in enumerate(entries):
        for inst in entry["instances"]:
            rows.append(
                {
                    "video_id": inst["video_id"],
                    "gloss": entry["gloss"],
                    "master_label": gloss_index,
                    "master_split": inst["split"],
                    "signer_id": inst["signer_id"],
                    "fps": inst["fps"],
                    "frame_start": inst["frame_start"],
                    "frame_end": inst["frame_end"],
                }
            )
    return pd.DataFrame(rows), [e["gloss"] for e in entries]


def load_nslt(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    rows = [
        {
            "video_id": vid,
            "split": info["subset"],
            "label": info["action"][0],
            "nslt_start": info["action"][1],
            "nslt_end": info["action"][2],
        }
        for vid, info in data.items()
    ]
    return pd.DataFrame(rows)


def probe_video(path: Path) -> dict:
    result = {"exists": path.is_file(), "readable": False, "num_frames": 0}
    result.update(width=0, height=0, video_fps=0.0)
    if not result["exists"]:
        return result
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        result["readable"] = bool(ok and frame is not None)
        result["num_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result["video_fps"] = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    return result


def probe_all(video_ids: list[str], workers: int) -> pd.DataFrame:
    paths = [config.VIDEO_DIR / f"{vid}.mp4" for vid in video_ids]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        probes = list(pool.map(probe_video, paths, chunksize=64))
    return pd.DataFrame(probes).assign(video_id=video_ids)


def exclusion_reason(row: pd.Series) -> str | None:
    if pd.isna(row["split"]):
        return "missing_nslt_annotation"
    if not row["exists"]:
        return "missing_video"
    if not row["readable"]:
        return "unreadable_video"
    if row["num_frames"] < 2:
        return "too_few_frames"
    return None


def resolve_clip_bounds(df: pd.DataFrame, tolerance: int = 2) -> pd.DataFrame:
    # Most resized videos are pre-trimmed; trim only when the NSLT segment lies strictly inside.
    segment = df["nslt_end"] - df["nslt_start"] + 1
    inside = (df["nslt_end"] <= df["num_frames"]) & ((segment - df["num_frames"]).abs() > tolerance)
    df["clip_start"] = (df["nslt_start"] - 1).where(inside, 0).astype(int)
    df["clip_end"] = df["nslt_end"].where(inside, df["num_frames"]).astype(int)
    return df


def build_report(df: pd.DataFrame, classes: dict[int, str], master_glosses: list[str]) -> dict:
    valid = df[df["excluded_reason"].isna()]
    per_class = valid.groupby("label").size().reindex(range(config.NUM_CLASSES), fill_value=0)
    split_counts = {s: int((valid["split"] == s).sum()) for s in config.SPLITS}
    train_per_class = (
        valid[valid["split"] == "train"]
        .groupby("label")
        .size()
        .reindex(range(config.NUM_CLASSES), fill_value=0)
    )
    disk_ids = {p.stem for p in config.VIDEO_DIR.glob("*.mp4")}
    return {
        "annotated_instances": int(len(df)),
        "videos_on_disk": len(disk_ids),
        "videos_on_disk_without_annotation": len(disk_ids - set(df["video_id"])),
        "valid_videos": int(len(valid)),
        "excluded": dict(Counter(df["excluded_reason"].dropna())),
        "num_classes_class_list": len(classes),
        "num_classes_master": len(master_glosses),
        "class_list_matches_master": all(classes.get(i) == g for i, g in enumerate(master_glosses)),
        "nslt_label_matches_master": bool(
            (
                df.dropna(subset=["label"])["label"] == df.dropna(subset=["label"])["master_label"]
            ).all()
        ),
        "nslt_split_matches_master": bool(
            (
                df.dropna(subset=["split"])["split"] == df.dropna(subset=["split"])["master_split"]
            ).all()
        ),
        "classes_with_valid_samples": int((per_class > 0).sum()),
        "classes_without_train_samples": int((train_per_class == 0).sum()),
        "classes_without_test_samples": int(
            config.NUM_CLASSES - valid[valid["split"] == "test"]["label"].nunique()
        ),
        "split_counts": split_counts,
        "samples_per_class": {
            "min": int(per_class.min()),
            "max": int(per_class.max()),
            "mean": round(float(per_class.mean()), 2),
            "median": float(per_class.median()),
        },
        "imbalance_ratio_max_min": round(float(per_class.max() / max(per_class.min(), 1)), 2),
        "unique_signers": int(valid["signer_id"].nunique()),
        "frames": valid["num_frames"].describe().round(1).to_dict(),
        "resolutions": {
            f"{w}x{h}": int(n)
            for (w, h), n in valid.groupby(["width", "height"]).size().nlargest(5).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.ensure_dirs()

    classes = load_class_list(config.CLASS_LIST)
    master, master_glosses = load_master(config.MASTER_JSON)
    nslt = load_nslt(config.NSLT_JSON)

    duplicates = master[master["video_id"].duplicated(keep=False)]
    if not duplicates.empty:
        raise ValueError(
            f"Duplicate video IDs in master JSON: {sorted(duplicates['video_id'])[:10]}"
        )
    nslt_only = sorted(set(nslt["video_id"]) - set(master["video_id"]))
    master_only = sorted(set(master["video_id"]) - set(nslt["video_id"]))

    # NSLT-2000 defines the official splits and labels; master JSON adds metadata.
    df = nslt.merge(master, on="video_id", how="outer")
    df["gloss"] = df["gloss"].fillna(df["label"].map(lambda i: master_glosses[int(i)]))
    df["master_label"] = df["master_label"].fillna(df["label"])
    df["master_split"] = df["master_split"].fillna(df["split"])
    df["signer_id"] = df["signer_id"].fillna(-1).astype(int)
    df = df.merge(probe_all(df["video_id"].tolist(), args.workers), on="video_id")
    df["excluded_reason"] = df.apply(exclusion_reason, axis=1)

    report = build_report(df, classes, master_glosses)
    report["duplicate_video_ids"] = int(len(duplicates))
    report["nslt_ids_missing_from_master"] = nslt_only
    report["master_ids_missing_from_nslt"] = master_only
    report["class_list_mismatches"] = {
        i: {"class_list": classes.get(i), "master": g}
        for i, g in enumerate(master_glosses)
        if classes.get(i) != g
    }

    df[df["excluded_reason"].notna()][
        ["video_id", "gloss", "master_split", "excluded_reason"]
    ].to_csv(config.REPORT_DIR / "excluded_samples.csv", index=False)

    valid = df[df["excluded_reason"].isna()].copy()
    valid["label"] = valid["label"].astype(int)
    valid = resolve_clip_bounds(valid)
    report["clips_trimmed_to_nslt_segment"] = int(
        ((valid["clip_start"] > 0) | (valid["clip_end"] < valid["num_frames"])).sum()
    )
    valid.groupby(["label", "gloss", "split"]).size().unstack(fill_value=0).to_csv(
        config.REPORT_DIR / "class_counts.csv"
    )
    valid[
        [
            "video_id",
            "label",
            "gloss",
            "split",
            "signer_id",
            "num_frames",
            "video_fps",
            "clip_start",
            "clip_end",
            "width",
            "height",
        ]
    ].to_csv(config.MANIFEST_CSV, index=False)

    (config.REPORT_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2))
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
