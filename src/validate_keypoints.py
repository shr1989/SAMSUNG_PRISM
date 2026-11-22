import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger("validate_keypoints")

GROUPS = {
    "pose": slice(0, config.NUM_POSE),
    "left_hand": config.LEFT_HAND_SLICE,
    "right_hand": config.RIGHT_HAND_SLICE,
}


def expected_frames(clip_start: int, clip_end: int, stride: int = config.FRAME_STRIDE) -> int:
    return len(range(clip_start, clip_end, stride))


def array_problems(keypoints: np.ndarray, expected: int, status_frames: int | None) -> list[str]:
    if keypoints.dtype != np.float32 or keypoints.shape[1:] != (
        config.NUM_LANDMARKS,
        config.COORDS,
    ):
        return [f"shape/dtype {keypoints.shape} {keypoints.dtype}"]
    problems = []
    if not 0 < len(keypoints) <= expected:
        problems.append(f"{len(keypoints)} frames, expected 1..{expected}")
    if status_frames is not None and len(keypoints) != status_frames:
        problems.append(f"{len(keypoints)} frames, status records {status_frames}")
    if np.isinf(keypoints).any():
        problems.append("infinite values")
    for group, part in GROUPS.items():
        nan = np.isnan(keypoints[:, part])
        if not (nan.all(axis=(1, 2)) | ~nan.any(axis=(1, 2))).all():
            problems.append(f"partially missing {group}")
    return problems


def wrist_distances(keypoints: np.ndarray, hand_wrist: int, pose_wrist: int) -> np.ndarray:
    both = ~np.isnan(keypoints[:, hand_wrist, 0]) & ~np.isnan(keypoints[:, pose_wrist, 0])
    return np.linalg.norm(
        keypoints[both, hand_wrist, :2] - keypoints[both, pose_wrist, :2], axis=-1
    )


def ordering_summary(arrays: list[np.ndarray]) -> dict:
    # Each hand's wrist must sit nearer the same-side pose wrist than the opposite one.
    pairs = {
        "left_hand": (config.LEFT_HAND_SLICE.start, config.LEFT_WRIST, config.RIGHT_WRIST),
        "right_hand": (config.RIGHT_HAND_SLICE.start, config.RIGHT_WRIST, config.LEFT_WRIST),
    }
    summary = {}
    for name, (hand, same, other) in pairs.items():
        same_d = np.concatenate([wrist_distances(k, hand, same) for k in arrays])
        other_d = np.concatenate([wrist_distances(k, hand, other) for k in arrays])
        if not len(same_d):
            summary[name] = {"testable": False}
            continue
        summary[name] = {
            "median_same_side": float(np.median(same_d)),
            "median_opposite_side": float(np.median(other_d)),
            "ordering_ok": bool(np.median(same_d) < np.median(other_d)),
        }
    return summary


def validate(manifest: pd.DataFrame, keypoint_dir: Path, ordering_sample: int = 2000) -> dict:
    status_path = keypoint_dir / config.EXTRACTION_STATUS_CSV.name
    status = pd.read_csv(status_path, dtype={"video_id": str}, keep_default_na=False)
    status = status.drop_duplicates("video_id", keep="last").set_index("video_id")
    arrays_on_disk = {p.stem for p in keypoint_dir.glob("*.npy")}
    rows, ordering_arrays, missing_hand_frames, frames = [], [], 0, 0
    for row in manifest.itertuples():
        state = status["status"].get(row.video_id, "pending")
        record = {"video_id": row.video_id, "split": row.split, "label": row.label}
        if state == "failed":
            rows.append(record | {"state": "failed", "detail": status.loc[row.video_id, "error"]})
            continue
        if state != "ok":
            rows.append(record | {"state": "pending", "detail": ""})
            continue
        if row.video_id not in arrays_on_disk:
            rows.append(record | {"state": "corrupted", "detail": "status ok but array missing"})
            continue
        try:
            keypoints = np.load(keypoint_dir / f"{row.video_id}.npy")
        except (OSError, ValueError) as exc:
            rows.append(record | {"state": "corrupted", "detail": f"unreadable: {exc!r}"})
            continue
        problems = array_problems(
            keypoints,
            expected_frames(row.clip_start, row.clip_end),
            int(status.loc[row.video_id, "frames"]),
        )
        if problems:
            rows.append(record | {"state": "corrupted", "detail": "; ".join(problems)})
            continue
        rows.append(record | {"state": "completed", "detail": ""})
        frames += len(keypoints)
        hands = np.isnan(keypoints[:, config.NUM_POSE :, 0]).reshape(len(keypoints), 2, -1)
        missing_hand_frames += int(hands.all(axis=-1).sum())
        if len(ordering_arrays) < ordering_sample:
            ordering_arrays.append(keypoints)
    table = pd.DataFrame(rows)
    counts = table.groupby(["split", "state"]).size().unstack(fill_value=0)
    completed = table[table["state"] == "completed"]
    orphans = sorted(arrays_on_disk - set(status.index[status["status"] == "ok"]))
    return {
        "table": table,
        "report": {
            "manifest_videos": len(manifest),
            "counts": table["state"]
            .value_counts()
            .reindex(["completed", "pending", "failed", "corrupted"], fill_value=0)
            .to_dict(),
            "counts_by_split": counts.to_dict(orient="index"),
            "classes_with_completed_video": {
                split: int(completed[completed["split"] == split]["label"].nunique())
                for split in config.SPLITS
            },
            "arrays_without_ok_status": len(orphans),
            "temp_files": len(list(keypoint_dir.glob(".*tmp*"))),
            "frames_in_completed_arrays": frames,
            "missing_hand_slots_fraction": missing_hand_frames / max(2 * frames, 1),
            "landmark_ordering": ordering_summary(ordering_arrays) if ordering_arrays else {},
            "frame_stride": config.FRAME_STRIDE,
            "keypoint_dir": str(keypoint_dir),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=config.TRAINING_MANIFEST_CSV)
    parser.add_argument("--keypoint-dir", type=Path, default=config.KEYPOINT_DIR)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.ensure_dirs()

    manifest = pd.read_csv(args.manifest, dtype={"video_id": str})
    result = validate(manifest, args.keypoint_dir)
    name = args.name or args.manifest.stem
    report = result["report"] | {"manifest": args.manifest.name}
    problems = result["table"][result["table"]["state"].isin(["failed", "corrupted"])]
    problems.to_csv(config.REPORT_DIR / f"keypoint_problems_{name}.csv", index=False)
    (config.REPORT_DIR / f"keypoint_validation_{name}.json").write_text(
        json.dumps(report, indent=2)
    )
    log.info(json.dumps(report, indent=2))
    ordering_ok = all(v.get("ordering_ok", True) for v in report["landmark_ordering"].values())
    sys.exit(0 if report["counts"]["corrupted"] == 0 and ordering_ok else 1)


if __name__ == "__main__":
    main()
