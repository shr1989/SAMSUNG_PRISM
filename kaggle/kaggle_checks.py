import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

REFERENCE = Path(__file__).with_name("reference.json")
GROUPS = {
    "pose": slice(0, config.NUM_POSE),
    "left_hand": config.LEFT_HAND_SLICE,
    "right_hand": config.RIGHT_HAND_SLICE,
}


def split_hash(path: Path) -> dict:
    manifest = pd.read_csv(path, dtype={"video_id": str})
    rows = sorted(
        f"{v},{s},{label}"
        for v, s, label in zip(manifest.video_id, manifest.split, manifest.label, strict=True)
    )
    return {"rows": len(rows), "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest()}


MANIFESTS = {"manifest": config.MANIFEST_CSV, "reduced_manifest": config.REDUCED_MANIFEST_CSV}


def check_manifests(keys: list[str]) -> bool:
    reference = json.loads(REFERENCE.read_text())
    ok = True
    for key in keys:
        path = MANIFESTS[key]
        actual = split_hash(path)
        match = actual == reference[key]
        ok &= match
        print(
            f"{key}: {actual['rows']} rows, splits/labels "
            f"{'MATCH' if match else 'DIFFER'} reference"
        )
    return ok


def check_parity(rate_tolerance: float, xy_tolerance: float) -> bool:
    from src.extract_keypoints import create_holistic, detection_rates, extract_video

    reference = json.loads(REFERENCE.read_text())
    ok = True
    for ref in reference["reference_videos"]:
        with create_holistic() as holistic:
            keypoints = extract_video(
                holistic,
                config.VIDEO_DIR / f"{ref['video_id']}.mp4",
                ref["clip_start"],
                ref["clip_end"],
                reference["frame_stride"],
            )
        rates = dict(zip(GROUPS, detection_rates(keypoints), strict=True))
        frames_ok = len(keypoints) == ref["frames"]
        print(f"video {ref['video_id']}: frames {len(keypoints)} (reference {ref['frames']})")
        ok &= frames_ok
        for group, part in GROUPS.items():
            rate_diff = abs(rates[group] - ref["detection_rate"][group])
            xy = np.nanmean(keypoints[:, part, :2], axis=(0, 1))
            xy_diff = float(np.nanmax(np.abs(xy - np.asarray(ref["mean_xy"][group]))))
            group_ok = rate_diff <= rate_tolerance and (
                math.isnan(xy_diff) or xy_diff <= xy_tolerance
            )
            ok &= group_ok
            print(
                f"  {group:10s} detection {rates[group]:.3f} vs {ref['detection_rate'][group]:.3f}"
                f"  mean-xy diff {xy_diff:.4f}  {'ok' if group_ok else 'OUT OF TOLERANCE'}"
            )
    return ok


def wrist_distances(keypoints: np.ndarray, hand_wrist: int, pose_wrist: int) -> np.ndarray:
    both = ~np.isnan(keypoints[:, hand_wrist, 0]) & ~np.isnan(keypoints[:, pose_wrist, 0])
    return np.linalg.norm(
        keypoints[both, hand_wrist, :2] - keypoints[both, pose_wrist, :2], axis=-1
    )


def check_ordering(arrays: list[np.ndarray]) -> bool:
    # Each hand's wrist must sit nearer the same-side pose wrist than the opposite one.
    left_wrist, right_wrist = config.LEFT_HAND_SLICE.start, config.RIGHT_HAND_SLICE.start
    pairs = {
        "left hand": (left_wrist, config.LEFT_WRIST, config.RIGHT_WRIST),
        "right hand": (right_wrist, config.RIGHT_WRIST, config.LEFT_WRIST),
    }
    ok = True
    for name, (hand, same, other) in pairs.items():
        same_d = np.concatenate([wrist_distances(k, hand, same) for k in arrays])
        other_d = np.concatenate([wrist_distances(k, hand, other) for k in arrays])
        if not len(same_d):
            print(f"  {name}: not detected in sample, ordering not testable")
            continue
        correct = float(np.median(same_d)) < float(np.median(other_d))
        ok &= correct
        print(
            f"  {name}: median distance to same-side pose wrist {np.median(same_d):.3f}, "
            f"opposite {np.median(other_d):.3f} -> {'ordering ok' if correct else 'SWAPPED'}"
        )
    return ok


def check_samples(directory: Path) -> bool:
    manifest = pd.read_csv(config.MANIFEST_CSV, dtype={"video_id": str}).set_index("video_id")
    files = sorted(directory.glob("*.npy"))
    arrays = [np.load(path) for path in files]
    problems = []
    missing_hand_frames = 0
    for path, keypoints in zip(files, arrays, strict=True):
        row = manifest.loc[path.stem]
        expected = len(range(row.clip_start, row.clip_end, config.FRAME_STRIDE))
        if keypoints.dtype != np.float32 or keypoints.shape[1:] != (
            config.NUM_LANDMARKS,
            config.COORDS,
        ):
            problems.append(f"{path.stem}: bad shape/dtype {keypoints.shape} {keypoints.dtype}")
        elif not 0 < len(keypoints) <= expected:
            problems.append(f"{path.stem}: {len(keypoints)} frames, expected {expected}")
        for group, part in GROUPS.items():
            nan = np.isnan(keypoints[:, part])
            if not (nan.all(axis=(1, 2)) | ~nan.any(axis=(1, 2))).all():
                problems.append(f"{path.stem}: partial NaN in {group}")
            if group != "pose":
                missing_hand_frames += int(nan.all(axis=(1, 2)).sum())
    frames = sum(len(k) for k in arrays)
    print(f"{len(files)} arrays, {frames} frames, per-frame shape (75, 3), float32 verified")
    print(f"missing-hand frames stored as all-NaN groups: {missing_hand_frames}")
    ordering_ok = check_ordering(arrays)
    print(f"{len(problems)} problems")
    for problem in problems:
        print("  " + problem)
    return bool(files) and not problems and ordering_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    manifests = sub.add_parser("manifests")
    manifests.add_argument("keys", nargs="+", choices=sorted(MANIFESTS))
    parity = sub.add_parser("parity")
    parity.add_argument("--rate-tolerance", type=float, default=0.2)
    parity.add_argument("--xy-tolerance", type=float, default=0.03)
    samples = sub.add_parser("samples")
    samples.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "manifests":
        ok = check_manifests(args.keys)
    elif args.command == "parity":
        ok = check_parity(args.rate_tolerance, args.xy_tolerance)
    else:
        ok = check_samples(args.directory)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
