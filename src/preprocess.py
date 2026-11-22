import json
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger("preprocess")

GROUPS = {
    "pose": slice(0, config.NUM_POSE),
    "left_hand": config.LEFT_HAND_SLICE,
    "right_hand": config.RIGHT_HAND_SLICE,
}


@dataclass
class PreprocessConfig:
    num_frames: int = config.NUM_FRAMES
    max_fill_gap: int = config.MAX_FILL_GAP
    clip_value: float = config.CLIP_VALUE
    frame_stride: int = config.FRAME_STRIDE
    dropped_pose_landmarks: list[int] = field(default_factory=lambda: list(config.DROPPED_POSE))
    landmark_order: list[str] = field(default_factory=lambda: list(config.LANDMARK_ORDER))

    def save(self, path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "PreprocessConfig":
        return cls(**json.loads(path.read_text()))


def _missing_runs(missing: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for t, flag in enumerate(missing):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            runs.append((start, t))
            start = None
    if start is not None:
        runs.append((start, len(missing)))
    return runs


def fill_missing(keypoints: np.ndarray, max_gap: int) -> np.ndarray:
    # Short interior gaps: linear interpolation. Short trailing gaps: hold last value.
    # Leading gaps and gaps longer than max_gap stay missing.
    out = keypoints.copy()
    frames = len(out)
    for part in GROUPS.values():
        missing = np.isnan(out[:, part, 0]).any(axis=1)
        for start, stop in _missing_runs(missing):
            if stop - start > max_gap or start == 0:
                continue
            before = out[start - 1, part]
            if stop == frames:
                out[start:stop, part] = before
                continue
            weights = (np.arange(start, stop) - (start - 1)) / (stop - (start - 1))
            after = out[stop, part]
            out[start:stop, part] = before + weights[:, None, None] * (after - before)
    return out


def shoulder_frame(keypoints: np.ndarray) -> tuple[np.ndarray, float]:
    left = keypoints[:, config.LEFT_SHOULDER]
    right = keypoints[:, config.RIGHT_SHOULDER]
    centre = (left + right) / 2
    width = np.linalg.norm(left[:, :2] - right[:, :2], axis=-1)
    valid = np.isfinite(width) & (width > config.MIN_SHOULDER_WIDTH)
    if not valid.any():
        return np.zeros(3, dtype=np.float32), 1.0
    # One clip-level reference keeps per-frame shoulder jitter out of the hand trajectories.
    return np.median(centre[valid], axis=0), float(np.median(width[valid]))


def normalise(keypoints: np.ndarray) -> np.ndarray:
    centre, width = shoulder_frame(keypoints)
    return (keypoints - centre) / width


def resample(keypoints: np.ndarray, num_frames: int) -> np.ndarray:
    # Linear interpolation between neighbouring frames; nearest frame where either is missing.
    positions = np.linspace(0, len(keypoints) - 1, num_frames)
    low = np.floor(positions).astype(int)
    high = np.minimum(low + 1, len(keypoints) - 1)
    weight = (positions - low)[:, None, None]
    blended = (1 - weight) * keypoints[low] + weight * keypoints[high]
    nearest = keypoints[positions.round().astype(int)]
    return np.where(np.isnan(blended), nearest, blended).astype(np.float32)


def preprocess(keypoints: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if keypoints.ndim != 3 or keypoints.shape[1:] != (config.NUM_LANDMARKS, config.COORDS):
        raise ValueError(f"expected (T, 75, 3), got {keypoints.shape}")
    if len(keypoints) == 0:
        raise ValueError("empty sequence")
    out = normalise(fill_missing(keypoints.astype(np.float32), cfg.max_fill_gap))
    out[:, cfg.dropped_pose_landmarks] = np.nan
    out = np.clip(out, -cfg.clip_value, cfg.clip_value)
    out = resample(out, cfg.num_frames)
    # Missing landmarks become exactly zero; the model reads all-zero groups as absent.
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_split(manifest: pd.DataFrame, ok_ids: set[str], split: str) -> tuple[list, pd.DataFrame]:
    rows = manifest[(manifest["split"] == split) & manifest["video_id"].isin(ok_ids)]
    arrays = [np.load(config.KEYPOINT_DIR / f"{vid}.npy") for vid in rows["video_id"]]
    return arrays, rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.ensure_dirs()

    manifest = pd.read_csv(config.TRAINING_MANIFEST_CSV, dtype={"video_id": str})
    status = pd.read_csv(config.EXTRACTION_STATUS_CSV, dtype={"video_id": str})
    status = status.drop_duplicates("video_id", keep="last")
    ok_ids = set(status.loc[status["status"] == "ok", "video_id"])
    ok_ids &= {p.stem for p in config.KEYPOINT_DIR.glob("*.npy")}

    cfg = PreprocessConfig()
    cfg.save(config.PREPROCESS_CONFIG_JSON)
    report = {"config": asdict(cfg)}
    for split in config.SPLITS:
        arrays, rows = load_split(manifest, ok_ids, split)
        features = np.stack([preprocess(seq, cfg) for seq in arrays])
        labels = rows["label"].to_numpy(dtype=np.int64)
        np.savez(
            config.PROCESSED_DIR / f"{split}.npz",
            features=features,
            labels=labels,
            video_ids=rows["video_id"].to_numpy(),
            signer_ids=rows["signer_id"].to_numpy(),
        )
        selected = int((manifest["split"] == split).sum())
        report[split] = {
            "selected": selected,
            "processed": len(rows),
            "missing_or_failed": selected - len(rows),
            "classes": int(np.unique(labels).size),
            "shape": list(features.shape),
            "finite": bool(np.isfinite(features).all()),
        }
    (config.REPORT_DIR / "preprocess_report.json").write_text(json.dumps(report, indent=2))
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
