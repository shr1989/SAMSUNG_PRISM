import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.inspect_dataset import (
    load_class_list,
    load_master,
    load_nslt,
    probe_video,
    resolve_clip_bounds,
)


def test_load_class_list(tmp_path):
    path = tmp_path / "classes.txt"
    path.write_text("0\tbook\n1\tdrink\n\n")
    assert load_class_list(path) == {0: "book", 1: "drink"}


def test_load_master_and_nslt(tmp_path):
    instance = {
        "video_id": "00001",
        "split": "train",
        "signer_id": 3,
        "fps": 25,
        "frame_start": 1,
        "frame_end": -1,
    }
    master_path = tmp_path / "master.json"
    master_path.write_text(json.dumps([{"gloss": "book", "instances": [instance]}]))
    nslt_path = tmp_path / "nslt.json"
    nslt_path.write_text(json.dumps({"00001": {"subset": "train", "action": [0, 1, 40]}}))

    master, glosses = load_master(master_path)
    nslt = load_nslt(nslt_path)
    assert glosses == ["book"]
    assert master.loc[0, "master_label"] == 0
    assert nslt.loc[0].to_dict() == {
        "video_id": "00001",
        "split": "train",
        "label": 0,
        "nslt_start": 1,
        "nslt_end": 40,
    }


def test_resolve_clip_bounds():
    df = pd.DataFrame(
        {"nslt_start": [1, 1, 5], "nslt_end": [60, 300, 40], "num_frames": [60, 120, 120]}
    )
    out = resolve_clip_bounds(df)
    assert out["clip_start"].tolist() == [0, 0, 4]
    assert out["clip_end"].tolist() == [60, 120, 40]


def test_probe_missing_video(tmp_path):
    result = probe_video(tmp_path / "missing.mp4")
    assert not result["exists"] and not result["readable"]


def test_probe_unreadable_video(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(np.random.default_rng(0).bytes(256))
    result = probe_video(path)
    assert result["exists"] and not result["readable"]


@pytest.mark.skipif(not config.MANIFEST_CSV.exists(), reason="manifest not generated")
def test_manifest_integrity():
    manifest = pd.read_csv(config.MANIFEST_CSV, dtype={"video_id": str})
    assert manifest["video_id"].is_unique
    assert manifest["label"].between(0, config.NUM_CLASSES - 1).all()
    assert manifest["label"].nunique() == config.NUM_CLASSES
    assert set(manifest["split"]) == set(config.SPLITS)
    assert (manifest["clip_end"] > manifest["clip_start"]).all()
    for split in ("val", "test"):
        assert not set(manifest[manifest["split"] == split]["video_id"]) & set(
            manifest[manifest["split"] == "train"]["video_id"]
        )


def test_select_final_takes_first_ranks_per_split():
    from src.select_subset import select_final

    reduced = pd.DataFrame(
        {
            "priority": range(8),
            "video_id": [str(i) for i in range(8)],
            "split": ["train"] * 4 + ["val"] * 2 + ["test"] * 2,
            "label": [0] * 8,
            "class_rank": [0, 1, 2, 3, 0, 1, 0, 1],
        }
    )
    final = select_final(reduced, {"train": 3, "val": 1, "test": 1})
    assert final["video_id"].tolist() == ["0", "1", "2", "4", "6"]


@pytest.mark.skipif(not config.FINAL_MANIFEST_CSV.exists(), reason="final manifest not generated")
def test_final_manifest_integrity():
    final = pd.read_csv(config.FINAL_MANIFEST_CSV, dtype={"video_id": str})
    full = pd.read_csv(config.MANIFEST_CSV, dtype={"video_id": str}).set_index("video_id")
    assert len(final) == 10_000 and final["video_id"].is_unique
    counts = final.groupby(["split", "label"]).size().unstack(0)
    assert counts.shape == (config.NUM_CLASSES, 3)
    assert (
        (counts["train"] == 3).all() and (counts["val"] == 1).all() and (counts["test"] == 1).all()
    )
    assert (full.loc[final["video_id"], "split"].to_numpy() == final["split"].to_numpy()).all()
    assert (full.loc[final["video_id"], "label"].to_numpy() == final["label"].to_numpy()).all()
