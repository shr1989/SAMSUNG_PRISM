"""Write predictions.jsonl (one line per held-out clip) by calling src.main.predict on videos."""

import argparse
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from src import config
from src.main import predict

log = logging.getLogger("make_predictions")

_args: dict = {}


def _init(videos_dir: str, model: str) -> None:
    _args.update(videos_dir=Path(videos_dir), model=model)


def _run(row: dict) -> dict:
    record = {
        "video_id": row["video_id"],
        "label": int(row["label"]),
        "true_word": str(row["gloss"]).upper(),
    }
    try:
        result = predict(
            _args["videos_dir"] / f"{row['video_id']}.mp4",
            int(row["clip_start"]),
            int(row["clip_end"]),
            _args["model"],
        )
    except (OSError, ValueError) as error:
        return record | {"error": str(error)}
    top5 = [
        {"word": p["word"], "label": p["label"], "confidence": p["confidence"]}
        for p in result["top5"]
    ]
    return record | {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "top5": top5,
        "correct_top1": top5[0]["label"] == record["label"],
        "correct_top5": record["label"] in [p["label"] for p in top5],
        "latency_ms": result["latency_ms"],
        "frames_used": result["frames_used"],
        "model": result["model"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=Path, default=config.VIDEO_DIR)
    parser.add_argument("--manifest", type=Path, default=config.FINAL_MANIFEST_CSV)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="fp32")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=config.ROOT / "predictions.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.videos_dir.is_dir():
        raise SystemExit(f"videos directory not found: {args.videos_dir}")

    manifest = pd.read_csv(args.manifest, dtype={"video_id": str})
    rows = manifest[manifest["split"] == args.split].sort_values("video_id")
    if args.limit:
        rows = rows.head(args.limit)
    with Pool(args.workers, _init, (str(args.videos_dir), args.model)) as pool:
        records = pool.map(_run, rows.to_dict("records"), chunksize=4)

    with args.out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(records, indent=1))
    scored = [r for r in records if "error" not in r]
    summary = {
        "clips": len(records),
        "errors": len(records) - len(scored),
        "top1": sum(r["correct_top1"] for r in scored) / max(1, len(scored)),
        "top5": sum(r["correct_top5"] for r in scored) / max(1, len(scored)),
        "split": args.split,
        "manifest": args.manifest.name,
        "model": args.model,
    }
    log.info(json.dumps(summary))
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
