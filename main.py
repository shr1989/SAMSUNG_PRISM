import argparse
import json
from functools import lru_cache
from pathlib import Path

from src.inference import DEFAULT_MODEL, Recognizer


@lru_cache(maxsize=2)
def _recognizer(model_path: str) -> Recognizer:
    return Recognizer(Path(model_path))


def predict(video_path: str | Path, model_path: str | Path = DEFAULT_MODEL) -> dict:
    return _recognizer(str(model_path)).predict_video(Path(video_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(predict(args.video, args.model), indent=2))
