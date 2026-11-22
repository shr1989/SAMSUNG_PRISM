from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from src import config


@dataclass
class SplitData:
    features: torch.Tensor
    labels: torch.Tensor
    video_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def to(self, device: torch.device) -> "SplitData":
        return SplitData(self.features.to(device), self.labels.to(device), self.video_ids)


def load_split(split: str) -> SplitData:
    with np.load(config.PROCESSED_DIR / f"{split}.npz", allow_pickle=True) as data:
        features = torch.from_numpy(data["features"].astype(np.float32))
        labels = torch.from_numpy(data["labels"].astype(np.int64))
        video_ids = data["video_ids"]
    expected = (config.NUM_FRAMES, config.NUM_LANDMARKS, config.COORDS)
    if tuple(features.shape[1:]) != expected:
        raise ValueError(f"{split}: expected (N, {expected}), got {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise ValueError(f"{split}: non-finite features")
    return SplitData(features, labels, video_ids)


def check_disjoint(*splits: SplitData) -> None:
    seen: set[str] = set()
    for split in splits:
        ids = set(split.video_ids.tolist())
        if ids & seen:
            raise ValueError("video IDs shared between splits")
        seen |= ids


def batches(
    data: SplitData, batch_size: int, shuffle: bool, generator: torch.Generator | None = None
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    order = torch.randperm(len(data), generator=generator) if shuffle else torch.arange(len(data))
    order = order.to(data.labels.device)
    for start in range(0, len(data), batch_size):
        index = order[start : start + batch_size]
        yield data.features[index], data.labels[index]
