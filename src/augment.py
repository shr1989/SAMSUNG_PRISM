import math

import torch

from src import config


def _mirror_index() -> list[int]:
    index = list(range(config.NUM_LANDMARKS))
    for a, b in config.POSE_MIRROR_PAIRS:
        index[a], index[b] = b, a
    left = range(config.LEFT_HAND_SLICE.start, config.LEFT_HAND_SLICE.stop)
    right = range(config.RIGHT_HAND_SLICE.start, config.RIGHT_HAND_SLICE.stop)
    for a, b in zip(left, right, strict=True):
        index[a], index[b] = b, a
    return index


MIRROR_INDEX = _mirror_index()


def presence_mask(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().sum(dim=-1, keepdim=True) > 0).to(x.dtype)


def mirror(x: torch.Tensor) -> torch.Tensor:
    # Reflect across the body midline and swap left/right landmark identities.
    out = x[:, :, MIRROR_INDEX].clone()
    out[..., 0] = -out[..., 0]
    return out


def rotation_matrices(yaw: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    cy, sy, cr, sr = yaw.cos(), yaw.sin(), roll.cos(), roll.sin()
    zeros, ones = torch.zeros_like(yaw), torch.ones_like(yaw)
    yaw_m = torch.stack([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], -1).view(-1, 3, 3)
    roll_m = torch.stack([cr, -sr, zeros, sr, cr, zeros, zeros, zeros, ones], -1).view(-1, 3, 3)
    return roll_m @ yaw_m


def rotate(x: torch.Tensor, yaw_deg: torch.Tensor, roll_deg: torch.Tensor) -> torch.Tensor:
    mask = presence_mask(x)
    rot = rotation_matrices(yaw_deg * math.pi / 180, roll_deg * math.pi / 180).to(x)
    return torch.einsum("btlc,bdc->btld", x, rot) * mask


def temporal_resample(x: torch.Tensor, generator: torch.Generator, min_keep: float) -> torch.Tensor:
    batch, frames = x.shape[:2]
    keep = min_keep + (1 - min_keep) * torch.rand(batch, generator=generator)
    start = (1 - keep) * torch.rand(batch, generator=generator)
    grid = torch.linspace(0, 1, frames)
    positions = (start[:, None] + keep[:, None] * grid) * (frames - 1)
    index = positions.round().long().clamp(0, frames - 1).to(x.device)
    return x[torch.arange(batch, device=x.device)[:, None], index]


def hide_hand(
    x: torch.Tensor, fraction: torch.Tensor, apply: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    batch, frames = x.shape[:2]
    span = (fraction * frames).round().long()
    start = (torch.rand(batch, generator=generator) * (frames - span + 1)).long()
    t = torch.arange(frames)[None]
    hidden_frames = (t >= start[:, None]) & (t < (start + span)[:, None]) & apply[:, None]
    use_left = torch.rand(batch, generator=generator) < 0.5
    landmark = torch.zeros(batch, config.NUM_LANDMARKS, dtype=torch.bool)
    landmark[use_left, config.LEFT_HAND_SLICE] = True
    landmark[~use_left, config.RIGHT_HAND_SLICE] = True
    drop = (hidden_frames[:, :, None] & landmark[:, None]).to(x.device)
    return x.masked_fill(drop[..., None], 0.0)


def occlude(x: torch.Tensor, fraction: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    batch = x.shape[0]
    return hide_hand(
        x, torch.full((batch,), fraction), torch.ones(batch, dtype=torch.bool), generator
    )


def augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    batch = x.shape[0]

    def uniform(low: float, high: float, size=(batch,)) -> torch.Tensor:
        return low + (high - low) * torch.rand(size, generator=generator)

    x = temporal_resample(x, generator, min_keep=0.7)
    mask = presence_mask(x)
    flip = (torch.rand(batch, generator=generator) < 0.3).to(x.device)
    x = torch.where(flip[:, None, None, None], mirror(x), x)
    mask = torch.where(flip[:, None, None, None], mask[:, :, MIRROR_INDEX], mask)
    x = rotate(x, uniform(-20, 20), uniform(-10, 10))
    scale = uniform(0.9, 1.1).to(x)[:, None, None, None]
    shift = torch.cat([uniform(-0.05, 0.05, (batch, 2)), torch.zeros(batch, 1)], -1).to(x)
    x = x * scale + shift[:, None, None]
    x = x + 0.01 * torch.randn(x.shape, generator=generator).to(x)
    x = x * mask
    x = hide_hand(x, uniform(0.1, 0.4), torch.rand(batch, generator=generator) < 0.3, generator)
    dropout = (torch.rand(x.shape[:3], generator=generator) < 0.02).to(x.device)
    return x.masked_fill(dropout[..., None], 0.0)
