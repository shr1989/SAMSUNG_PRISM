import logging

import torch
from torch import nn

from src import config

log = logging.getLogger("model")

POSE_KEEP = (0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
HAND_SIZE_REF = 9
NUM_STATIC = len(POSE_KEEP) * 2 + 2 * config.NUM_HAND * 3 + 4
NUM_FEATURES = 2 * NUM_STATIC + 2


def hand_presence(hand: torch.Tensor) -> torch.Tensor:
    return (hand.abs().sum(dim=(-1, -2)) > 0).to(hand.dtype)


def hand_local(hand: torch.Tensor) -> torch.Tensor:
    # Wrist-relative handshape, scaled by wrist-to-middle-knuckle distance.
    relative = hand - hand[:, :, :1]
    size = torch.linalg.vector_norm(relative[:, :, HAND_SIZE_REF, :2], dim=-1)
    scaled = relative / size.clamp_min(1e-3)[..., None, None]
    return scaled.clamp(-config.CLIP_VALUE, config.CLIP_VALUE).flatten(2)


def frame_features(x: torch.Tensor) -> torch.Tensor:
    batch, frames = x.shape[:2]
    pose = x[:, :, list(POSE_KEEP), :2].flatten(2)
    left, right = x[:, :, config.LEFT_HAND_SLICE], x[:, :, config.RIGHT_HAND_SLICE]
    left_present, right_present = hand_presence(left), hand_presence(right)
    static = torch.cat(
        [pose, hand_local(left), hand_local(right), left[:, :, 0, :2], right[:, :, 0, :2]], dim=-1
    )
    hand_dims = config.NUM_HAND * 3
    valid = torch.cat(
        [
            torch.ones(batch, frames, pose.shape[-1], dtype=x.dtype, device=x.device),
            left_present[..., None].expand(-1, -1, hand_dims),
            right_present[..., None].expand(-1, -1, hand_dims),
            left_present[..., None].expand(-1, -1, 2),
            right_present[..., None].expand(-1, -1, 2),
        ],
        dim=-1,
    )
    velocity = (static[:, 1:] - static[:, :-1]) * valid[:, 1:] * valid[:, :-1]
    velocity = torch.cat([torch.zeros_like(static[:, :1]), velocity], dim=1)
    flags = torch.stack([left_present, right_present], dim=-1)
    return torch.cat([static, velocity, flags], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel, stride=stride, padding=kernel // 2)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.residual = stride == 1 and in_dim == out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm(self.conv(x)))
        return x + out if self.residual else out


class SignClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        hidden: int = 192,
        temporal: int = 256,
        gru_hidden: int = 128,
        use_gru: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(NUM_FEATURES))
        self.register_buffer("feature_std", torch.ones(NUM_FEATURES))
        self.encoder = nn.Sequential(
            nn.Linear(NUM_FEATURES, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2)
        )
        self.temporal = nn.Sequential(
            ConvBlock(hidden, hidden, 5),
            ConvBlock(hidden, temporal, 5, stride=2),
            ConvBlock(temporal, temporal, 3),
        )
        self.gru = (
            nn.GRU(temporal, gru_hidden, batch_first=True, bidirectional=True) if use_gru else None
        )
        pooled = 2 * (2 * gru_hidden if use_gru else temporal)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(pooled, num_classes))

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1e-4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = (frame_features(x) - self.feature_mean) / self.feature_std
        hidden = self.temporal(self.encoder(features).transpose(1, 2)).transpose(1, 2)
        if self.gru is not None:
            hidden, _ = self.gru(hidden)
        pooled = torch.cat([hidden.mean(dim=1), hidden.amax(dim=1)], dim=-1)
        return self.head(pooled)


def build_model(arch: str) -> SignClassifier:
    if arch not in ("cnn_gru", "cnn"):
        raise ValueError(f"unknown architecture {arch}")
    return SignClassifier(use_gru=arch == "cnn_gru")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for arch in ("cnn_gru", "cnn"):
        net = build_model(arch).eval()
        out = net(torch.randn(64, config.NUM_FRAMES, config.NUM_LANDMARKS, config.COORDS))
        log.info("%s params=%d output=%s", arch, count_parameters(net), tuple(out.shape))
