import logging

import torch
from torch import nn

from src import config

log = logging.getLogger("model")

HAND_SIZE_REF = 9
HAND_DIMS = config.NUM_HAND * config.COORDS
BODY_DIMS = config.NUM_LANDMARKS * config.COORDS
NUM_FEATURES_576 = BODY_DIMS + 2 * HAND_DIMS + BODY_DIMS


def landmark_presence(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().sum(dim=-1) > 0).to(x.dtype)


def masked_velocity(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    velocity = (values[:, 1:] - values[:, :-1]) * valid[:, 1:] * valid[:, :-1]
    return torch.cat([torch.zeros_like(values[:, :1]), velocity], dim=1)


def hand_upright(hand: torch.Tensor) -> torch.Tensor:
    # Wrist-relative, rotated so wrist->middle knuckle points up (-y), unit length.
    relative = hand - hand[:, :, :1]
    axis = relative[:, :, HAND_SIZE_REF, :2]
    size = torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-3)
    ux, uy = (axis / size).unbind(-1)
    x, y, z = relative.unbind(-1)
    rotated_x = -uy[..., None] * x + ux[..., None] * y
    rotated_y = -ux[..., None] * x - uy[..., None] * y
    upright = torch.stack([rotated_x, rotated_y, z], dim=-1) / size[..., None]
    return upright.clamp(-config.CLIP_VALUE, config.CLIP_VALUE).flatten(2)


def features_576(x: torch.Tensor) -> torch.Tensor:
    body = x.flatten(2)
    left, right = x[:, :, config.LEFT_HAND_SLICE], x[:, :, config.RIGHT_HAND_SLICE]
    hands = torch.cat([hand_upright(left), hand_upright(right)], dim=-1)
    valid = landmark_presence(x)[..., None].expand_as(x).flatten(2)
    return torch.cat([body, hands, masked_velocity(body, valid)], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int, stride: int = 1, act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel, stride=stride, padding=kernel // 2)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = act()
        self.residual = stride == 1 and in_dim == out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm(self.conv(x)))
        return x + out if self.residual else out


class ResidualDownBlock(ConvBlock):
    def __init__(self, dim: int, kernel: int):
        super().__init__(dim, dim, kernel, stride=2, act=nn.ReLU)
        self.shortcut = nn.AvgPool1d(2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.act(self.norm(self.conv(x)))


class FeatureStandardised(nn.Module):
    num_features: int

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(self.num_features))
        self.register_buffer("feature_std", torch.ones(self.num_features))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1e-4))

    def standardised_features(self, x: torch.Tensor) -> torch.Tensor:
        return (self.features(x) - self.feature_mean) / self.feature_std


class TemporalCNN(FeatureStandardised):
    """Baseline: (B, 64, 75, 3) landmarks -> (B, 64, 576) features -> 1D-CNN -> (B, 2000)."""

    num_features = NUM_FEATURES_576

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        hidden: int = 192,
        kernel: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(self.num_features, hidden), nn.LayerNorm(hidden), nn.ReLU()
        )
        self.blocks = nn.Sequential(
            ConvBlock(hidden, hidden, kernel, act=nn.ReLU), ResidualDownBlock(hidden, kernel)
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return features_576(x)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.projection(features).transpose(1, 2))
        return self.head(hidden.mean(dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.standardised_features(x))


ARCHITECTURES = {"tcnn": TemporalCNN}


def build_model(arch: str) -> FeatureStandardised:
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch}; choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def size_mb(model: nn.Module) -> float:
    tensors = [*model.parameters(), *model.buffers()]
    return sum(t.numel() * t.element_size() for t in tensors) / 2**20


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sample = torch.randn(64, config.NUM_FRAMES, config.NUM_LANDMARKS, config.COORDS)
    for arch in ARCHITECTURES:
        net = build_model(arch).eval()
        log.info(
            "%s params=%d size=%.2f MB features=%s output=%s",
            arch,
            count_parameters(net),
            size_mb(net),
            tuple(net.features(sample).shape),
            tuple(net(sample).shape),
        )
