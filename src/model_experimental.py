import torch
from torch import nn

from src import config
from src.model import HAND_DIMS, HAND_SIZE_REF, ConvBlock, FeatureStandardised, masked_velocity

# Local experiments only (32-frame, 314-feature CNN and CNN-GRU). Not the selected architecture
# and not part of the Kaggle package.

POSE_KEEP = (0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
NUM_STATIC_314 = len(POSE_KEEP) * 2 + 2 * HAND_DIMS + 4
NUM_FEATURES_314 = 2 * NUM_STATIC_314 + 2


def hand_presence(hand: torch.Tensor) -> torch.Tensor:
    return (hand.abs().sum(dim=(-1, -2)) > 0).to(hand.dtype)


def hand_local(hand: torch.Tensor) -> torch.Tensor:
    relative = hand - hand[:, :, :1]
    size = torch.linalg.vector_norm(relative[:, :, HAND_SIZE_REF, :2], dim=-1)
    scaled = relative / size.clamp_min(1e-3)[..., None, None]
    return scaled.clamp(-config.CLIP_VALUE, config.CLIP_VALUE).flatten(2)


def features_314(x: torch.Tensor) -> torch.Tensor:
    batch, frames = x.shape[:2]
    pose = x[:, :, list(POSE_KEEP), :2].flatten(2)
    left, right = x[:, :, config.LEFT_HAND_SLICE], x[:, :, config.RIGHT_HAND_SLICE]
    left_present, right_present = hand_presence(left), hand_presence(right)
    static = torch.cat(
        [pose, hand_local(left), hand_local(right), left[:, :, 0, :2], right[:, :, 0, :2]], dim=-1
    )
    valid = torch.cat(
        [
            torch.ones(batch, frames, pose.shape[-1], dtype=x.dtype, device=x.device),
            left_present[..., None].expand(-1, -1, HAND_DIMS),
            right_present[..., None].expand(-1, -1, HAND_DIMS),
            left_present[..., None].expand(-1, -1, 2),
            right_present[..., None].expand(-1, -1, 2),
        ],
        dim=-1,
    )
    flags = torch.stack([left_present, right_present], dim=-1)
    return torch.cat([static, masked_velocity(static, valid), flags], dim=-1)


class ExperimentalClassifier(FeatureStandardised):
    """Experimental 314-feature variants (cnn, cnn_gru); not the selected architecture."""

    num_features = NUM_FEATURES_314

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        hidden: int = 192,
        temporal: int = 256,
        gru_hidden: int = 128,
        use_gru: bool = False,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.num_features, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2)
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

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return features_314(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.standardised_features(x)
        hidden = self.temporal(self.encoder(features).transpose(1, 2)).transpose(1, 2)
        if self.gru is not None:
            hidden, _ = self.gru(hidden)
        pooled = torch.cat([hidden.mean(dim=1), hidden.amax(dim=1)], dim=-1)
        return self.head(pooled)


EXPERIMENTS = {
    "cnn": lambda: ExperimentalClassifier(use_gru=False),
    "cnn_gru": lambda: ExperimentalClassifier(use_gru=True),
}
