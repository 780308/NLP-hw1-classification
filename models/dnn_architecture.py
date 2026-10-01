"""Three fixed fully connected candidates for frozen REP-006-FUSION input."""

from __future__ import annotations

from torch import Tensor, nn

INPUT_DIM = 190 * 7 * 7
ARCHITECTURE_IDS = ("DNN-ARCH-A", "DNN-ARCH-B", "DNN-ARCH-C")


class ArchitectureDNN(nn.Module):
    def __init__(self, architecture_id: str, input_dim: int = INPUT_DIM) -> None:
        super().__init__()
        if input_dim != INPUT_DIM:
            raise ValueError(f"Frozen REP-006 requires {INPUT_DIM} inputs, got {input_dim}")
        if architecture_id == "DNN-ARCH-A":
            layers = (
                nn.Flatten(),
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 2),
            )
        elif architecture_id == "DNN-ARCH-B":
            layers = (
                nn.Flatten(),
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 2),
            )
        elif architecture_id == "DNN-ARCH-C":
            layers = (
                nn.Flatten(),
                nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 2),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture_id}")
        self.architecture_id = architecture_id
        self.network = nn.Sequential(*layers)

    def forward(self, feature_map: Tensor) -> Tensor:
        return self.network(feature_map)
