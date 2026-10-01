"""Three shallow CNN candidates for a frozen 190-channel, 7x7 feature map."""

from __future__ import annotations

import torch
from torch import Tensor, nn

ARCHITECTURE_IDS = ("CNN-ARCH-A", "CNN-ARCH-B", "CNN-ARCH-C")


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int,
                 padding: int = 0, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = conv_bn_relu(64, 64, 3, padding=1)
        self.second = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.second(self.first(x)) + x)


class MultiScaleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch1 = conv_bn_relu(96, 32, 1)
        self.branch3 = conv_bn_relu(96, 32, 3, padding=1)
        self.branch_dilated = conv_bn_relu(96, 32, 3, padding=2, dilation=2)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat((self.branch1(x), self.branch3(x), self.branch_dilated(x)), dim=1)


class ArchitectureCNN(nn.Module):
    def __init__(self, architecture_id: str) -> None:
        super().__init__()
        if architecture_id == "CNN-ARCH-A":
            self.features = nn.Sequential(
                conv_bn_relu(190, 64, 1),
                conv_bn_relu(64, 64, 3, padding=1),
                conv_bn_relu(64, 128, 3, padding=1),
            )
        elif architecture_id == "CNN-ARCH-B":
            self.features = nn.Sequential(
                conv_bn_relu(190, 64, 1),
                ResidualBlock(), ResidualBlock(),
                conv_bn_relu(64, 128, 1),
            )
        elif architecture_id == "CNN-ARCH-C":
            self.features = nn.Sequential(
                conv_bn_relu(190, 96, 1),
                MultiScaleBlock(),
                conv_bn_relu(96, 128, 1),
            )
        else:
            raise ValueError(f"Unknown CNN architecture: {architecture_id}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, 2)
        self.architecture_id = architecture_id

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1:] != (190, 7, 7):
            raise ValueError(f"Expected [B,190,7,7], got {tuple(x.shape)}")
        return self.classifier(torch.flatten(self.pool(self.features(x)), start_dim=1))
