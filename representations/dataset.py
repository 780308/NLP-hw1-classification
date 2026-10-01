"""Memory-mapped feature maps with split-specific normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedRepresentationDataset(Dataset):
    def __init__(
        self,
        features_path: Path,
        labels_path: Path,
        indices: list[int],
        channel_indices: tuple[int, ...],
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
    ) -> None:
        self.features_path = features_path
        self.labels = np.load(labels_path, mmap_mode="r")
        self.indices = indices
        self.channels = np.asarray(channel_indices)
        self.mean = np.asarray(channel_mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(channel_std, dtype=np.float32)[:, None, None]
        self.features: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        if self.features is None:
            self.features = np.load(self.features_path, mmap_mode="r")
        index = self.indices[item]
        feature_map = np.array(self.features[index, self.channels], dtype=np.float32, copy=True)
        feature_map = (feature_map - self.mean) / self.std
        return torch.from_numpy(feature_map), int(self.labels[index])


class CachedRawDataset(Dataset):
    def __init__(self, features_path: Path, labels_path: Path, indices: list[int]) -> None:
        self.features_path = features_path
        self.labels = np.load(labels_path, mmap_mode="r")
        self.indices = indices
        self.features: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        if self.features is None:
            self.features = np.load(self.features_path, mmap_mode="r")
        index = self.indices[item]
        image = np.asarray(self.features[index], dtype=np.float32).copy() / 127.5 - 1.0
        return torch.from_numpy(image), int(self.labels[index])
