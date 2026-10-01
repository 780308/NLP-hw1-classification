"""Load the assignment's flat cat.N.jpg / dog.N.jpg image directories."""

from __future__ import annotations

import random
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_TO_IDX = {"cat": 0, "dog": 1}
IMAGE_NAME = re.compile(r"^(cat|dog)\.(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
Sample = tuple[Path, int]


def list_samples(directory: Path) -> list[Sample]:
    """Read labels from filenames and reject unknown image names."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")

    samples: list[Sample] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        match = IMAGE_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected image filename: {path}")
        samples.append((path, CLASS_TO_IDX[match.group(1).lower()]))

    if not samples:
        raise ValueError(f"No images found in {directory}")
    return samples


def stratified_split(
    samples: list[Sample], validation_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Make a reproducible, class-balanced split using training images only."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    rng = random.Random(seed)
    train_samples: list[Sample] = []
    validation_samples: list[Sample] = []
    for class_index in CLASS_TO_IDX.values():
        class_samples = [item for item in samples if item[1] == class_index]
        rng.shuffle(class_samples)
        validation_count = round(len(class_samples) * validation_fraction)
        if validation_count == 0 or validation_count == len(class_samples):
            raise ValueError(f"Class {class_index} is too small for this split")
        validation_samples.extend(class_samples[:validation_count])
        train_samples.extend(class_samples[validation_count:])

    train_samples.sort(key=lambda item: item[0].name)
    validation_samples.sort(key=lambda item: item[0].name)
    return train_samples, validation_samples


def image_transform(input_size: int, training: bool) -> transforms.Compose:
    steps: list = [transforms.Resize((input_size, input_size))]
    if training:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    return transforms.Compose(steps)


class CatDogDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return self.transform(rgb), label
