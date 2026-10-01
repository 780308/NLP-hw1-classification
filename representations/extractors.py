"""Four non-neural feature families on one aligned 7x7 spatial grid."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMAGE_SIZE = 128
REGION_SIZE = 32
STRIDE = 16
GRID_SIZE = 7
GROUP_SLICES = {
    "hog": (0, 36),
    "lbp": (36, 46),
    "hsv": (46, 62),
    "rootsift": (62, 190),
}


def _regions() -> list[tuple[int, int]]:
    return [(y, x) for y in range(0, 97, STRIDE) for x in range(0, 97, STRIDE)]


def extract_hog(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blocks = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=False,
    )
    if blocks.shape != (7, 7, 2, 2, 9):
        raise AssertionError(f"Unexpected HOG shape: {blocks.shape}")
    return blocks.reshape(7, 7, 36).transpose(2, 0, 1).astype(np.float32)


def extract_lbp(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    codes = local_binary_pattern(gray, P=8, R=1, method="uniform")
    result = np.empty((10, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for index, (y, x) in enumerate(_regions()):
        region = codes[y : y + REGION_SIZE, x : x + REGION_SIZE]
        counts = np.bincount(region.astype(np.int32).ravel(), minlength=10)[:10]
        result[:, index // 7, index % 7] = counts / max(1, counts.sum())
    return result

def extract_hsv(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    result = np.empty((16, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for index, (y, x) in enumerate(_regions()):
        region = hsv[y : y + REGION_SIZE, x : x + REGION_SIZE]
        parts = []
        for channel, bins, limit in ((0, 8, 180), (1, 4, 256), (2, 4, 256)):
            counts, _ = np.histogram(region[:, :, channel], bins=bins, range=(0, limit))
            parts.append(counts.astype(np.float32) / max(1, counts.sum()))
        result[:, index // 7, index % 7] = np.concatenate(parts)
    return result


def extract_rootsift(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    keypoints = [
        cv2.KeyPoint(float(x + 16), float(y + 16), 32.0, 0.0, 0.0, 0, index)
        for index, (y, x) in enumerate(_regions())
    ]
    returned, descriptors = cv2.SIFT_create().compute(gray, keypoints)
    if descriptors is None or descriptors.shape != (49, 128) or len(returned) != 49:
        raise AssertionError("SIFT did not return exactly 49 spatial descriptors")
    for index, (keypoint, (y, x)) in enumerate(zip(returned, _regions())):
        if keypoint.class_id != index or abs(keypoint.pt[0] - x - 16) > 0.01 or abs(keypoint.pt[1] - y - 16) > 0.01:
            raise AssertionError("SIFT changed keypoint order or location")
    descriptors = descriptors.astype(np.float32)
    descriptors /= np.abs(descriptors).sum(axis=1, keepdims=True) + 1e-12
    np.sqrt(descriptors, out=descriptors)
    return descriptors.reshape(7, 7, 128).transpose(2, 0, 1)


def extract_all(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected a 128x128 RGB uint8 image")
    groups = (extract_hog(rgb), extract_lbp(rgb), extract_hsv(rgb), extract_rootsift(rgb))
    result = np.concatenate(groups, axis=0)
    if result.shape != (190, 7, 7) or not np.isfinite(result).all():
        raise AssertionError("Non-finite or mis-shaped handcrafted feature map")
    return result
