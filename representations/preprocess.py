"""Aspect-preserving, reflection-padded image preprocessing."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps


def preprocess_for_handcrafted(image: Image.Image, output_size: int = 128) -> np.ndarray:
    """Return an EXIF-corrected RGB uint8 square without cropping or stretching."""
    if output_size < 2:
        raise ValueError("output_size must be at least 2")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    width, height = rgb.size
    if width < 1 or height < 1:
        raise ValueError("Image has an empty dimension")
    scale = min(output_size / width, output_size / height)
    new_width = max(1, min(output_size, round(width * scale)))
    new_height = max(1, min(output_size, round(height * scale)))
    resized = np.asarray(rgb.resize((new_width, new_height), Image.Resampling.LANCZOS))
    left = (output_size - new_width) // 2
    right = output_size - new_width - left
    top = (output_size - new_height) // 2
    bottom = output_size - new_height - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101
    )
    if padded.shape != (output_size, output_size, 3) or padded.dtype != np.uint8:
        raise AssertionError(f"Unexpected preprocessed image: {padded.shape}, {padded.dtype}")
    return padded
