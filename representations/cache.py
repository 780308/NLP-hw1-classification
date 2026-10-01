"""Build and validate the deterministic Stage-I feature banks and splits."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image

from dataset import CLASS_TO_IDX, list_samples, stratified_split
from training import write_json

from .extractors import GROUP_SLICES, extract_all
from .preprocess import preprocess_for_handcrafted

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "outputs/stage1/cache"
HANDCRAFTED = CACHE_ROOT / "handcrafted128"
RAW = CACHE_ROOT / "raw64"
SPLIT_DIR = ROOT / "report/splits"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def source_samples():
    samples = list_samples(ROOT / "data/train")
    labels = [label for _, label in samples]
    if len(samples) != 2000 or Counter(labels) != {0: 1000, 1: 1000}:
        raise ValueError(f"Unexpected training dataset size/distribution: {len(samples)}, {Counter(labels)}")
    return samples


def cache_config(samples) -> dict:
    filenames = [path.name for path, _ in samples]
    return {
        "schema_version": 1,
        "image_size": 128,
        "preprocessing": "EXIF transpose, RGB, LANCZOS fit, centered BORDER_REFLECT_101",
        "grid": {"region_size": 32, "stride": 16, "height": 7, "width": 7},
        "feature_groups": GROUP_SLICES,
        "source_filenames": filenames,
        "source_hashes": {
            name: sha256_file(ROOT / "representations" / name)
            for name in ("preprocess.py", "extractors.py")
        },
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "Pillow", "scikit-image", "opencv-python-headless")
        },
    }


def _existing_manifest(directory: Path, config_hash: str) -> dict | None:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"Incomplete cache at {directory}; inspect before rebuilding")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != config_hash:
        raise RuntimeError(f"Cache config changed at {directory}; do not silently reuse it")
    return manifest


def build_cache() -> tuple[dict, dict]:
    samples = source_samples()
    filenames = [path.name for path, _ in samples]
    labels = np.asarray([label for _, label in samples], dtype=np.uint8)
    config = cache_config(samples)
    handmade_hash = stable_sha256(config)
    raw_config = {
        "schema_version": 1,
        "source_filenames": filenames,
        "method": "RGB, direct PIL bilinear resize 64x64, uint8 CHW; adapter scales to [-1,1]",
        "code_sha256": sha256_file(ROOT / "representations/cache.py"),
    }
    raw_hash = stable_sha256(raw_config)
    handmade_manifest = _existing_manifest(HANDCRAFTED, handmade_hash)
    raw_manifest = _existing_manifest(RAW, raw_hash)
    if handmade_manifest is None:
        HANDCRAFTED.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        bank = open_memmap(HANDCRAFTED / "features.npy", mode="w+", dtype=np.float32, shape=(2000, 190, 7, 7))
        for index, (path, _) in enumerate(samples):
            with Image.open(path) as image:
                rgb = preprocess_for_handcrafted(image)
            bank[index] = extract_all(rgb)
            if (index + 1) % 200 == 0:
                print(f"Handcrafted cache: {index + 1}/2000", flush=True)
        bank.flush()
        del bank
        np.save(HANDCRAFTED / "labels.npy", labels)
        write_json(HANDCRAFTED / "filenames.json", filenames)
        handmade_manifest = {
            "schema_version": 1,
            "source_dataset": "data/train",
            "image_count": 2000,
            "image_size": 128,
            "preprocessing": {"aspect_ratio_preserved": True, "padding": "reflect", "random_augmentation": False},
            "grid": {"spatial_height": 7, "spatial_width": 7, "region_size": 32, "stride": 16},
            "channel_groups": GROUP_SLICES,
            "dtype": "float32",
            "feature_shape": [2000, 190, 7, 7],
            "dependency_versions": config["dependency_versions"],
            "code_commit": current_commit(),
            "config_sha256": handmade_hash,
            "filenames_sha256": stable_sha256(filenames),
            "extraction_seconds": time.perf_counter() - started,
            "bytes_on_disk": (HANDCRAFTED / "features.npy").stat().st_size,
        }
        write_json(HANDCRAFTED / "manifest.json", handmade_manifest)
    if raw_manifest is None:
        RAW.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        bank = open_memmap(RAW / "features.npy", mode="w+", dtype=np.uint8, shape=(2000, 3, 64, 64))
        for index, (path, _) in enumerate(samples):
            with Image.open(path) as image:
                rgb = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
                bank[index] = np.asarray(rgb).transpose(2, 0, 1)
        bank.flush()
        del bank
        np.save(RAW / "labels.npy", labels)
        write_json(RAW / "filenames.json", filenames)
        raw_manifest = {
            "schema_version": 1,
            "source_dataset": "data/train",
            "image_count": 2000,
            "feature_shape": [2000, 3, 64, 64],
            "dtype": "uint8",
            "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
            "code_commit": current_commit(),
            "config_sha256": raw_hash,
            "filenames_sha256": stable_sha256(filenames),
            "extraction_seconds": time.perf_counter() - started,
            "bytes_on_disk": (RAW / "features.npy").stat().st_size,
        }
        write_json(RAW / "manifest.json", raw_manifest)
    return handmade_manifest, raw_manifest


def validate_cache() -> dict:
    samples = source_samples()
    config = cache_config(samples)
    manifest = _existing_manifest(HANDCRAFTED, stable_sha256(config))
    if manifest is None:
        raise FileNotFoundError("Build handcrafted cache first")
    filenames = json.loads((HANDCRAFTED / "filenames.json").read_text(encoding="utf-8"))
    labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    features = np.load(HANDCRAFTED / "features.npy", mmap_mode="r")
    expected_names = [path.name for path, _ in samples]
    if filenames != expected_names or len(set(filenames)) != 2000:
        raise AssertionError("Cache filenames differ from the dataset loader")
    if list(labels) != [label for _, label in samples] or features.shape != (2000, 190, 7, 7):
        raise AssertionError("Cache labels or feature shape do not match sources")
    if manifest["filenames_sha256"] != stable_sha256(filenames):
        raise AssertionError("Filename digest mismatch")
    for start in range(0, 2000, 100):
        if not np.isfinite(features[start : start + 100]).all():
            raise AssertionError(f"Non-finite cache values near row {start}")
    raw_config = {
        "schema_version": 1,
        "source_filenames": expected_names,
        "method": "RGB, direct PIL bilinear resize 64x64, uint8 CHW; adapter scales to [-1,1]",
        "code_sha256": sha256_file(ROOT / "representations/cache.py"),
    }
    raw_manifest = _existing_manifest(RAW, stable_sha256(raw_config))
    if raw_manifest is None:
        raise FileNotFoundError("Build raw control cache first")
    raw_features = np.load(RAW / "features.npy", mmap_mode="r")
    raw_labels = np.load(RAW / "labels.npy", mmap_mode="r")
    raw_filenames = json.loads((RAW / "filenames.json").read_text(encoding="utf-8"))
    if raw_features.shape != (2000, 3, 64, 64) or raw_features.dtype != np.uint8:
        raise AssertionError("Raw control cache shape/dtype mismatch")
    if list(raw_labels) != list(labels) or raw_filenames != filenames:
        raise AssertionError("Raw and handcrafted cache order differs")
    return manifest


def ensure_split(seed: int) -> dict:
    samples = source_samples()
    train, validation = stratified_split(samples, 0.1, seed)
    train_names = [path.name for path, _ in train]
    validation_names = [path.name for path, _ in validation]
    if seed == 42:
        baseline = json.loads((ROOT / "report/experiments/dnn-001/split.json").read_text(encoding="utf-8"))
        if train_names != baseline["train"] or validation_names != baseline["internal_validation"]:
            raise AssertionError("Seed-42 split differs from DNN-001")
    data = {
        "seed": seed,
        "validation_fraction": 0.1,
        "train": train_names,
        "internal_validation": validation_names,
        "class_counts": {
            "train": dict(Counter("cat" if label == 0 else "dog" for _, label in train)),
            "internal_validation": dict(Counter("cat" if label == 0 else "dog" for _, label in validation)),
        },
    }
    data["sha256"] = stable_sha256(data)
    path = SPLIT_DIR / f"stage1_seed{seed}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != data:
            raise AssertionError(f"Existing split changed: {path}")
    else:
        write_json(path, data)
    return data


def split_indices(split: dict) -> tuple[list[int], list[int], list[str]]:
    filenames = json.loads((HANDCRAFTED / "filenames.json").read_text(encoding="utf-8"))
    lookup = {name: index for index, name in enumerate(filenames)}
    train_indices = [lookup[name] for name in split["train"]]
    validation_indices = [lookup[name] for name in split["internal_validation"]]
    if len(set(train_indices + validation_indices)) != 2000:
        raise AssertionError("Split does not partition training images")
    return train_indices, validation_indices, filenames


def normalization(train_indices: list[int], channels: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    features = np.load(HANDCRAFTED / "features.npy", mmap_mode="r")
    selected = np.asarray(features[train_indices][:, channels], dtype=np.float32)
    mean = selected.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = selected.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std
