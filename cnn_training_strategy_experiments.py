"""Compare three fixed training interventions for frozen CNN-ARCH-C."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from cnn_architecture_experiments import context, run_record, write_predictions
from models.cnn_architecture import ArchitectureCNN
from representations.cache import (
    HANDCRAFTED, ROOT, cache_config, current_commit, sha256_file,
    source_samples, stable_sha256,
)
from representations.dataset import CachedRepresentationDataset
from representations.extractors import GROUP_SLICES, extract_all
from representations.preprocess import preprocess_for_handcrafted
from training import run_epoch, seed_everything, select_device, write_history, write_json

BATCH_ID = "CNN-TRAIN-001"
ARCHITECTURE_ID = "CNN-ARCH-C"
REPRESENTATION_ID = "REP-006-FUSION"
SEEDS = (42, 123, 2026)
T0 = "CNN-TRAIN-T0"
STRATEGIES = ("CNN-TRAIN-T1-FLIP", "CNN-TRAIN-T2-DROPOUT", "CNN-TRAIN-T3-COSINE")
FLIP_CACHE = ROOT / "outputs/stage1/cache/handcrafted128_flipped"
OUTPUT = ROOT / "outputs/cnn_training" / BATCH_ID
REPORT = ROOT / "report/cnn_training"
ARCHIVE_FILES = ("config.json", "normalization.json", "history.csv", "metrics.json", "train_summary.json", "predictions.csv")
BASE_MANIFEST_HASH = "97f9453652a511952fad677d596feb41e33d2c9c9f94024ec616c9916767ecd6"
FIELDS = (
    "strategy_id", "seed", "parameter_count", "training_sample_count", "best_epoch",
    "train_accuracy_at_best_epoch", "final_train_accuracy", "validation_loss",
    "validation_accuracy", "cat_accuracy", "dog_accuracy", "balanced_accuracy",
    "macro_f1", "fit_seconds", "split_sha256", "feature_cache_manifest_sha256",
    "flipped_cache_manifest_sha256", "code_commit", "status", "source_run",
)


class DropoutCNN(nn.Module):
    """The selected CNN with exactly one post-fusion Dropout2d intervention."""

    def __init__(self) -> None:
        super().__init__()
        self.base = ArchitectureCNN(ARCHITECTURE_ID)
        self.dropout = nn.Dropout2d(p=0.10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (190, 7, 7):
            raise ValueError(f"Expected [B,190,7,7], got {tuple(x.shape)}")
        fused = self.base.features(x)
        return self.base.classifier(torch.flatten(self.base.pool(self.dropout(fused)), start_dim=1))


def model_for(strategy_id: str) -> nn.Module:
    return DropoutCNN() if strategy_id == STRATEGIES[1] else ArchitectureCNN(ARCHITECTURE_ID)


def flip_source(image: Image.Image) -> np.ndarray:
    # EXIF orientation is resolved before mirroring the RGB image; the frozen
    # 128x128 preprocessing and four feature extractors are then used unchanged.
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    return extract_all(preprocess_for_handcrafted(ImageOps.mirror(rgb)))


def flip_config() -> dict:
    train_samples = source_samples()
    base = cache_config(train_samples)
    return {
        "schema_version": 1, "source_dataset": "data/train",
        "transform": "EXIF-corrected RGB horizontal mirror before frozen 128x128 preprocessing and feature extraction",
        "base_feature_config_sha256": stable_sha256(base),
        "source_filenames": [path.name for path, _ in train_samples],
        "feature_groups": GROUP_SLICES,
        "dependency_versions": base["dependency_versions"],
        "source_hashes": base["source_hashes"],
    }


def verify_base() -> tuple[tuple[int, ...], dict[int, dict]]:
    channels, prepared = context()
    if sha256_file(HANDCRAFTED / "manifest.json") != BASE_MANIFEST_HASH:
        raise AssertionError("Frozen Stage-I cache manifest changed")
    decision = json.loads((ROOT / "report/cnn_architecture/architecture_decision.json").read_text(encoding="utf-8"))
    if decision["selected_architecture"] != ARCHITECTURE_ID:
        raise AssertionError("Selected CNN architecture changed")
    return channels, prepared


def smoke() -> None:
    channels, prepared = verify_base()
    samples = source_samples()
    with Image.open(samples[0][0]) as image:
        flipped = flip_source(image)
    original = np.load(HANDCRAFTED / "features.npy", mmap_mode="r")[0]
    if flipped.shape != (190, 7, 7) or not np.isfinite(flipped).all() or np.array_equal(flipped, original):
        raise AssertionError("RGB-flipped handcrafted representation is invalid")
    dataset = CachedRepresentationDataset(HANDCRAFTED / "features.npy", HANDCRAFTED / "labels.npy",
                                          prepared[42]["train_indices"][:8], channels,
                                          prepared[42]["mean"], prepared[42]["std"])
    features, labels = next(iter(DataLoader(dataset, batch_size=8)))
    normal = ArchitectureCNN(ARCHITECTURE_ID)
    dropout = DropoutCNN()
    if features.shape != (8, 190, 7, 7) or sum(isinstance(layer, nn.Dropout2d) for layer in dropout.modules()) != 1:
        raise AssertionError("Frozen input or T2 dropout placement changed")
    dropout.train()
    layer = dropout.dropout
    mask_train = layer(torch.ones(32, 128, 7, 7))
    dropout.eval()
    mask_eval = layer(torch.ones(32, 128, 7, 7))
    if torch.equal(mask_train, mask_eval) or not torch.equal(mask_eval, torch.ones_like(mask_eval)):
        raise AssertionError("Dropout2d train/eval behavior invalid")
    for model in (normal, dropout):
        nn.CrossEntropyLoss()(model(features), labels).backward()
        if model(features).shape != (8, 2):
            raise AssertionError("CNN strategy output has wrong shape")
    optimizer = torch.optim.AdamW(normal.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    before = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    if not 1e-6 < optimizer.param_groups[0]["lr"] < before:
        raise AssertionError("T3 cosine learning rate did not decrease")
    print("T1 RGB flip/extraction, T2 dropout train/eval and T3 cosine schedule smoke checks OK", flush=True)


def build_flip_cache() -> None:
    verify_base()
    config = flip_config()
    expected_hash = stable_sha256(config)
    manifest_path = FLIP_CACHE / "manifest.json"
    if manifest_path.exists():
        validate_flip_cache()
        print("Reused compatible flipped training cache", flush=True)
        return
    if FLIP_CACHE.exists() and any(FLIP_CACHE.iterdir()):
        raise RuntimeError(f"Partial flipped cache exists; inspect before retrying: {FLIP_CACHE}")
    FLIP_CACHE.mkdir(parents=True)
    samples = source_samples()
    features = open_memmap(FLIP_CACHE / "features.npy", mode="w+", dtype=np.float32,
                           shape=(2000, 190, 7, 7))
    started = time.perf_counter()
    for index, (path, _) in enumerate(samples):
        with Image.open(path) as image:
            features[index] = flip_source(image)
        if (index + 1) % 200 == 0:
            print(f"Flipped training features: {index + 1}/2000", flush=True)
    features.flush()
    del features
    np.save(FLIP_CACHE / "labels.npy", np.asarray([label for _, label in samples], dtype=np.uint8))
    filenames = [f"{path.name}::hflip" for path, _ in samples]
    write_json(FLIP_CACHE / "filenames.json", filenames)
    write_json(manifest_path, {
        "source_dataset": "data/train", "transform": "horizontal_flip",
        "base_feature_config_sha256": config["base_feature_config_sha256"],
        "base_cache_manifest_sha256": BASE_MANIFEST_HASH,
        "flipped_cache_config_sha256": expected_hash,
        "filenames_sha256": stable_sha256(filenames),
        "image_count": 2000, "feature_shape": [2000, 190, 7, 7],
        "channel_groups": {key: list(value) for key, value in GROUP_SLICES.items()},
        "dependency_versions": config["dependency_versions"],
        "code_commit": current_commit(),
        "extraction_seconds": time.perf_counter() - started,
    })
    validate_flip_cache()


def validate_flip_cache() -> dict:
    verify_base()
    path = FLIP_CACHE / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    samples = source_samples()
    filenames = [f"{sample.name}::hflip" for sample, _ in samples]
    cached_names = json.loads((FLIP_CACHE / "filenames.json").read_text(encoding="utf-8"))
    labels = np.load(FLIP_CACHE / "labels.npy", mmap_mode="r")
    features = np.load(FLIP_CACHE / "features.npy", mmap_mode="r")
    if (manifest["flipped_cache_config_sha256"] != stable_sha256(flip_config())
            or manifest["base_cache_manifest_sha256"] != BASE_MANIFEST_HASH
            or manifest["filenames_sha256"] != stable_sha256(filenames)
            or manifest["feature_shape"] != [2000, 190, 7, 7]
            or manifest["channel_groups"] != {key: list(value) for key, value in GROUP_SLICES.items()}
            or cached_names != filenames or list(labels) != [label for _, label in samples]
            or features.shape != (2000, 190, 7, 7) or features.dtype != np.float32):
        raise AssertionError("Flipped cache manifest, order, labels or geometry mismatch")
    for start in range(0, 2000, 100):
        if not np.isfinite(features[start:start + 100]).all():
            raise AssertionError(f"Nonfinite flipped features near row {start}")
    print("Flipped cache valid: [2000,190,7,7]", flush=True)
    return manifest


def t1_normalization(train_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    original = np.load(HANDCRAFTED / "features.npy", mmap_mode="r")
    flipped = np.load(FLIP_CACHE / "features.npy", mmap_mode="r")
    selected = np.concatenate((original[train_indices], flipped[train_indices]), axis=0).astype(np.float32)
    mean = selected.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = selected.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def run_dir(strategy_id: str, seed: int) -> Path:
    return OUTPUT / strategy_id / f"seed{seed}"


def archive_run(strategy_id: str, seed: int) -> None:
    source = run_dir(strategy_id, seed)
    target = REPORT / "experiments" / strategy_id / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVE_FILES:
        if not (source / name).exists():
            raise FileNotFoundError(f"Missing formal run artifact: {source / name}")
        shutil.copy2(source / name, target / name)


def train_one(strategy_id: str, seed: int, prepared: dict, channels: tuple[int, ...], output: Path) -> None:
    seed_everything(seed)
    device = select_device("auto")
    split = prepared["split"]
    t1 = strategy_id == STRATEGIES[0]
    if t1:
        mean, std = t1_normalization(prepared["train_indices"])
        flip_manifest_hash = sha256_file(FLIP_CACHE / "manifest.json")
    else:
        mean, std = prepared["mean"], prepared["std"]
        flip_manifest_hash = ""
    original_train = CachedRepresentationDataset(HANDCRAFTED / "features.npy", HANDCRAFTED / "labels.npy",
                                                 prepared["train_indices"], channels, mean, std)
    if t1:
        flipped_train = CachedRepresentationDataset(FLIP_CACHE / "features.npy", FLIP_CACHE / "labels.npy",
                                                    prepared["train_indices"], channels, mean, std)
        train_dataset = ConcatDataset((original_train, flipped_train))
    else:
        train_dataset = original_train
    validation_dataset = CachedRepresentationDataset(HANDCRAFTED / "features.npy", HANDCRAFTED / "labels.npy",
                                                     prepared["validation_indices"], channels, mean, std)
    loader_args = {"batch_size": 32, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, shuffle=True, generator=torch.Generator().manual_seed(seed), **loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)
    model = model_for(strategy_id).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    config = {
        "batch_id": BATCH_ID, "strategy_id": strategy_id, "architecture_id": ARCHITECTURE_ID,
        "representation_id": REPRESENTATION_ID, "representation_shape": [190, 7, 7],
        "seed": seed, "training_sample_count": len(train_dataset),
        "original_training_samples": 1800, "validation_samples": 200,
        "parameter_count": parameter_count, "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": sha256_file(HANDCRAFTED / "manifest.json"),
        "flipped_cache_manifest_sha256": flip_manifest_hash,
        "augmentation": "horizontal_flip" if t1 else "none",
        "augmented_training_samples": 3600 if t1 else 0,
        "dropout2d_p": 0.10 if strategy_id == STRATEGIES[1] else 0.0,
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 30, "eta_min": 1e-6} if strategy_id == STRATEGIES[2] else None,
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "batch_size": 32, "max_epochs": 30, "early_stopping_patience": 6,
        "loss": "CrossEntropyLoss", "checkpoint_selection": "highest validation accuracy; tie lower validation loss",
        "device": str(device), "code_commit": current_commit(),
    }
    write_json(output / "config.json", config)
    write_json(output / "normalization.json", {
        "source": "1800 original and 1800 flipped training features only" if t1 else "1800 original training features only",
        "split_sha256": split["sha256"], "mean": mean.tolist(), "std": std.tolist(),
    })
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
                 if strategy_id == STRATEGIES[2] else None)
    history = []
    best_accuracy, best_loss = -1.0, float("inf")
    lowest_loss, stale = float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, 31):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        history.append({
            "epoch": epoch, "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_cat_accuracy": validation_metrics["cat_accuracy"],
            "validation_dog_accuracy": validation_metrics["dog_accuracy"],
        })
        write_history(output / "history.csv", history)
        accuracy, loss = validation_metrics["accuracy"], validation_metrics["loss"]
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_accuracy, best_loss = accuracy, loss
            torch.save({"model_state": model.state_dict(), "strategy_id": strategy_id,
                        "epoch": epoch, "validation_metrics": validation_metrics,
                        "scheduler_state": scheduler.state_dict() if scheduler else None,
                        "config": config}, output / "best_model.pt")
        if loss < lowest_loss - 1e-4:
            lowest_loss, stale = loss, 0
        else:
            stale += 1
        print(f"{strategy_id} seed {seed} epoch {epoch:02d}: train={train_metrics['accuracy']:.3f}, val={accuracy:.3f}, loss={loss:.4f}, lr={learning_rate:.7f}", flush=True)
        if scheduler is not None:
            scheduler.step()
        if stale >= 6:
            break
    fit_seconds = time.perf_counter() - started
    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    labels_all, logits_all = [], []
    with torch.no_grad():
        for features, labels in validation_loader:
            logits_all.append(model(features.to(device, non_blocking=True)).cpu())
            labels_all.append(labels)
    labels = torch.cat(labels_all).numpy()
    cache_labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    if not np.array_equal(labels, cache_labels[prepared["validation_indices"]]):
        raise AssertionError("Validation labels differ from frozen split")
    measured = write_predictions(output / "predictions.csv", split["internal_validation"],
                                 labels, torch.cat(logits_all).numpy())
    if abs(measured["validation_accuracy"] - checkpoint["validation_metrics"]["accuracy"]) > 1e-9:
        raise AssertionError("Reloaded strategy checkpoint changed validation accuracy")
    best_epoch = checkpoint["epoch"]
    metrics = {
        "experiment_id": f"{BATCH_ID}-{strategy_id}-seed{seed}",
        "strategy_id": strategy_id, "seed": seed,
        "parameter_count": parameter_count, "training_sample_count": len(train_dataset),
        "best_epoch": best_epoch, "epochs_run": len(history),
        "train_accuracy_at_best_epoch": history[best_epoch - 1]["train_accuracy"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "validation_loss": checkpoint["validation_metrics"]["loss"],
        **measured, "fit_seconds": fit_seconds,
        "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": config["feature_cache_manifest_sha256"],
        "flipped_cache_manifest_sha256": flip_manifest_hash,
        "augmentation": config["augmentation"],
        "augmented_training_samples": config["augmented_training_samples"],
        "code_commit": config["code_commit"], "status": "complete",
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "train_summary.json", {
        "strategy_id": strategy_id, "seed": seed, "epochs_run": len(history),
        "best_epoch": best_epoch, "fit_seconds": fit_seconds,
        "training_sample_count": len(train_dataset), "parameter_count": parameter_count,
        "best_validation_metrics": checkpoint["validation_metrics"],
        "scheduler": config["scheduler"], "device": str(device),
    })


def run_one(strategy_id: str, seed: int, prepared: dict, channels: tuple[int, ...]) -> None:
    output = run_dir(strategy_id, seed)
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (metrics.get("status") != "complete" or metrics.get("strategy_id") != strategy_id
                or metrics.get("seed") != seed or metrics.get("split_sha256") != prepared["split"]["sha256"]):
            raise RuntimeError(f"Existing strategy run incompatible: {output}")
        archive_run(strategy_id, seed)
        print(f"Reused completed {strategy_id} seed {seed}", flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Partial strategy run exists; inspect before retrying: {output}")
    output.mkdir(parents=True)
    try:
        train_one(strategy_id, seed, prepared, channels, output)
        archive_run(strategy_id, seed)
    except Exception as error:
        write_json(output / "failure.json", {"strategy_id": strategy_id, "seed": seed,
                   "status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        raise


def run_all() -> None:
    channels, prepared = verify_base()
    validate_flip_cache()
    for strategy_id in STRATEGIES:
        for seed in SEEDS:
            run_one(strategy_id, seed, prepared[seed], channels)
    print("All nine new strategy runs completed", flush=True)


def record(strategy_id: str, seed: int) -> tuple[dict, list[dict], Path]:
    if strategy_id == T0:
        metrics, history, path = run_record(ARCHITECTURE_ID, seed)
    else:
        directory = REPORT / "experiments" / strategy_id / f"seed{seed}"
        path = directory / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        if metrics["strategy_id"] != strategy_id or metrics["seed"] != seed or metrics["status"] != "complete":
            raise AssertionError(f"Invalid strategy run: {path}")
    return metrics, history, path


def rows_from_runs() -> list[dict]:
    rows = []
    for strategy_id in (T0, *STRATEGIES):
        for seed in SEEDS:
            metrics, _, path = record(strategy_id, seed)
            rows.append({
                "strategy_id": strategy_id, "seed": seed,
                "parameter_count": metrics["parameter_count"],
                "training_sample_count": metrics.get("training_sample_count", 1800),
                "best_epoch": metrics["best_epoch"],
                "train_accuracy_at_best_epoch": metrics["train_accuracy_at_best_epoch"],
                "final_train_accuracy": metrics["final_train_accuracy"],
                "validation_loss": metrics["validation_loss"],
                "validation_accuracy": metrics["validation_accuracy"],
                "cat_accuracy": metrics["cat_accuracy"], "dog_accuracy": metrics["dog_accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"],
                "fit_seconds": metrics["fit_seconds"], "split_sha256": metrics["split_sha256"],
                "feature_cache_manifest_sha256": metrics["feature_cache_manifest_sha256"],
                "flipped_cache_manifest_sha256": metrics.get("flipped_cache_manifest_sha256", ""),
                "code_commit": metrics["code_commit"], "status": metrics["status"],
                "source_run": str(path.relative_to(ROOT)).replace("\\", "/"),
            })
    return rows


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for strategy_id in (T0, *STRATEGIES):
        subset = [row for row in rows if row["strategy_id"] == strategy_id]
        if len(subset) != 3:
            raise AssertionError(f"Incomplete strategy group: {strategy_id}")
        accuracies = [row["validation_accuracy"] for row in subset]
        cat = statistics.mean(row["cat_accuracy"] for row in subset)
        dog = statistics.mean(row["dog_accuracy"] for row in subset)
        late_loss_rises = 0
        for row in subset:
            _, history, _ = record(strategy_id, row["seed"])
            if (len(history) > row["best_epoch"]
                    and float(history[-1]["validation_loss"]) - row["validation_loss"] >= 0.10):
                late_loss_rises += 1
        result[strategy_id] = {
            "strategy_id": strategy_id, "parameter_count": subset[0]["parameter_count"],
            "training_sample_count": subset[0]["training_sample_count"],
            "mean_validation_accuracy": statistics.mean(accuracies),
            "sample_std_validation_accuracy": statistics.stdev(accuracies),
            "worst_seed_accuracy": min(accuracies),
            "mean_cat_accuracy": cat, "mean_dog_accuracy": dog,
            "cat_dog_gap": abs(cat - dog),
            "mean_balanced_accuracy": statistics.mean(row["balanced_accuracy"] for row in subset),
            "mean_macro_f1": statistics.mean(row["macro_f1"] for row in subset),
            "median_best_epoch": int(statistics.median(row["best_epoch"] for row in subset)),
            "mean_final_train_accuracy": statistics.mean(row["final_train_accuracy"] for row in subset),
            "late_validation_loss_rise_runs": late_loss_rises,
            "mean_fit_seconds": statistics.mean(row["fit_seconds"] for row in subset),
            "seed_accuracies": {str(row["seed"]): row["validation_accuracy"] for row in subset},
        }
    return result


def decide(grouped: dict) -> dict:
    baseline = grouped[T0]
    assessments = {}
    for strategy_id in STRATEGIES:
        current = grouped[strategy_id]
        gain = current["mean_validation_accuracy"] - baseline["mean_validation_accuracy"]
        std_reduction = baseline["sample_std_validation_accuracy"] - current["sample_std_validation_accuracy"]
        worst_gain = current["worst_seed_accuracy"] - baseline["worst_seed_accuracy"]
        gap_reduction = baseline["cat_dog_gap"] - current["cat_dog_gap"]
        if gain >= 0.01 - 1e-12:
            band, retain = "strong", True
        elif gain >= 0.005 - 1e-12:
            band = "moderate"
            retain = std_reduction >= 0.005 - 1e-12 or worst_gain >= 0.01 - 1e-12 or gap_reduction >= 0.02 - 1e-12
        else:
            band = "negligible_or_negative"
            retain = worst_gain >= 0.02 - 1e-12 or (std_reduction >= 0.01 - 1e-12 and gain >= -0.005 - 1e-12)
        assessments[strategy_id] = {
            "mean_accuracy_change": gain, "sample_std_reduction": std_reduction,
            "worst_seed_gain": worst_gain, "cat_dog_gap_reduction": gap_reduction,
            "improvement_band": band, "retained": bool(retain),
        }
    candidates = [grouped[strategy_id] for strategy_id in STRATEGIES if assessments[strategy_id]["retained"]]
    selected = (max(candidates, key=lambda item: (item["mean_validation_accuracy"], item["worst_seed_accuracy"],
                                             -item["cat_dog_gap"], -item["parameter_count"]))["strategy_id"]
                if candidates else T0)
    strong = [strategy_id for strategy_id in STRATEGIES if assessments[strategy_id]["improvement_band"] == "strong"]
    return {"selected_individual_strategy": selected, "assessments": assessments,
            "strong_improvement_strategies": strong, "combination_eligible": len(strong) >= 2,
            "combination_executed": False}


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def summarize() -> None:
    rows = rows_from_runs()
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "training_strategy_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    grouped = aggregate(rows)
    decision = decide(grouped)
    write_json(REPORT / "training_strategy_decision.json", {"batch_id": BATCH_ID, "models": grouped, **decision})
    lines = ["# CNN-TRAIN-001：冻结 CNN-C 的训练策略比较", "",
        "仅用 `data/train` 的种子 42、123、2026 三份既有 1800/200 划分。输入始终为 `REP-006-FUSION` `[190,7,7]`；T0 直接引用 CNN-ARCH-C 的既有运行，没有重训；九次新运行仅分别改变训练策略。未访问 `data/val`。", "",
        "T1 从 EXIF 校正的原始 RGB 图水平镜像后，按冻结的 128×128 预处理和 HOG/LBP/HSV/RootSIFT 提取器重新生成特征；每种子使用 1800 原图 + 1800 镜像训练图，仅用这些训练图拟合归一化。验证始终为 200 张未镜像原图。T2 仅在融合 Conv→BN→ReLU 后加入 Dropout2d(0.10)。T3 仅在原训练中加入 CosineAnnealingLR（T_max=30，eta_min=1e-6）。", "",
        "三种新运行均固定 CrossEntropyLoss、AdamW、初始学习率 3e-4、权重衰减 1e-4、批量 32、最多 30 轮、验证损失早停耐心 6；检查点按最高内部验证准确率，同分时选较低损失。", "",
        "## 四策略汇总", "", "标准差为三种子的样本标准差。", "",
        "| 策略 | 训练样本/种子 | 平均验证准确率 ± 标准差 | 最差种子 | 猫/狗均值 | 类别差 | 平衡准确率 | Macro F1 | 最佳轮中位数 | 平均训练秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy_id in (T0, *STRATEGIES):
        item = grouped[strategy_id]
        lines.append(f"| {strategy_id} | {item['training_sample_count']} | "
                     f"{pct(item['mean_validation_accuracy'])} ± {pct(item['sample_std_validation_accuracy'])} | "
                     f"{pct(item['worst_seed_accuracy'])} | {pct(item['mean_cat_accuracy'])}/{pct(item['mean_dog_accuracy'])} | "
                     f"{pct(item['cat_dog_gap'])} | {pct(item['mean_balanced_accuracy'])} | {pct(item['mean_macro_f1'])} | "
                     f"{item['median_best_epoch']} | {item['mean_fit_seconds']:.2f} |")
    lines += ["", "## 逐种子与训练轨迹", "",
        "| 策略 | 种子 | 最佳轮/末轮 | 验证准确率 | 最佳轮训练→末轮训练准确率 | 最佳轮→末轮验证损失 | 最后学习率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        _, history, _ = record(row["strategy_id"], row["seed"])
        final = history[-1]
        loss_last = float(final["validation_loss"])
        lr = final.get("learning_rate")
        lr_text = f"{float(lr):.7f}" if lr is not None else "历史记录无 LR 列"
        lines.append(f"| {row['strategy_id']} | {row['seed']} | {row['best_epoch']}/{len(history)} | "
                     f"{pct(row['validation_accuracy'])} | {pct(row['train_accuracy_at_best_epoch'])}→{pct(row['final_train_accuracy'])} | "
                     f"{row['validation_loss']:.4f}→{loss_last:.4f} ({loss_last-row['validation_loss']:+.4f}) | {lr_text} |")
    lines += ["", "训练准确率来自训练过程逐批统计（含 BatchNorm/可能的 Dropout），不等于最终检查点对训练集的重评估。后期验证损失上升而训练准确率继续增加时提示过拟合风险；最佳轮后的行为只由实际观察轮次描述。", "",
        "| 策略 | 最佳轮中位数 | 末轮训练准确率均值 | 最佳轮后验证损失升高至少 0.10 的运行数 | 验证准确率样本标准差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy_id in (T0, *STRATEGIES):
        item = grouped[strategy_id]
        lines.append(f"| {strategy_id} | {item['median_best_epoch']} | {pct(item['mean_final_train_accuracy'])} | "
                     f"{item['late_validation_loss_rise_runs']}/3 | {pct(item['sample_std_validation_accuracy'])} |")
    lines += ["", "该表用于比较验证峰值是否推迟、末轮训练拟合程度、后期验证损失及种子波动；这些指标本身不替代预设的策略选择规则。", "",
        "## 预设规则判定", ""]
    for strategy_id in STRATEGIES:
        assessment = decision["assessments"][strategy_id]
        lines.append(f"- {strategy_id}：均值变化 {assessment['mean_accuracy_change']*100:+.2f} 个百分点；"
                     f"标准差减少 {assessment['sample_std_reduction']*100:+.2f} 点；最差种子变化 "
                     f"{assessment['worst_seed_gain']*100:+.2f} 点；类别差减少 "
                     f"{assessment['cat_dog_gap_reduction']*100:+.2f} 点；档位 `{assessment['improvement_band']}`，"
                     f"{'保留' if assessment['retained'] else '不保留'}。")
    lines += ["", f"按预设规则选定 **{decision['selected_individual_strategy']}**。"
              f"达到均值至少 +1.0 个百分点强改善的单策略数：**{len(decision['strong_improvement_strategies'])}**；"
              f"组合实验{'符合触发条件，但本次未执行' if decision['combination_eligible'] else '未触发'}。", "",
              "本阶段没有全量 2000 张训练、CNN 保留集评估、组合实验或其他超参数搜索。", ""]
    (REPORT / "training_strategy_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected {decision['selected_individual_strategy']}; combination eligible={decision['combination_eligible']}", flush=True)


def audit() -> None:
    channels, prepared = verify_base()
    if len(channels) != 190:
        raise AssertionError("Frozen input channel count changed")
    validate_flip_cache()
    directories = list((REPORT / "experiments").glob("**/metrics.json"))
    if len(directories) != 9:
        raise AssertionError(f"Expected nine new runs; found {len(directories)}")
    for strategy_id in STRATEGIES:
        for seed in SEEDS:
            metrics, history, path = record(strategy_id, seed)
            directory = path.parent
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            normal = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
            split = prepared[seed]["split"]
            t1 = strategy_id == STRATEGIES[0]
            if (metrics["split_sha256"] != split["sha256"]
                    or config["split_sha256"] != split["sha256"]
                    or metrics["feature_cache_manifest_sha256"] != BASE_MANIFEST_HASH
                    or config["architecture_id"] != ARCHITECTURE_ID
                    or config["representation_shape"] != [190, 7, 7]
                    or config["optimizer"] != "AdamW" or config["learning_rate"] != 3e-4
                    or config["weight_decay"] != 1e-4 or config["batch_size"] != 32
                    or config["max_epochs"] != 30 or config["early_stopping_patience"] != 6
                    or config["loss"] != "CrossEntropyLoss"
                    or metrics["training_sample_count"] != (3600 if t1 else 1800)
                    or config["augmented_training_samples"] != (3600 if t1 else 0)
                    or config["augmentation"] != ("horizontal_flip" if t1 else "none")
                    or config["dropout2d_p"] != (0.10 if strategy_id == STRATEGIES[1] else 0.0)
                    or (config["scheduler"] is not None) != (strategy_id == STRATEGIES[2])):
                raise AssertionError(f"Strategy protocol mismatch: {directory}")
            expected_parameters = sum(parameter.numel() for parameter in model_for(strategy_id).parameters()
                                      if parameter.requires_grad)
            if metrics["parameter_count"] != expected_parameters or config["parameter_count"] != expected_parameters:
                raise AssertionError(f"Frozen CNN parameter count changed: {directory}")
            expected_mean, expected_std = (t1_normalization(prepared[seed]["train_indices"]) if t1
                                           else (prepared[seed]["mean"], prepared[seed]["std"]))
            if (normal["split_sha256"] != split["sha256"]
                    or not np.array_equal(np.asarray(normal["mean"], dtype=np.float32), expected_mean)
                    or not np.array_equal(np.asarray(normal["std"], dtype=np.float32), expected_std)):
                raise AssertionError(f"Training-only normalization mismatch: {directory}")
            if t1 and metrics["flipped_cache_manifest_sha256"] != sha256_file(FLIP_CACHE / "manifest.json"):
                raise AssertionError(f"T1 flipped cache hash mismatch: {directory}")
            if strategy_id == STRATEGIES[2]:
                rates = [float(item["learning_rate"]) for item in history]
                if config["scheduler"] != {"name": "CosineAnnealingLR", "T_max": 30, "eta_min": 1e-6} or any(next_lr >= lr for lr, next_lr in zip(rates, rates[1:])):
                    raise AssertionError(f"Cosine scheduler configuration/history mismatch: {directory}")
                checkpoint = torch.load(run_dir(strategy_id, seed) / "best_model.pt", map_location="cpu", weights_only=True)
                if checkpoint["scheduler_state"] is None or checkpoint["scheduler_state"]["T_max"] != 30 or checkpoint["scheduler_state"]["eta_min"] != 1e-6:
                    raise AssertionError(f"Cosine scheduler checkpoint state missing: {directory}")
            elif any(float(item["learning_rate"]) != 3e-4 for item in history):
                raise AssertionError(f"Unscheduled learning rate changed: {directory}")
            with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                predictions = list(csv.DictReader(stream))
            if len(predictions) != 200 or [row["filename"] for row in predictions] != split["internal_validation"]:
                raise AssertionError(f"Validation prediction order mismatch: {directory}")
            matrix = [[0, 0], [0, 0]]
            for row in predictions:
                actual, predicted = int(row["true_label"]), int(row["predicted_label"])
                if (actual != (0 if row["filename"].startswith("cat.") else 1)
                        or predicted not in (0, 1) or int(row["correct"]) != int(actual == predicted)
                        or abs(float(row["prob_cat"]) + float(row["prob_dog"]) - 1) > 1e-6):
                    raise AssertionError(f"Invalid prediction row: {directory}")
                matrix[actual][predicted] += 1
            if (sum(matrix[0]) != 100 or sum(matrix[1]) != 100
                    or matrix != metrics["confusion_matrix"]
                    or abs((matrix[0][0] + matrix[1][1]) / 200 - metrics["validation_accuracy"]) > 1e-12
                    or abs(float(history[metrics["best_epoch"] - 1]["validation_accuracy"]) - metrics["validation_accuracy"]) > 1e-12):
                raise AssertionError(f"Validation metrics/history mismatch: {directory}")
    print("Audited 9 new strategy runs, 3 baseline runs and 1,800 new internal-validation predictions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "build-flip-cache", "validate-flip-cache",
                                            "run-all", "summarize", "audit"))
    command = parser.parse_args().command
    {"smoke": smoke, "build-flip-cache": build_flip_cache,
     "validate-flip-cache": validate_flip_cache, "run-all": run_all,
     "summarize": summarize, "audit": audit}[command]()


if __name__ == "__main__":
    main()
