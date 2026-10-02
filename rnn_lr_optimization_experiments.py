"""RNN-TRAIN-003: compare three learning-rate policies on frozen RNN2D features."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from cnn_architecture_experiments import write_predictions
from cnn_training_strategy_experiments import FLIP_CACHE, t1_normalization, validate_flip_cache
from dnn_architecture_experiments import make_dataset
from models.rnn2d import ARCHITECTURES, VisionRNN2DClassifier
from representations.cache import HANDCRAFTED, ROOT, current_commit, sha256_file
from representations.dataset import CachedRepresentationDataset
from rnn2d_architecture_experiments import context, epoch_pass
from training import seed_everything, select_device, write_history, write_json

BATCH_ID = "RNN-TRAIN-003"
ARCHITECTURE_ID = "RNN2D-ARCH-L"  # Report alias: RNN2D-ARCH-BASE.
REPRESENTATION_ID = "REP-006-FUSION"
SEEDS = (42, 123, 2026)
T0 = "RNN-LR-T0-CONST-3E4"
T1 = "RNN-LR-T1-CONST-1P5E4"
T2 = "RNN-LR-T2-COSINE"
T3 = "RNN-LR-T3-PLATEAU"
STRATEGIES = (T1, T2, T3)
ALL_STRATEGIES = (T0, *STRATEGIES)
OUTPUT = ROOT / "outputs/rnn_training" / BATCH_ID
REPORT = ROOT / "report/rnn_lr_optimization"
BASE_REPORT = ROOT / "report/rnn_training/experiments/RNN-TRAIN-T1-FLIP"
ARTIFACTS = ("config.json", "normalization.json", "history.csv", "metrics.json",
             "train_summary.json", "predictions.csv")


def policy(strategy_id: str) -> tuple[float, dict | None]:
    if strategy_id == T1:
        return 1.5e-4, None
    if strategy_id == T2:
        return 3e-4, {"type": "CosineAnnealingLR", "T_max": 20, "eta_min": 1e-5}
    if strategy_id == T3:
        return 3e-4, {"type": "ReduceLROnPlateau", "mode": "min", "factor": 0.3,
                      "patience": 2, "min_lr": 1e-5}
    raise ValueError(strategy_id)


def make_scheduler(optimizer: torch.optim.Optimizer, settings: dict | None):
    if settings is None:
        return None
    if settings["type"] == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings["T_max"], eta_min=settings["eta_min"])
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=settings["mode"], factor=settings["factor"],
        patience=settings["patience"], min_lr=settings["min_lr"])


def smoke() -> None:
    channels, prepared = context()
    validate_flip_cache()
    if channels != tuple(range(190)) or ARCHITECTURES[ARCHITECTURE_ID] != {
            "embed_dim": 320, "rnn_hidden_size": 80, "num_blocks": 3}:
        raise AssertionError("Frozen representation or architecture changed")
    indices = prepared[42]["train_indices"]
    mean, std = t1_normalization(indices)
    train = ConcatDataset((make_dataset(indices, channels, mean, std),
                           CachedRepresentationDataset(FLIP_CACHE / "features.npy",
                                                       FLIP_CACHE / "labels.npy", indices, channels, mean, std)))
    validation = make_dataset(prepared[42]["validation_indices"], channels, mean, std)
    model = VisionRNN2DClassifier(ARCHITECTURE_ID)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    baseline = json.loads((BASE_REPORT / "seed42/metrics.json").read_text(encoding="utf-8"))
    if len(train) != 3600 or len(validation) != 200 or count != baseline["parameter_count"]:
        raise AssertionError("Samples or model parameters differ from T0")
    x, y = next(iter(DataLoader(train, batch_size=8)))
    for strategy in STRATEGIES:
        optimizer = torch.optim.AdamW(model.parameters(), lr=policy(strategy)[0], weight_decay=1e-4)
        scheduler = make_scheduler(optimizer, policy(strategy)[1])
        optimizer.zero_grad(set_to_none=True)
        nn.CrossEntropyLoss()(model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            if strategy == T3:
                scheduler.step(0.7)
            else:
                scheduler.step()
        if model(x).shape != (8, 2):
            raise AssertionError("RNN output shape changed")
    print(f"Frozen REP-006, RGB-flip cache, 3600/200 split, {count:,} parameters, LR policies: OK")


def run_dir(strategy_id: str, seed: int) -> Path:
    return OUTPUT / strategy_id / f"seed{seed}"


def archive_run(strategy_id: str, seed: int) -> None:
    source = run_dir(strategy_id, seed)
    target = REPORT / "experiments" / strategy_id / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
        shutil.copy2(source / name, target / name)


def train_one(strategy_id: str, seed: int, prepared: dict, channels: tuple[int, ...], directory: Path) -> None:
    seed_everything(seed)
    device = select_device("auto")
    split = prepared["split"]
    mean, std = t1_normalization(prepared["train_indices"])
    original = make_dataset(prepared["train_indices"], channels, mean, std)
    flipped = CachedRepresentationDataset(FLIP_CACHE / "features.npy", FLIP_CACHE / "labels.npy",
                                           prepared["train_indices"], channels, mean, std)
    train_dataset = ConcatDataset((original, flipped))
    validation_dataset = make_dataset(prepared["validation_indices"], channels, mean, std)
    loader_args = {"batch_size": 32, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)
    model = VisionRNN2DClassifier(ARCHITECTURE_ID).to(device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    baseline = json.loads((BASE_REPORT / f"seed{seed}/metrics.json").read_text(encoding="utf-8"))
    if count != baseline["parameter_count"] or len(train_dataset) != 3600 or len(validation_dataset) != 200:
        raise AssertionError("Frozen protocol mismatch")
    initial_lr, scheduler_config = policy(strategy_id)
    flip_hash = sha256_file(FLIP_CACHE / "manifest.json")
    config = {
        "batch_id": BATCH_ID, "strategy_id": strategy_id, "architecture_id": ARCHITECTURE_ID,
        "architecture_alias": "RNN2D-ARCH-BASE", "representation_id": REPRESENTATION_ID,
        "representation_shape": [190, 7, 7], **ARCHITECTURES[ARCHITECTURE_ID],
        "augmentation": "horizontal_flip", "dropout_probability": 0.0, "seed": seed,
        "training_sample_count": 3600, "validation_samples": 200, "parameter_count": count,
        "class_to_idx": {"cat": 0, "dog": 1}, "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": sha256_file(HANDCRAFTED / "manifest.json"),
        "flipped_cache_manifest_sha256": flip_hash, "optimizer": "AdamW",
        "learning_rate": initial_lr, "weight_decay": 1e-4, "batch_size": 32,
        "max_epochs": 40, "early_stopping_patience": 8, "gradient_clip_norm": 1.0,
        "loss": "CrossEntropyLoss", "scheduler": scheduler_config,
        "scheduler_step": "after_validation" if strategy_id == T3 else (
            "after_training_epoch" if strategy_id == T2 else None),
        "checkpoint_selection": "highest internal-validation accuracy; tie lower validation loss",
        "device": str(device), "code_commit": current_commit(),
    }
    write_json(directory / "config.json", config)
    write_json(directory / "normalization.json", {
        "source": "1800 original and 1800 flipped training maps",
        "split_sha256": split["sha256"], "mean": mean.tolist(), "std": std.tolist(),
    })
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = make_scheduler(optimizer, scheduler_config)
    history, reductions = [], []
    best_accuracy, best_loss = -1.0, float("inf")
    lowest_loss, stale = float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, 41):
        used_lr = optimizer.param_groups[0]["lr"]
        train = epoch_pass(model, train_loader, criterion, device, optimizer)
        validation = epoch_pass(model, validation_loader, criterion, device)
        if train["samples"] != 3600 or validation["samples"] != 200:
            raise AssertionError("Epoch omitted samples")
        history.append({
            "epoch": epoch, "learning_rate": used_lr,
            "train_loss": train["loss"], "train_accuracy": train["accuracy"],
            "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
            "validation_cat_accuracy": validation["cat_accuracy"],
            "validation_dog_accuracy": validation["dog_accuracy"],
        })
        write_history(directory / "history.csv", history)
        if scheduler is not None:
            if strategy_id == T3:
                scheduler.step(validation["loss"])
            else:
                scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]
        if strategy_id == T3 and next_lr < used_lr - 1e-15:
            reductions.append({"epoch": epoch, "learning_rate_after_reduction": next_lr})
        accuracy, loss = validation["accuracy"], validation["loss"]
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_accuracy, best_loss = accuracy, loss
            torch.save({"model_state": model.state_dict(), "strategy_id": strategy_id,
                        "epoch": epoch, "validation_metrics": validation, "config": config,
                        "scheduler_config": scheduler_config,
                        "scheduler_state": scheduler.state_dict() if scheduler is not None else None},
                       directory / "best_model.pt")
        if loss < lowest_loss - 1e-4:
            lowest_loss, stale = loss, 0
        else:
            stale += 1
        print(f"{strategy_id} seed {seed} epoch {epoch:02d}: lr={used_lr:.8g}, "
              f"train={train['accuracy']:.3f}, val={accuracy:.3f}, loss={loss:.4f}", flush=True)
        if stale >= 8:
            break
    fit_seconds = time.perf_counter() - started
    checkpoint = torch.load(directory / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for features, labels in validation_loader:
            logits_all.append(model(features.to(device, non_blocking=True)).cpu())
            labels_all.append(labels)
    labels = torch.cat(labels_all).numpy()
    cached_labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    if not np.array_equal(labels, cached_labels[prepared["validation_indices"]]):
        raise AssertionError("Validation labels differ from frozen split")
    measured = write_predictions(directory / "predictions.csv", split["internal_validation"],
                                 labels, torch.cat(logits_all).numpy())
    if abs(measured["validation_accuracy"] - checkpoint["validation_metrics"]["accuracy"]) > 1e-12:
        raise AssertionError("Reloaded checkpoint changed validation accuracy")
    best_epoch = checkpoint["epoch"]
    diagnostics = {
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "minimum_learning_rate_reached": min(float(item["learning_rate"]) for item in history),
        "learning_rate_at_best_epoch": history[best_epoch - 1]["learning_rate"],
        "number_of_lr_reductions": len(reductions) if strategy_id == T3 else None,
        "epochs_of_lr_reductions": [item["epoch"] for item in reductions] if strategy_id == T3 else [],
        "learning_rates_after_reduction": [item["learning_rate_after_reduction"] for item in reductions]
        if strategy_id == T3 else [],
    }
    write_json(directory / "metrics.json", {
        "experiment_id": f"{BATCH_ID}-{strategy_id}-seed{seed}",
        "strategy_id": strategy_id, "architecture_id": ARCHITECTURE_ID,
        "representation_id": REPRESENTATION_ID, "seed": seed,
        "parameter_count": count, "training_sample_count": 3600,
        "initial_learning_rate": initial_lr, "scheduler_type": scheduler_config["type"] if scheduler_config else "none",
        "weight_decay": 1e-4, "batch_size": 32, "gradient_clip_norm": 1.0,
        "best_epoch": best_epoch, "final_epoch": len(history), "epochs_run": len(history),
        "train_accuracy_at_best_epoch": history[best_epoch - 1]["train_accuracy"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "validation_loss": checkpoint["validation_metrics"]["loss"], **measured,
        "fit_seconds": fit_seconds, "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": config["feature_cache_manifest_sha256"],
        "flipped_cache_manifest_sha256": flip_hash, "code_commit": config["code_commit"],
        "status": "complete", **diagnostics,
    })
    write_json(directory / "train_summary.json", {
        "strategy_id": strategy_id, "seed": seed, "epochs_run": len(history),
        "best_epoch": best_epoch, "parameter_count": count, "training_sample_count": 3600,
        "fit_seconds": fit_seconds, "best_validation_metrics": checkpoint["validation_metrics"],
        "scheduler_config": scheduler_config, "scheduler_state_at_best": checkpoint["scheduler_state"],
        "device": str(device), **diagnostics,
    })


def run_one(strategy_id: str, seed: int, prepared: dict, channels: tuple[int, ...]) -> None:
    directory = run_dir(strategy_id, seed)
    if (directory / "metrics.json").exists():
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if (metrics.get("status") != "complete" or metrics.get("strategy_id") != strategy_id
                or metrics.get("seed") != seed or metrics.get("split_sha256") != prepared["split"]["sha256"]):
            raise AssertionError(f"Existing run incompatible: {directory}")
        archive_run(strategy_id, seed)
        print(f"Reused complete {strategy_id} seed {seed}", flush=True)
        return
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Partial run exists: {directory}")
    directory.mkdir(parents=True)
    try:
        train_one(strategy_id, seed, prepared, channels, directory)
        archive_run(strategy_id, seed)
    except Exception as error:
        write_json(directory / "failure.json", {"strategy_id": strategy_id, "seed": seed,
                                                  "error": str(error), "traceback": traceback.format_exc()})
        raise


def run_all() -> None:
    channels, prepared = context()
    validate_flip_cache()
    for strategy_id in STRATEGIES:
        for seed in SEEDS:
            run_one(strategy_id, seed, prepared[seed], channels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run-all", "summarize", "audit"))
    command = parser.parse_args().command
    if command in ("summarize", "audit"):
        from rnn_lr_optimization_report import audit, summarize
        {"summarize": summarize, "audit": audit}[command]()
    else:
        {"smoke": smoke, "run-all": run_all}[command]()


if __name__ == "__main__":
    main()
