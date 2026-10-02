"""Run three frozen training interventions for the selected two-axis RNN."""

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

BATCH_ID = "RNN-TRAIN-001"
ARCHITECTURE_ID = "RNN2D-ARCH-L"
REPRESENTATION_ID = "REP-006-FUSION"
SEEDS = (42, 123, 2026)
T0 = "RNN-TRAIN-T0-BASE"
STRATEGIES = ("RNN-TRAIN-T1-FLIP", "RNN-TRAIN-T2-DROPOUT", "RNN-TRAIN-T3-WD")
ALL_STRATEGIES = (T0, *STRATEGIES)
COMBINATION_STRATEGY = "RNN-TRAIN-T4-FLIP-WD"
OUTPUT = ROOT / "outputs/rnn_training" / BATCH_ID
REPORT = ROOT / "report/rnn_training"
BASE_REPORT = ROOT / "report/rnn2d_architecture/experiments" / ARCHITECTURE_ID
ARTIFACTS = ("config.json", "normalization.json", "history.csv", "metrics.json",
             "train_summary.json", "predictions.csv")


def strategy_settings(strategy_id: str) -> dict:
    if strategy_id not in (*STRATEGIES, COMBINATION_STRATEGY):
        raise ValueError(f"Unknown training strategy: {strategy_id}")
    return {
        "augmentation": "horizontal_flip" if strategy_id in (STRATEGIES[0], COMBINATION_STRATEGY) else "none",
        "dropout_probability": 0.10 if strategy_id == STRATEGIES[1] else 0.0,
        "weight_decay": 5e-4 if strategy_id in (STRATEGIES[2], COMBINATION_STRATEGY) else 1e-4,
    }


def model_for(strategy_id: str) -> VisionRNN2DClassifier:
    return VisionRNN2DClassifier(ARCHITECTURE_ID,
                                 dropout_probability=strategy_settings(strategy_id)["dropout_probability"])


def smoke() -> None:
    channels, prepared = context()
    validate_flip_cache()
    original = np.load(HANDCRAFTED / "features.npy", mmap_mode="r")
    flipped = np.load(FLIP_CACHE / "features.npy", mmap_mode="r")
    labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    flip_labels = np.load(FLIP_CACHE / "labels.npy", mmap_mode="r")
    indices = prepared[42]["train_indices"]
    if (channels != tuple(range(190)) or original.shape != (2000, 190, 7, 7)
            or flipped.shape != original.shape
            or not np.array_equal(labels[indices], flip_labels[indices])):
        raise AssertionError("Original/flipped REP-006 cache or label alignment changed")
    mean, std = t1_normalization(indices)
    original_train = make_dataset(indices, channels, mean, std)
    flipped_train = CachedRepresentationDataset(FLIP_CACHE / "features.npy", FLIP_CACHE / "labels.npy",
                                                 indices, channels, mean, std)
    validation = make_dataset(prepared[42]["validation_indices"], channels, mean, std)
    if (len(ConcatDataset((original_train, flipped_train))) != 3600 or len(validation) != 200
            or not torch.allclose(original_train[0][0], torch.as_tensor((original[indices[0]] - mean[:, None, None]) / std[:, None, None]))):
        raise AssertionError("T1 training/validation selection or normalization changed")
    features, targets = next(iter(DataLoader(validation, batch_size=8)))
    baseline_count = sum(p.numel() for p in VisionRNN2DClassifier(ARCHITECTURE_ID).parameters())
    for strategy_id in STRATEGIES:
        model = model_for(strategy_id)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count != baseline_count or model(features).shape != (8, 2):
            raise AssertionError("Frozen RNN model size or output changed")
        nn.CrossEntropyLoss()(model(features), targets).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                      weight_decay=strategy_settings(strategy_id)["weight_decay"])
        optimizer.step()
        if optimizer.param_groups[0]["weight_decay"] != strategy_settings(strategy_id)["weight_decay"]:
            raise AssertionError("Weight decay changed")
    model = model_for(STRATEGIES[1])
    for block in model.blocks:
        drops = [layer for layer in block.modules() if isinstance(layer, nn.Dropout)]
        if len(drops) != 3 or any(layer.p != 0.10 for layer in drops):
            raise AssertionError("T2 must use exactly three p=0.10 Dropouts per block")
        model.train()
        train_sample = block.fusion_dropout(torch.ones(128, 320))
        model.eval()
        eval_sample = block.fusion_dropout(torch.ones(128, 320))
        if torch.equal(train_sample, eval_sample) or not torch.equal(eval_sample, torch.ones_like(eval_sample)):
            raise AssertionError("T2 train/eval dropout behavior changed")
    print("T1 aligned RGB-flip cache, T2 exact dropout, T3 weight decay, and all model updates OK", flush=True)


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


def train_one(strategy_id: str, seed: int, prepared: dict, channels: tuple[int, ...], directory: Path,
              batch_id: str = BATCH_ID) -> None:
    seed_everything(seed)
    device = select_device("auto")
    split = prepared["split"]
    settings = strategy_settings(strategy_id)
    t1 = settings["augmentation"] == "horizontal_flip"
    mean, std = t1_normalization(prepared["train_indices"]) if t1 else (prepared["mean"], prepared["std"])
    train_original = make_dataset(prepared["train_indices"], channels, mean, std)
    if t1:
        train_flipped = CachedRepresentationDataset(FLIP_CACHE / "features.npy", FLIP_CACHE / "labels.npy",
                                                      prepared["train_indices"], channels, mean, std)
        train_dataset = ConcatDataset((train_original, train_flipped))
    else:
        train_dataset = train_original
    validation_dataset = make_dataset(prepared["validation_indices"], channels, mean, std)
    loader_args = {"batch_size": 32, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)
    model = model_for(strategy_id).to(device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    baseline_count = json.loads((BASE_REPORT / f"seed{seed}" / "metrics.json").read_text(encoding="utf-8"))["parameter_count"]
    if count != baseline_count or len(train_dataset) != (3600 if t1 else 1800) or len(validation_dataset) != 200:
        raise AssertionError("RNN training strategy altered model size or sample counts")
    flip_hash = sha256_file(FLIP_CACHE / "manifest.json") if t1 else ""
    config = {
        "batch_id": batch_id, "strategy_id": strategy_id, "architecture_id": ARCHITECTURE_ID,
        "representation_id": REPRESENTATION_ID, "representation_shape": [190, 7, 7],
        **ARCHITECTURES[ARCHITECTURE_ID], **settings,
        "seed": seed, "training_sample_count": len(train_dataset), "validation_samples": 200,
        "parameter_count": count, "class_to_idx": {"cat": 0, "dog": 1},
        "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": sha256_file(HANDCRAFTED / "manifest.json"),
        "flipped_cache_manifest_sha256": flip_hash,
        "optimizer": "AdamW", "learning_rate": 3e-4, "batch_size": 32,
        "max_epochs": 40, "early_stopping_patience": 8, "gradient_clip_norm": 1.0,
        "loss": "CrossEntropyLoss", "scheduler": None,
        "checkpoint_selection": "highest internal-validation accuracy; tie lower validation loss",
        "device": str(device), "code_commit": current_commit(),
    }
    write_json(directory / "config.json", config)
    write_json(directory / "normalization.json", {
        "source": "1800 original and 1800 flipped training maps" if t1 else "1800 original training maps",
        "split_sha256": split["sha256"], "mean": mean.tolist(), "std": std.tolist(),
    })
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=settings["weight_decay"])
    history = []
    best_accuracy, best_loss = -1.0, float("inf")
    lowest_loss, stale = float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, 41):
        train = epoch_pass(model, train_loader, criterion, device, optimizer)
        validation = epoch_pass(model, validation_loader, criterion, device)
        if train["samples"] != len(train_dataset) or validation["samples"] != 200:
            raise AssertionError("Strategy epoch omitted samples")
        history.append({
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train["loss"], "train_accuracy": train["accuracy"],
            "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
            "validation_cat_accuracy": validation["cat_accuracy"],
            "validation_dog_accuracy": validation["dog_accuracy"],
        })
        write_history(directory / "history.csv", history)
        accuracy, loss = validation["accuracy"], validation["loss"]
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_accuracy, best_loss = accuracy, loss
            torch.save({"model_state": model.state_dict(), "strategy_id": strategy_id,
                        "epoch": epoch, "validation_metrics": validation, "config": config},
                       directory / "best_model.pt")
        if loss < lowest_loss - 1e-4:
            lowest_loss, stale = loss, 0
        else:
            stale += 1
        print(f"{strategy_id} seed {seed} epoch {epoch:02d}: train={train['accuracy']:.3f}, val={accuracy:.3f}, loss={loss:.4f}", flush=True)
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
        raise AssertionError("Reloaded RNN strategy checkpoint changed validation accuracy")
    best_epoch = checkpoint["epoch"]
    write_json(directory / "metrics.json", {
        "experiment_id": f"{batch_id}-{strategy_id}-seed{seed}",
        "strategy_id": strategy_id, "architecture_id": ARCHITECTURE_ID,
        "representation_id": REPRESENTATION_ID, "seed": seed,
        "parameter_count": count, "training_sample_count": len(train_dataset),
        "best_epoch": best_epoch, "epochs_run": len(history),
        "train_accuracy_at_best_epoch": history[best_epoch - 1]["train_accuracy"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "validation_loss": checkpoint["validation_metrics"]["loss"], **measured,
        "fit_seconds": fit_seconds, "learning_rate": 3e-4, "weight_decay": settings["weight_decay"],
        "dropout_probability": settings["dropout_probability"],
        "augmentation": settings["augmentation"], "gradient_clip_norm": 1.0,
        "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": config["feature_cache_manifest_sha256"],
        "flipped_cache_manifest_sha256": flip_hash,
        "code_commit": config["code_commit"], "status": "complete",
    })
    write_json(directory / "train_summary.json", {
        "strategy_id": strategy_id, "seed": seed, "epochs_run": len(history),
        "best_epoch": best_epoch, "parameter_count": count,
        "training_sample_count": len(train_dataset), "fit_seconds": fit_seconds,
        "best_validation_metrics": checkpoint["validation_metrics"], "device": str(device),
    })


def run_one(strategy_id: str, seed: int, prepared: dict, channels: tuple[int, ...]) -> None:
    directory = run_dir(strategy_id, seed)
    if (directory / "metrics.json").exists():
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if (metrics.get("status") != "complete" or metrics.get("strategy_id") != strategy_id
                or metrics.get("seed") != seed or metrics.get("split_sha256") != prepared["split"]["sha256"]):
            raise AssertionError(f"Existing strategy run incompatible: {directory}")
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
        from rnn_training_strategy_report import audit, summarize
        {"summarize": summarize, "audit": audit}[command]()
    else:
        {"smoke": smoke, "run-all": run_all}[command]()


if __name__ == "__main__":
    main()
