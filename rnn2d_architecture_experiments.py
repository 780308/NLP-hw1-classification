"""Compare three scales of the frozen two-axis residual standard-RNN classifier."""

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
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from cnn_architecture_experiments import write_predictions
from dnn_architecture_experiments import frozen_representation, make_dataset, prepare_seed
from models.rnn2d import (
    ARCHITECTURE_IDS, ARCHITECTURES, VisionRNN2DClassifier,
    horizontal_sequences, restore_vertical, vertical_sequences,
)
from representations.cache import HANDCRAFTED, ROOT, current_commit, sha256_file, validate_cache
from training import seed_everything, select_device, write_history, write_json

BATCH_ID = "RNN-ARCH-002"
SEEDS = (42, 123, 2026)
REPRESENTATION_ID = "REP-006-FUSION"
OUTPUT = ROOT / "outputs/rnn2d_architecture" / BATCH_ID
REPORT = ROOT / "report/rnn2d_architecture"
ARTIFACTS = ("config.json", "normalization.json", "history.csv", "metrics.json",
             "train_summary.json", "predictions.csv")
RESULT_FIELDS = (
    "experiment_id", "architecture_id", "spatial_representation", "seed",
    "parameter_count", "embed_dim", "rnn_hidden_size", "num_blocks",
    "best_epoch", "train_accuracy_at_best_epoch", "final_train_accuracy",
    "validation_loss", "validation_accuracy", "cat_accuracy", "dog_accuracy",
    "balanced_accuracy", "macro_f1", "fit_seconds", "gradient_clip_norm",
    "split_sha256", "feature_cache_manifest_sha256", "code_commit", "status", "source_run",
)


def context() -> tuple[tuple[int, ...], dict[int, dict]]:
    validate_cache()
    channels = frozen_representation()
    if channels != tuple(range(190)):
        raise AssertionError("REP-006 channel order changed")
    return channels, {seed: prepare_seed(seed, channels) for seed in SEEDS}


def epoch_pass(model: nn.Module, loader: DataLoader, criterion: nn.Module,
               device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> dict:
    training = optimizer is not None
    model.train(training)
    confusion = [[0, 0], [0, 0]]
    loss_sum = seen = 0
    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            batch_size = len(labels)
            seen += batch_size
            loss_sum += loss.item() * batch_size
            predicted = logits.argmax(dim=1)
            for actual, guess in zip(labels.tolist(), predicted.tolist()):
                confusion[actual][guess] += 1
    if seen == 0:
        raise AssertionError("Empty RNN training or validation loader")
    return {"loss": loss_sum / seen, "accuracy": (confusion[0][0] + confusion[1][1]) / seen,
            "cat_accuracy": confusion[0][0] / sum(confusion[0]),
            "dog_accuracy": confusion[1][1] / sum(confusion[1]),
            "samples": seen, "confusion_matrix": confusion}


def smoke() -> None:
    channels, prepared = context()
    batch = next(iter(DataLoader(make_dataset(prepared[42]["train_indices"][:8], channels,
                                              prepared[42]["mean"], prepared[42]["std"]), batch_size=8)))
    features, labels = batch
    marker = torch.arange(49).reshape(1, 7, 7, 1)
    if (horizontal_sequences(marker).shape != (7, 7, 1)
            or vertical_sequences(marker).shape != (7, 7, 1)
            or not torch.equal(restore_vertical(vertical_sequences(marker), 1), marker)
            or marker[0, 2, 4, 0] != horizontal_sequences(marker)[2, 4, 0]
            or marker[0, 2, 4, 0] != vertical_sequences(marker)[4, 2, 0]):
        raise AssertionError("Horizontal/vertical spatial coordinates are misaligned")
    for architecture_id in ARCHITECTURE_IDS:
        model = VisionRNN2DClassifier(architecture_id)
        spec = ARCHITECTURES[architecture_id]
        tokens = model.projection(features)
        if tokens.shape != (8, 7, 7, spec["embed_dim"]):
            raise AssertionError("Token projection shape mismatch")
        for block in model.blocks:
            if (block.horizontal_rnn is block.vertical_rnn
                    or set(id(p) for p in block.horizontal_rnn.parameters())
                    & set(id(p) for p in block.vertical_rnn.parameters())):
                raise AssertionError("Horizontal and vertical RNN parameters are shared")
            tokens = block(tokens)
            if tokens.shape != (8, 7, 7, spec["embed_dim"]):
                raise AssertionError("Residual block shape mismatch")
        recurrent = [layer for layer in model.modules() if isinstance(layer, nn.RNN)]
        if (len(recurrent) != 2 * spec["num_blocks"]
                or len({id(p) for layer in recurrent for p in layer.parameters()})
                   != sum(len(tuple(layer.parameters())) for layer in recurrent)
                or any(layer.nonlinearity != "tanh" or not layer.bidirectional
                       or not layer.batch_first or layer.dropout != 0 for layer in recurrent)
                or any(isinstance(layer, (nn.Conv2d, nn.LSTM, nn.GRU)) for layer in model.modules())
                or model(features).shape != (8, 2)):
            raise AssertionError(f"RNN2D model topology mismatch: {architecture_id}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True)
        nn.CrossEntropyLoss()(model(features), labels).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{architecture_id}: forward/backward/clip/step OK; {count:,} parameters", flush=True)


def tiny_overfit() -> None:
    channels, prepared = context()
    indices = prepared[42]["train_indices"]
    labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    subset = [index for index in indices if labels[index] == 0][:32]
    subset += [index for index in indices if labels[index] == 1][:32]
    if len(subset) != 64:
        raise AssertionError("Tiny diagnostic needs 32 training examples per class")
    seed_everything(42)
    device = select_device("auto")
    dataset = make_dataset(subset, channels, prepared[42]["mean"], prepared[42]["std"])
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(42))
    inspect_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model = VisionRNN2DClassifier("RNN2D-ARCH-S").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(1, 201):
        epoch_pass(model, train_loader, criterion, device, optimizer)
        measured = epoch_pass(model, inspect_loader, criterion, device)
        if measured["accuracy"] >= 0.95:
            write_json(REPORT / "tiny_overfit_diagnostic.json", {
                "architecture_id": "RNN2D-ARCH-S", "seed": 42, "samples": 64,
                "epoch": epoch, "training_accuracy_on_subset": measured["accuracy"],
                "formal_result": False, "status": "passed",
            })
            print(f"Tiny overfit diagnostic passed at epoch {epoch}: {measured['accuracy']:.2%}", flush=True)
            return
    raise AssertionError(f"RNN2D-ARCH-S only fit {measured['accuracy']:.2%} of 64 training images")


def run_dir(architecture_id: str, seed: int) -> Path:
    return OUTPUT / architecture_id / f"seed{seed}"


def archive_run(architecture_id: str, seed: int) -> None:
    source = run_dir(architecture_id, seed)
    target = REPORT / "experiments" / architecture_id / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    for filename in ARTIFACTS:
        if not (source / filename).exists():
            raise FileNotFoundError(f"Incomplete RNN run: {source / filename}")
        shutil.copy2(source / filename, target / filename)


def train_one(architecture_id: str, seed: int, prepared: dict, channels: tuple[int, ...], directory: Path) -> None:
    seed_everything(seed)
    device = select_device("auto")
    spec = ARCHITECTURES[architecture_id]
    split = prepared["split"]
    train_dataset = make_dataset(prepared["train_indices"], channels, prepared["mean"], prepared["std"])
    validation_dataset = make_dataset(prepared["validation_indices"], channels, prepared["mean"], prepared["std"])
    loader_kwargs = {"batch_size": 32, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = VisionRNN2DClassifier(architecture_id).to(device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = {
        "batch_id": BATCH_ID, "architecture_id": architecture_id,
        "representation_id": REPRESENTATION_ID, "representation_shape": [190, 7, 7],
        **spec, "seed": seed, "train_samples": 1800, "validation_samples": 200,
        "class_to_idx": {"cat": 0, "dog": 1}, "parameter_count": count,
        "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": sha256_file(HANDCRAFTED / "manifest.json"),
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "batch_size": 32, "max_epochs": 40, "early_stopping_patience": 8,
        "gradient_clip_norm": 1.0, "rnn_dropout": 0.0,
        "loss": "CrossEntropyLoss", "augmentation": "none", "scheduler": None,
        "checkpoint_selection": "highest internal-validation accuracy; tie lower validation loss",
        "device": str(device), "code_commit": current_commit(),
    }
    write_json(directory / "config.json", config)
    write_json(directory / "normalization.json", {
        "source": "1800 training indices only", "split_sha256": split["sha256"],
        "mean": prepared["mean"].tolist(), "std": prepared["std"].tolist(),
    })
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = []
    best_accuracy, best_loss = -1.0, float("inf")
    lowest_loss, stale = float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, 41):
        train = epoch_pass(model, train_loader, criterion, device, optimizer)
        validation = epoch_pass(model, validation_loader, criterion, device)
        if train["samples"] != 1800 or validation["samples"] != 200:
            raise AssertionError("RNN run skipped training or validation samples")
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
            torch.save({"model_state": model.state_dict(), "architecture_id": architecture_id,
                        "epoch": epoch, "validation_metrics": validation,
                        "config": config}, directory / "best_model.pt")
        if loss < lowest_loss - 1e-4:
            lowest_loss, stale = loss, 0
        else:
            stale += 1
        print(f"{architecture_id} seed {seed} epoch {epoch:02d}: train={train['accuracy']:.3f}, val={accuracy:.3f}, loss={loss:.4f}", flush=True)
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
    logits = torch.cat(logits_all).numpy()
    cached_labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    if not np.array_equal(labels, cached_labels[prepared["validation_indices"]]):
        raise AssertionError("Validation labels differ from fixed split")
    measured = write_predictions(directory / "predictions.csv", split["internal_validation"], labels, logits)
    if abs(measured["validation_accuracy"] - checkpoint["validation_metrics"]["accuracy"]) > 1e-12:
        raise AssertionError("Reloaded RNN checkpoint differs from selected epoch")
    best_epoch = checkpoint["epoch"]
    write_json(directory / "metrics.json", {
        "experiment_id": f"{BATCH_ID}-{architecture_id}-seed{seed}",
        "architecture_id": architecture_id, "representation_id": REPRESENTATION_ID,
        "spatial_representation": "two_axis_7x7_grid", "seed": seed,
        "parameter_count": count, **spec,
        "best_epoch": best_epoch, "epochs_run": len(history),
        "train_accuracy_at_best_epoch": history[best_epoch - 1]["train_accuracy"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "validation_loss": checkpoint["validation_metrics"]["loss"],
        **measured, "fit_seconds": fit_seconds, "gradient_clip_norm": 1.0,
        "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": config["feature_cache_manifest_sha256"],
        "code_commit": config["code_commit"], "status": "complete",
    })
    write_json(directory / "train_summary.json", {
        "architecture_id": architecture_id, "seed": seed, "epochs_run": len(history),
        "best_epoch": best_epoch, "parameter_count": count, "fit_seconds": fit_seconds,
        "best_validation_metrics": checkpoint["validation_metrics"], "device": str(device),
    })


def run_one(architecture_id: str, seed: int, prepared: dict, channels: tuple[int, ...]) -> None:
    directory = run_dir(architecture_id, seed)
    if (directory / "metrics.json").exists():
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if (metrics.get("status") != "complete" or metrics.get("architecture_id") != architecture_id
                or metrics.get("seed") != seed or metrics.get("split_sha256") != prepared["split"]["sha256"]):
            raise AssertionError(f"Existing RNN run incompatible: {directory}")
        archive_run(architecture_id, seed)
        print(f"Reused complete {architecture_id} seed {seed}", flush=True)
        return
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Partial RNN run exists: {directory}")
    directory.mkdir(parents=True)
    try:
        train_one(architecture_id, seed, prepared, channels, directory)
        archive_run(architecture_id, seed)
    except Exception as error:
        write_json(directory / "failure.json", {"architecture_id": architecture_id, "seed": seed,
                                                  "error": str(error), "traceback": traceback.format_exc()})
        raise


def run_all() -> None:
    channels, prepared = context()
    for architecture_id in ARCHITECTURE_IDS:
        for seed in SEEDS:
            run_one(architecture_id, seed, prepared[seed], channels)


def record(architecture_id: str, seed: int) -> tuple[dict, list[dict], Path]:
    directory = REPORT / "experiments" / architecture_id / f"seed{seed}"
    with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if not history or metrics["architecture_id"] != architecture_id or metrics["seed"] != seed:
        raise AssertionError(f"Invalid RNN record: {directory}")
    return metrics, history, directory


def aggregate() -> tuple[list[dict], dict]:
    rows, grouped = [], {}
    for architecture_id in ARCHITECTURE_IDS:
        subset = []
        for seed in SEEDS:
            metrics, _, directory = record(architecture_id, seed)
            row = {key: metrics[key] for key in RESULT_FIELDS if key != "source_run"}
            row["source_run"] = str(directory.relative_to(ROOT)).replace("\\", "/")
            rows.append(row)
            subset.append(metrics)
        accuracies = [item["validation_accuracy"] for item in subset]
        cat = statistics.mean(item["cat_accuracy"] for item in subset)
        dog = statistics.mean(item["dog_accuracy"] for item in subset)
        grouped[architecture_id] = {
            "architecture_id": architecture_id, "parameter_count": subset[0]["parameter_count"],
            "mean_validation_accuracy": statistics.mean(accuracies),
            "sample_std_validation_accuracy": statistics.stdev(accuracies),
            "worst_seed_accuracy": min(accuracies),
            "mean_cat_accuracy": cat, "mean_dog_accuracy": dog,
            "absolute_cat_dog_gap": abs(cat - dog),
            "mean_balanced_accuracy": statistics.mean(item["balanced_accuracy"] for item in subset),
            "mean_macro_f1": statistics.mean(item["macro_f1"] for item in subset),
            "median_best_epoch": int(statistics.median(item["best_epoch"] for item in subset)),
            "mean_fit_seconds": statistics.mean(item["fit_seconds"] for item in subset),
            "seed_accuracies": {str(item["seed"]): item["validation_accuracy"] for item in subset},
        }
    return rows, grouped


def select(grouped: dict) -> dict:
    ranked = sorted(grouped.values(), key=lambda item: (-item["mean_validation_accuracy"], item["architecture_id"]))
    top, second = ranked[:2]
    gap = top["mean_validation_accuracy"] - second["mean_validation_accuracy"]
    if gap >= 0.0075 - 1e-12:
        chosen, rule = top, "mean_accuracy_gap_at_least_0.75_pp"
    else:
        worst_gap = top["worst_seed_accuracy"] - second["worst_seed_accuracy"]
        if abs(worst_gap) >= 0.01 - 1e-12:
            chosen, rule = (top if worst_gap > 0 else second), "worst_seed_gap_at_least_1.0_pp"
        else:
            smaller, larger = sorted((top, second), key=lambda item: (item["parameter_count"], item["architecture_id"]))
            larger_floor = min(larger["mean_cat_accuracy"], larger["mean_dog_accuracy"])
            smaller_floor = min(smaller["mean_cat_accuracy"], smaller["mean_dog_accuracy"])
            if larger_floor - smaller_floor >= 0.02 - 1e-12:
                chosen, rule = larger, "larger_model_class_floor_gain_at_least_2.0_pp"
            else:
                chosen, rule = smaller, "fewer_parameters"
    return {"selected_architecture": chosen["architecture_id"], "selection_rule": rule,
            "top_two_mean_accuracy_gap": gap,
            "ranked_by_mean_accuracy": [item["architecture_id"] for item in ranked]}


def plot_curves() -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 10))
    colors = {42: "#1f77b4", 123: "#ff7f0e", 2026: "#2ca02c"}
    for row, architecture_id in enumerate(ARCHITECTURE_IDS):
        for seed in SEEDS:
            _, history, _ = record(architecture_id, seed)
            epochs = [int(item["epoch"]) for item in history]
            for column, metric in enumerate(("accuracy", "loss")):
                axis = axes[row, column]
                axis.plot(epochs, [float(item[f"train_{metric}"]) for item in history],
                          color=colors[seed], label=f"{seed} train")
                axis.plot(epochs, [float(item[f"validation_{metric}"]) for item in history],
                          color=colors[seed], linestyle="--", label=f"{seed} validation")
        for axis in axes[row]:
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
        axes[row, 0].set_ylabel(f"{architecture_id}\nAccuracy")
        axes[row, 1].set_ylabel(f"{architecture_id}\nCross-entropy")
    figure.suptitle("RNN-ARCH-002: two-axis RNN train and internal-validation trajectories")
    figure.tight_layout()
    figure.savefig(REPORT / "training_curves.png", dpi=150)
    plt.close(figure)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "tiny-overfit", "run-all", "summarize", "audit"))
    command = parser.parse_args().command
    if command in ("summarize", "audit"):
        from rnn2d_architecture_report import audit, summarize
        {"summarize": summarize, "audit": audit}[command]()
    else:
        {"smoke": smoke, "tiny-overfit": tiny_overfit, "run-all": run_all}[command]()


if __name__ == "__main__":
    main()
