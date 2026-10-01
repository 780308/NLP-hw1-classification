"""Shared experiment helpers for DNN, CNN, and RNN runs."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    seen = 0
    confusion = [[0, 0], [0, 0]]

    with torch.set_grad_enabled(training):
        for batch_number, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            loss_sum += loss.item() * batch_size
            seen += batch_size
            predictions = logits.argmax(dim=1)
            for actual, predicted in zip(targets.tolist(), predictions.tolist()):
                confusion[actual][predicted] += 1
            if max_batches is not None and batch_number >= max_batches:
                break

    if seen == 0:
        raise ValueError("Cannot compute metrics for an empty loader")
    correct_cat = confusion[0][0]
    correct_dog = confusion[1][1]
    total_cat = sum(confusion[0])
    total_dog = sum(confusion[1])
    return {
        "loss": loss_sum / seen,
        "accuracy": (correct_cat + correct_dog) / seen,
        "cat_accuracy": correct_cat / total_cat if total_cat else None,
        "dog_accuracy": correct_dog / total_dog if total_dog else None,
        "samples": seen,
        "confusion_matrix": confusion,
    }


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_history(path: Path, history: list[dict]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict], path: Path, model_name: str) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Internal validation")
    axes[0].set(title=f"{model_name.upper()} loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="Internal validation")
    axes[1].set(title=f"{model_name.upper()} accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_confusion(confusion: list[list[int]], path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.imshow(confusion, cmap="Blues")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["cat", "dog"],
             yticklabels=["cat", "dog"], xlabel="Predicted", ylabel="Actual", title=title)
    dark_cell_threshold = max(max(row) for row in confusion) * 0.7
    for row in range(2):
        for column in range(2):
            count = confusion[row][column]
            axis.text(
                column, row, str(count), ha="center", va="center",
                color="white" if count > dark_cell_threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
