"""Fixed linear and MLP probes with complete internal-validation records."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader

from training import run_epoch, seed_everything, seed_worker, select_device, write_history, write_json

from .cache import HANDCRAFTED, RAW, current_commit, sha256_file
from .dataset import CachedRawDataset, CachedRepresentationDataset
from .registry import RAW_REPRESENTATION_ID, Representation


class FixedMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    cat_total, dog_total = matrix.sum(axis=1)
    return {
        "validation_accuracy": float((predictions == labels).mean()),
        "cat_accuracy": float(matrix[0, 0] / cat_total),
        "dog_accuracy": float(matrix[1, 1] / dog_total),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": matrix.tolist(),
    }


def _write_predictions(path: Path, filenames: list[str], labels: np.ndarray, predictions: np.ndarray, margins: np.ndarray) -> None:
    if not (len(filenames) == len(labels) == len(predictions) == len(margins)):
        raise AssertionError("Prediction rows are not aligned")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "true_label", "predicted_label", "score_or_margin", "correct"))
        for name, label, prediction, margin in zip(filenames, labels, predictions, margins):
            writer.writerow((name, int(label), int(prediction), float(margin), int(label == prediction)))


def _normalized_matrix(indices: list[int], representation: Representation | None, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if representation is None:
        images = np.load(RAW / "features.npy", mmap_mode="r")
        return (images[indices].astype(np.float32) / 127.5 - 1.0).reshape(len(indices), -1)
    features = np.load(HANDCRAFTED / "features.npy", mmap_mode="r")
    values = np.asarray(features[indices][:, representation.channels], dtype=np.float32)
    values = (values - mean[None, :, None, None]) / std[None, :, None, None]
    return values.reshape(len(indices), -1)


def probe_context(batch_id: str, representation_id: str, probe: str, seed: int, split: dict, representation: Representation | None) -> dict:
    shape = [3, 64, 64] if representation is None else list(representation.shape)
    manifest_path = (RAW if representation is None else HANDCRAFTED) / "manifest.json"
    return {
        "experiment_id": f"{batch_id}-{representation_id}-{probe}-seed{seed}",
        "batch_id": batch_id,
        "representation_id": representation_id,
        "probe": probe,
        "seed": seed,
        "split_sha256": split["sha256"],
        "representation_shape": shape,
        "flatten_dim": int(np.prod(shape)),
        "train_samples": len(split["train"]),
        "validation_samples": len(split["internal_validation"]),
        "feature_cache_manifest_sha256": sha256_file(manifest_path) if not representation_id.startswith("CTRL-LBOX64") and manifest_path.exists() else "",
        "code_commit": current_commit(),
    }


def run_linear(
    batch_id: str,
    representation_id: str,
    representation: Representation | None,
    split: dict,
    train_indices: list[int],
    validation_indices: list[int],
    labels: np.ndarray,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = probe_context(batch_id, representation_id, "LinearSVC", split["seed"], split, representation)
    config = {**context, "C": 1.0, "class_weight": None, "dual": "auto", "max_iter": 10000, "random_state": split["seed"]}
    normal = {"source": "fixed raw control" if representation is None else "training indices only", "mean": [0.5] * 3 if mean is None else mean.tolist(), "std": [0.5] * 3 if std is None else std.tolist()}
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "normalization.json", normal)
    x_train = _normalized_matrix(train_indices, representation, mean, std)
    x_validation = _normalized_matrix(validation_indices, representation, mean, std)
    y_train, y_validation = labels[train_indices], labels[validation_indices]
    classifier = LinearSVC(C=1.0, class_weight=None, dual="auto", max_iter=10000, random_state=split["seed"])
    started = time.perf_counter()
    classifier.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    iterations = int(np.max(classifier.n_iter_))
    predictions = classifier.predict(x_validation)
    margins = classifier.decision_function(x_validation)
    metrics = {
        **context,
        "parameter_count": None,
        "best_epoch": None,
        "train_accuracy": float((classifier.predict(x_train) == y_train).mean()),
        **_metrics(y_validation, predictions),
        "fit_seconds": fit_seconds,
        "iterations": iterations,
        "converged": iterations < 10000,
        "status": "complete",
    }
    _write_predictions(output_dir / "predictions.csv", split["internal_validation"], y_validation, predictions, margins)
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def run_mlp(
    batch_id: str,
    representation_id: str,
    representation: Representation | None,
    split: dict,
    train_indices: list[int],
    validation_indices: list[int],
    labels: np.ndarray,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    output_dir: Path,
    train_dataset=None,
    validation_dataset=None,
    train_augmentation: str = "none",
    num_workers: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = split["seed"]
    seed_everything(seed)
    device = select_device("auto")
    context = probe_context(batch_id, representation_id, "MLP", seed, split, representation)
    if train_dataset is None:
        if representation is None:
            train_dataset = CachedRawDataset(RAW / "features.npy", RAW / "labels.npy", train_indices)
            validation_dataset = CachedRawDataset(RAW / "features.npy", RAW / "labels.npy", validation_indices)
        else:
            train_dataset = CachedRepresentationDataset(HANDCRAFTED / "features.npy", HANDCRAFTED / "labels.npy", train_indices, representation.channels, mean, std)
            validation_dataset = CachedRepresentationDataset(HANDCRAFTED / "features.npy", HANDCRAFTED / "labels.npy", validation_indices, representation.channels, mean, std)
    normal = {"source": "fixed raw control" if representation is None else "training indices only", "mean": [0.5] * 3 if mean is None else mean.tolist(), "std": [0.5] * 3 if std is None else std.tolist()}
    config = {
        **context,
        "architecture": f"Flatten -> {context['flatten_dim']} -> 256 -> 64 -> 2",
        "dropout": 0.3, "loss": "CrossEntropyLoss", "optimizer": "AdamW",
        "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 32,
        "epochs_max": 30, "early_stopping_patience": 5,
        "checkpoint_selection": "highest validation accuracy; tie lower validation loss",
        "train_augmentation": train_augmentation, "device": str(device),
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "normalization.json", normal)
    loader_kwargs = {"batch_size": 32, "num_workers": num_workers, "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker, "persistent_workers": num_workers > 0}
    train_loader = DataLoader(train_dataset, shuffle=True, generator=torch.Generator().manual_seed(seed), **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = FixedMLP(context["flatten_dim"]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best_accuracy, best_loss = -1.0, float("inf")
    lowest_validation_loss, stale_epochs = float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, 31):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_cat_accuracy": validation_metrics["cat_accuracy"],
            "validation_dog_accuracy": validation_metrics["dog_accuracy"],
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)
        accuracy, loss = validation_metrics["accuracy"], validation_metrics["loss"]
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_accuracy, best_loss = accuracy, loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation_metrics": validation_metrics, "config": config}, output_dir / "best_model.pt")
        if loss < lowest_validation_loss - 1e-4:
            lowest_validation_loss, stale_epochs = loss, 0
        else:
            stale_epochs += 1
        print(f"{context['experiment_id']} epoch {epoch:02d}: train={train_metrics['accuracy']:.3f}, val={accuracy:.3f}, loss={loss:.4f}", flush=True)
        if stale_epochs >= 5:
            break
    fit_seconds = time.perf_counter() - started
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    observed_labels, predictions, margins = [], [], []
    with torch.no_grad():
        for images, targets in validation_loader:
            logits = model(images.to(device, non_blocking=True))
            observed_labels.extend(targets.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            margins.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
    observed_labels = np.asarray(observed_labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if not np.array_equal(observed_labels, labels[validation_indices]):
        raise AssertionError("Validation prediction order differs from split")
    measured = _metrics(observed_labels, predictions)
    if abs(measured["validation_accuracy"] - checkpoint["validation_metrics"]["accuracy"]) > 1e-9:
        raise AssertionError("Reloaded checkpoint changed validation accuracy")
    _write_predictions(output_dir / "predictions.csv", split["internal_validation"], observed_labels, predictions, np.asarray(margins))
    best_epoch = checkpoint["epoch"]
    metrics = {
        **context, "parameter_count": parameter_count, "best_epoch": best_epoch,
        "train_accuracy": history[best_epoch - 1]["train_accuracy"],
        **measured, "validation_loss": checkpoint["validation_metrics"]["loss"],
        "fit_seconds": fit_seconds, "status": "complete",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "train_summary.json", {"epochs_run": len(history), "best_epoch": best_epoch, "best_validation_metrics": checkpoint["validation_metrics"], "fit_seconds": fit_seconds, "parameter_count": parameter_count, "device": str(device)})
    return metrics


def tiny_overfit(split: dict) -> float:
    """Check that a HOG probe can memorize 64 balanced training examples."""
    from .cache import ensure_split, normalization, split_indices
    from .registry import get_representation

    if split["seed"] != 42:
        raise ValueError("Tiny diagnostic uses seed 42")
    _, _, filenames = split_indices(split)
    lookup = {name: index for index, name in enumerate(filenames)}
    cats = [lookup[name] for name in split["train"] if name.startswith("cat.")][:32]
    dogs = [lookup[name] for name in split["train"] if name.startswith("dog.")][:32]
    indices = cats + dogs
    representation = get_representation("REP-001-HOG")
    train_indices, _, _ = split_indices(split)
    mean, std = normalization(train_indices, representation.channels)
    dataset = CachedRepresentationDataset(HANDCRAFTED / "features.npy", HANDCRAFTED / "labels.npy", indices, representation.channels, mean, std)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(42))
    seed_everything(42)
    device = select_device("auto")
    model = FixedMLP(representation.flatten_dim, dropout=0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, 101):
        run_epoch(model, loader, nn.CrossEntropyLoss(), device, optimizer)
        evaluation = run_epoch(model, DataLoader(dataset, batch_size=32), nn.CrossEntropyLoss(), device)
        if evaluation["accuracy"] >= 0.95:
            print(f"Tiny overfit: {evaluation['accuracy']:.3f} at epoch {epoch}", flush=True)
            return evaluation["accuracy"]
    raise RuntimeError(f"Tiny overfit failed: training accuracy {evaluation['accuracy']:.3f}")
