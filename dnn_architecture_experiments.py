"""Select a pure DNN architecture on frozen REP-006-FUSION features."""

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
from torch import nn
from torch.utils.data import DataLoader

from models.dnn_architecture import ARCHITECTURE_IDS, INPUT_DIM, ArchitectureDNN
from representations.cache import (
    HANDCRAFTED, ROOT, current_commit, normalization, sha256_file,
    split_indices, stable_sha256, validate_cache,
)
from representations.dataset import CachedRepresentationDataset
from representations.probes import _metrics, _write_predictions
from representations.registry import get_representation
from training import run_epoch, seed_everything, seed_worker, select_device, write_history, write_json

BATCH_ID = "DNN-ARCH-001"
SEEDS = (42, 123, 2026)
REPRESENTATION_ID = "REP-006-FUSION"
OUTPUT_ROOT = ROOT / "outputs/dnn_architecture" / BATCH_ID
REPORT_ROOT = ROOT / "report/dnn_architecture"
ARCHIVE_FILES = ("config.json", "normalization.json", "metrics.json", "history.csv", "train_summary.json", "predictions.csv")
RESULT_FIELDS = (
    "architecture_id", "seed", "source_run", "parameter_count", "best_epoch",
    "train_accuracy_at_best_epoch", "final_train_accuracy", "validation_loss",
    "final_validation_loss", "validation_accuracy", "cat_accuracy", "dog_accuracy",
    "balanced_accuracy", "macro_f1", "fit_seconds", "split_sha256",
    "feature_cache_manifest_sha256", "code_commit", "status",
)


def frozen_representation() -> tuple[int, ...]:
    manifest = json.loads((ROOT / "report/stage1/representation_manifest.json").read_text(encoding="utf-8"))
    fusion = manifest.get("conditional_fusion", {})
    if manifest.get("selected_representation_id") != REPRESENTATION_ID:
        raise AssertionError("Stage-I winner is no longer REP-006-FUSION")
    if fusion.get("groups") != ["hog", "lbp", "hsv", "rootsift"] or fusion.get("shape") != [190, 7, 7]:
        raise AssertionError("Frozen feature-group definition changed")
    if fusion.get("flatten_dim") != INPUT_DIM:
        raise AssertionError("Frozen representation dimension changed")
    cache_hash = sha256_file(HANDCRAFTED / "manifest.json")
    if manifest["handcrafted_cache_manifest_sha256"] != cache_hash:
        raise AssertionError("Handcrafted cache manifest changed since Stage I")
    representation = get_representation(REPRESENTATION_ID, tuple(fusion["groups"]))
    if representation.channels != tuple(range(190)):
        raise AssertionError("Frozen fusion must use all 190 channels in canonical order")
    return representation.channels


def baseline_paths(seed: int) -> tuple[Path, Path, Path]:
    directory = ROOT / "report/stage1/experiments/S1B-001" / f"seed{seed}" / REPRESENTATION_ID / "MLP"
    return directory / "metrics.json", directory / "history.csv", directory / "normalization.json"


def split_for_seed(seed: int) -> dict:
    path = ROOT / "report/splits" / f"stage1_seed{seed}.json"
    split = json.loads(path.read_text(encoding="utf-8"))
    recorded_hash = split.pop("sha256")
    if split.get("seed") != seed or stable_sha256(split) != recorded_hash:
        raise AssertionError(f"Stage-I split hash mismatch: {path}")
    split["sha256"] = recorded_hash
    if len(split["train"]) != 1800 or len(split["internal_validation"]) != 200:
        raise AssertionError(f"Stage-I split size changed: {path}")
    if split["class_counts"] != {
        "train": {"cat": 900, "dog": 900},
        "internal_validation": {"cat": 100, "dog": 100},
    }:
        raise AssertionError(f"Stage-I split class balance changed: {path}")
    return split


def prepare_seed(seed: int, channels: tuple[int, ...]) -> dict:
    split = split_for_seed(seed)
    train_indices, validation_indices, _ = split_indices(split)
    mean, std = normalization(train_indices, channels)
    baseline_metrics_path, _, baseline_normalization_path = baseline_paths(seed)
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    baseline_normalization = json.loads(baseline_normalization_path.read_text(encoding="utf-8"))
    if baseline_metrics["split_sha256"] != split["sha256"]:
        raise AssertionError(f"Baseline used a different split for seed {seed}")
    if baseline_metrics["feature_cache_manifest_sha256"] != sha256_file(HANDCRAFTED / "manifest.json"):
        raise AssertionError(f"Baseline used a different cache for seed {seed}")
    if not np.array_equal(mean, np.asarray(baseline_normalization["mean"], dtype=np.float32)):
        raise AssertionError(f"Normalization mean changed for seed {seed}")
    if not np.array_equal(std, np.asarray(baseline_normalization["std"], dtype=np.float32)):
        raise AssertionError(f"Normalization std changed for seed {seed}")
    return {
        "split": split,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "mean": mean,
        "std": std,
    }


def make_dataset(indices: list[int], channels: tuple[int, ...], mean: np.ndarray, std: np.ndarray) -> CachedRepresentationDataset:
    return CachedRepresentationDataset(
        HANDCRAFTED / "features.npy", HANDCRAFTED / "labels.npy",
        indices, channels, mean, std,
    )


def run_directory(architecture_id: str, seed: int) -> Path:
    return OUTPUT_ROOT / architecture_id / f"seed{seed}"


def archive_run(architecture_id: str, seed: int) -> None:
    source = run_directory(architecture_id, seed)
    target = REPORT_ROOT / "experiments" / architecture_id / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVE_FILES:
        path = source / name
        if not path.exists():
            raise FileNotFoundError(f"Incomplete run artifact: {path}")
        shutil.copy2(path, target / name)


def run_one(architecture_id: str, seed: int, prepared: dict, channels: tuple[int, ...]) -> None:
    output_dir = run_directory(architecture_id, seed)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("status") != "complete" or metrics.get("split_sha256") != prepared["split"]["sha256"]:
            raise RuntimeError(f"Existing run is incompatible: {output_dir}")
        archive_run(architecture_id, seed)
        print(f"Reused completed {architecture_id} seed {seed}", flush=True)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Partial run exists; inspect before retrying: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        _train_one(architecture_id, seed, prepared, channels, output_dir)
        archive_run(architecture_id, seed)
    except Exception as error:
        write_json(output_dir / "failure.json", {
            "architecture_id": architecture_id, "seed": seed,
            "status": "failed", "error": str(error), "traceback": traceback.format_exc(),
        })
        raise


def _train_one(architecture_id: str, seed: int, prepared: dict, channels: tuple[int, ...], output_dir: Path) -> None:
    seed_everything(seed)
    device = select_device("auto")
    split = prepared["split"]
    mean, std = prepared["mean"], prepared["std"]
    train_indices, validation_indices = prepared["train_indices"], prepared["validation_indices"]
    labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    train_dataset = make_dataset(train_indices, channels, mean, std)
    validation_dataset = make_dataset(validation_indices, channels, mean, std)
    loader_kwargs = {
        "batch_size": 32, "num_workers": 0,
        "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=torch.Generator().manual_seed(seed), **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = ArchitectureDNN(architecture_id).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "batch_id": BATCH_ID, "architecture_id": architecture_id,
        "representation_id": REPRESENTATION_ID, "representation_shape": [190, 7, 7],
        "flatten_dim": INPUT_DIM, "seed": seed, "split_sha256": split["sha256"],
        "train_samples": 1800, "validation_samples": 200,
        "feature_cache_manifest_sha256": sha256_file(HANDCRAFTED / "manifest.json"),
        "parameter_count": parameter_count, "dropout": 0.3,
        "loss": "CrossEntropyLoss", "optimizer": "AdamW", "learning_rate": 1e-3,
        "weight_decay": 1e-4, "batch_size": 32, "epochs_max": 30,
        "early_stopping_patience": 5,
        "checkpoint_selection": "highest validation accuracy, tie lower validation loss",
        "train_augmentation": "none", "device": str(device), "code_commit": current_commit(),
    }
    config["config_sha256"] = stable_sha256(config)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "normalization.json", {
        "source": "training indices only", "split_sha256": split["sha256"],
        "mean": mean.tolist(), "std": std.tolist(),
    })
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best_accuracy, best_loss = -1.0, float("inf")
    lowest_validation_loss, stale_epochs = float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, 31):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        history.append({
            "epoch": epoch, "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_cat_accuracy": validation_metrics["cat_accuracy"],
            "validation_dog_accuracy": validation_metrics["dog_accuracy"],
        })
        write_history(output_dir / "history.csv", history)
        accuracy, loss = validation_metrics["accuracy"], validation_metrics["loss"]
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_accuracy, best_loss = accuracy, loss
            torch.save({
                "model_state": model.state_dict(), "architecture_id": architecture_id,
                "representation_id": REPRESENTATION_ID, "epoch": epoch,
                "validation_metrics": validation_metrics, "config": config,
            }, output_dir / "best_model.pt")
        if loss < lowest_validation_loss - 1e-4:
            lowest_validation_loss, stale_epochs = loss, 0
        else:
            stale_epochs += 1
        print(
            f"{architecture_id} seed {seed} epoch {epoch:02d}: "
            f"train={train_metrics['accuracy']:.3f}, val={accuracy:.3f}, loss={loss:.4f}",
            flush=True,
        )
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
        raise AssertionError("Validation order differs from frozen split")
    measured = _metrics(observed_labels, predictions)
    if abs(measured["validation_accuracy"] - checkpoint["validation_metrics"]["accuracy"]) > 1e-9:
        raise AssertionError("Reloaded checkpoint accuracy differs from training record")
    _write_predictions(output_dir / "predictions.csv", split["internal_validation"], observed_labels, predictions, np.asarray(margins))
    best_epoch = checkpoint["epoch"]
    metrics = {
        "experiment_id": f"{BATCH_ID}-{architecture_id}-seed{seed}",
        "batch_id": BATCH_ID, "architecture_id": architecture_id,
        "representation_id": REPRESENTATION_ID, "seed": seed,
        "parameter_count": parameter_count, "best_epoch": best_epoch,
        "epochs_run": len(history), "train_accuracy_at_best_epoch": history[best_epoch - 1]["train_accuracy"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "validation_loss": checkpoint["validation_metrics"]["loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        **measured, "fit_seconds": fit_seconds,
        "train_samples": 1800, "validation_samples": 200,
        "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": config["feature_cache_manifest_sha256"],
        "config_sha256": config["config_sha256"],
        "code_commit": config["code_commit"], "status": "complete",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "train_summary.json", {
        "architecture_id": architecture_id, "seed": seed,
        "epochs_run": len(history), "best_epoch": best_epoch,
        "parameter_count": parameter_count, "fit_seconds": fit_seconds,
        "best_validation_metrics": checkpoint["validation_metrics"],
        "device": str(device),
    })


def smoke() -> None:
    validate_cache()
    channels = frozen_representation()
    prepared = prepare_seed(42, channels)
    dataset = make_dataset(prepared["train_indices"][:8], channels, prepared["mean"], prepared["std"])
    images, targets = next(iter(DataLoader(dataset, batch_size=8)))
    if images.shape != (8, 190, 7, 7):
        raise AssertionError(f"Unexpected frozen feature batch: {images.shape}")
    for architecture_id in ARCHITECTURE_IDS:
        model = ArchitectureDNN(architecture_id)
        logits = model(images)
        if logits.shape != (8, 2):
            raise AssertionError(f"Unexpected {architecture_id} output: {logits.shape}")
        nn.CrossEntropyLoss()(logits, targets).backward()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"{architecture_id}: forward/backward OK, {parameter_count:,} parameters", flush=True)


def run_all() -> None:
    validate_cache()
    channels = frozen_representation()
    prepared = {seed: prepare_seed(seed, channels) for seed in SEEDS}
    for architecture_id in ARCHITECTURE_IDS:
        for seed in SEEDS:
            run_one(architecture_id, seed, prepared[seed], channels)
    print("All nine new architecture runs are complete", flush=True)


def source_run(architecture_id: str, seed: int) -> tuple[dict, list[dict], Path]:
    if architecture_id == "DNN-ARCH-BASE":
        metrics_path, history_path, _ = baseline_paths(seed)
    else:
        directory = REPORT_ROOT / "experiments" / architecture_id / f"seed{seed}"
        metrics_path, history_path = directory / "metrics.json", directory / "history.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with history_path.open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if metrics["seed"] != seed or metrics.get("status") != "complete":
        raise AssertionError(f"Incomplete or mismatched source run: {metrics_path}")
    return metrics, history, metrics_path


def result_rows() -> list[dict]:
    rows = []
    for architecture_id in ("DNN-ARCH-BASE", *ARCHITECTURE_IDS):
        for seed in SEEDS:
            metrics, history, path = source_run(architecture_id, seed)
            best_epoch = metrics["best_epoch"]
            best_row = history[best_epoch - 1]
            rows.append({
                "architecture_id": architecture_id, "seed": seed,
                "source_run": str(path.relative_to(ROOT)).replace("\\", "/"),
                "parameter_count": metrics["parameter_count"], "best_epoch": best_epoch,
                "train_accuracy_at_best_epoch": float(best_row["train_accuracy"]),
                "final_train_accuracy": float(history[-1]["train_accuracy"]),
                "validation_loss": metrics["validation_loss"],
                "final_validation_loss": float(history[-1]["validation_loss"]),
                "validation_accuracy": metrics["validation_accuracy"],
                "cat_accuracy": metrics["cat_accuracy"], "dog_accuracy": metrics["dog_accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"],
                "fit_seconds": metrics["fit_seconds"], "split_sha256": metrics["split_sha256"],
                "feature_cache_manifest_sha256": metrics["feature_cache_manifest_sha256"],
                "code_commit": metrics["code_commit"], "status": "complete",
            })
    return rows


def aggregate(rows: list[dict]) -> dict:
    grouped = {}
    for architecture_id in ("DNN-ARCH-BASE", *ARCHITECTURE_IDS):
        subset = [row for row in rows if row["architecture_id"] == architecture_id]
        values = [row["validation_accuracy"] for row in subset]
        grouped[architecture_id] = {
            "architecture_id": architecture_id,
            "parameter_count": subset[0]["parameter_count"],
            "mean_validation_accuracy": statistics.mean(values),
            "sample_std_validation_accuracy": statistics.stdev(values),
            "worst_seed_accuracy": min(values),
            "mean_cat_accuracy": statistics.mean(row["cat_accuracy"] for row in subset),
            "mean_dog_accuracy": statistics.mean(row["dog_accuracy"] for row in subset),
            "mean_macro_f1": statistics.mean(row["macro_f1"] for row in subset),
            "median_best_epoch": int(statistics.median(row["best_epoch"] for row in subset)),
            "seed_accuracies": {str(row["seed"]): row["validation_accuracy"] for row in subset},
        }
    return grouped


def select_architecture(grouped: dict) -> dict:
    ranked = sorted(grouped.values(), key=lambda item: (-item["mean_validation_accuracy"], item["parameter_count"], item["architecture_id"]))
    top, second = ranked[:2]
    gap = top["mean_validation_accuracy"] - second["mean_validation_accuracy"]
    if gap >= 0.0075 - 1e-12:
        selected = top
        reason = "Top mean validation accuracy leads the runner-up by at least 0.75 percentage point."
    else:
        smaller, larger = sorted((top, second), key=lambda item: item["parameter_count"])
        smaller_balance = min(smaller["mean_cat_accuracy"], smaller["mean_dog_accuracy"])
        larger_balance = min(larger["mean_cat_accuracy"], larger["mean_dog_accuracy"])
        if larger["worst_seed_accuracy"] - smaller["worst_seed_accuracy"] >= 0.02 - 1e-12:
            selected = larger
            reason = "Mean accuracy gap is below 0.75 point; larger model has at least 2.0 points better worst-seed accuracy."
        elif larger_balance - smaller_balance >= 0.02 - 1e-12:
            selected = larger
            reason = "Mean accuracy gap is below 0.75 point; larger model has at least 2.0 points better minimum class accuracy."
        else:
            selected = smaller
            reason = "Mean accuracy gap is below 0.75 point; smaller model wins the parameter-count tie break."
    return {
        "selected_architecture": selected["architecture_id"],
        "final_epochs_for_future_full_train": selected["median_best_epoch"],
        "top_two_mean_accuracy_gap": gap,
        "selection_reason": reason,
        "ranked_by_mean_accuracy": [item["architecture_id"] for item in ranked],
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def summarize() -> None:
    rows = result_rows()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    with (REPORT_ROOT / "architecture_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    grouped = aggregate(rows)
    selection = select_architecture(grouped)
    write_json(REPORT_ROOT / "architecture_decision.json", {"batch_id": BATCH_ID, "models": grouped, **selection})
    lines = [
        "# DNN 架构选择：DNN-ARCH-001", "",
        "输入固定为 Stage I 选定的 `REP-006-FUSION`：HOG+LBP+HSV+RootSIFT 拼接后的 `[190,7,7]`，展平为 9310 维。预处理、缓存、三份划分和训练集归一化均未改变。", "",
        "DNN-ARCH-BASE 直接引用 `report/stage1/experiments/S1B-001/seed*/REP-006-FUSION/MLP/`，没有重新训练。A、B、C 每种在种子 42、123、2026 上各训练一次。没有使用 `data/val`。", "",
        "训练固定为 CrossEntropyLoss、AdamW、学习率 `1e-3`、权重衰减 `1e-4`、批量 32、最多 30 轮、验证损失耐心 5 轮；检查点按最高内部验证准确率选择，同分时选较低验证损失。", "",
        "## 四种架构比较", "",
        "标准差为三个种子的样本标准差。", "",
        "| 架构 | 参数量 | 验证准确率（均值 ± 标准差） | 最差种子 | 猫 / 狗均值 | Macro F1 | 最佳轮次中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture_id in ("DNN-ARCH-BASE", *ARCHITECTURE_IDS):
        item = grouped[architecture_id]
        lines.append(
            f"| {architecture_id} | {item['parameter_count']:,} | "
            f"{pct(item['mean_validation_accuracy'])} ± {pct(item['sample_std_validation_accuracy'])} | "
            f"{pct(item['worst_seed_accuracy'])} | {pct(item['mean_cat_accuracy'])} / {pct(item['mean_dog_accuracy'])} | "
            f"{pct(item['mean_macro_f1'])} | {item['median_best_epoch']} |"
        )
    lines += ["", "## 三种子逐次结果与拟合行为", "",
        "训练准确率来自每轮训练过程统计（含 Dropout），与 Stage I 基线记录口径一致；它不是载入检查点后在训练集上的重评估。验证损失差为最终已观察轮次减去最佳准确率轮次。", "",
        "| 架构 | 种子 | 最佳轮 | 该轮训练准确率 | 最后训练准确率 | 最佳轮验证准确率 | 最佳轮→最后验证损失 | 拟合迹象 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        history = source_run(row["architecture_id"], row["seed"])[1]
        after = len(history) - row["best_epoch"]
        loss_delta = row["final_validation_loss"] - row["validation_loss"]
        train_delta = row["final_train_accuracy"] - row["train_accuracy_at_best_epoch"]
        strong = after >= 2 and loss_delta >= 0.10 and train_delta >= 0.05
        signal = "明显过拟合迹象" if strong else "未达到明显过拟合判据" if after else "最佳轮后无记录"
        lines.append(
            f"| {row['architecture_id']} | {row['seed']} | {row['best_epoch']} | "
            f"{pct(row['train_accuracy_at_best_epoch'])} | {pct(row['final_train_accuracy'])} | "
            f"{pct(row['validation_accuracy'])} | {row['validation_loss']:.4f}→{row['final_validation_loss']:.4f} "
            f"({loss_delta:+.4f}) | {signal} |"
        )
    lines += ["", "上述迹象判据仅用于描述曲线：最佳轮后至少 2 轮、验证损失升高至少 0.10、训练准确率升高至少 5 个百分点。分类仅基于已观察到的轮次，不代表后续未训练轮次的行为。", "",
        "## 固定选择", "",
        f"按预先规定的规则选定 **{selection['selected_architecture']}**。{selection['selection_reason']} 两名最高均值架构相差 {selection['top_two_mean_accuracy_gap'] * 100:.2f} 个百分点。", "",
        "## 后续最终 DNN 评估协议（本次未执行）", "",
        f"冻结所选架构与 `REP-006-FUSION` 以及上述训练配置；将三个最佳轮次的中位数 **{selection['final_epochs_for_future_full_train']}** 作为完整训练轮数。随后仅用全部 2000 张 `data/train` 图重新计算 190 通道归一化统计，以固定种子训练恰好该轮数，保存最终检查点。配置和检查点冻结后，才对 `data/val` 的 500 张图评估一次，并准确注明该集合曾用于历史 DNN-001 评估。", "",
        "## 局限", "",
        "三个 90/10 划分的验证样本彼此有重叠；比较反映这三个内部划分上的表现。Stage I 特征提取已固定，架构选择未评估 CNN、RNN 或保留测试图。", "",
    ]
    (REPORT_ROOT / "architecture_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected {selection['selected_architecture']} with future final_epochs={selection['final_epochs_for_future_full_train']}", flush=True)


def audit() -> None:
    validate_cache()
    channels = frozen_representation()
    if len(channels) != 190:
        raise AssertionError("Frozen channel count changed")
    prepared = {seed: prepare_seed(seed, channels) for seed in SEEDS}
    for architecture_id in ARCHITECTURE_IDS:
        for seed in SEEDS:
            directory = REPORT_ROOT / "experiments" / architecture_id / f"seed{seed}"
            metrics, history, _ = source_run(architecture_id, seed)
            split = prepared[seed]["split"]
            if metrics["split_sha256"] != split["sha256"]:
                raise AssertionError(f"Split mismatch in {directory}")
            if metrics["feature_cache_manifest_sha256"] != sha256_file(HANDCRAFTED / "manifest.json"):
                raise AssertionError(f"Cache mismatch in {directory}")
            with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                predictions = list(csv.DictReader(stream))
            if [row["filename"] for row in predictions] != split["internal_validation"]:
                raise AssertionError(f"Prediction order mismatch in {directory}")
            matrix = [[0, 0], [0, 0]]
            for row in predictions:
                true_label, prediction = int(row["true_label"]), int(row["predicted_label"])
                if true_label != (0 if row["filename"].startswith("cat.") else 1):
                    raise AssertionError(f"Incorrect source label in {directory}")
                if prediction not in (0, 1) or int(row["correct"]) != int(true_label == prediction):
                    raise AssertionError(f"Invalid prediction in {directory}")
                if not np.isfinite(float(row["score_or_margin"])):
                    raise AssertionError(f"Non-finite score in {directory}")
                matrix[true_label][prediction] += 1
            if len(predictions) != 200 or matrix != metrics["confusion_matrix"]:
                raise AssertionError(f"Prediction metrics mismatch in {directory}")
            if abs((matrix[0][0] + matrix[1][1]) / 200 - metrics["validation_accuracy"]) > 1e-9:
                raise AssertionError(f"Prediction accuracy mismatch in {directory}")
            if abs(float(history[metrics["best_epoch"] - 1]["validation_accuracy"]) - metrics["validation_accuracy"]) > 1e-9:
                raise AssertionError(f"History accuracy mismatch in {directory}")
    archived = list((REPORT_ROOT / "experiments").glob("**/metrics.json"))
    if len(archived) != 9:
        raise AssertionError(f"Expected nine new runs; found {len(archived)}")
    print("Audited nine new runs and 1,800 internal-validation predictions; three Stage-I baselines reused", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run-all", "summarize", "audit"))
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "run-all":
        run_all()
    elif args.command == "summarize":
        summarize()
    elif args.command == "audit":
        audit()


if __name__ == "__main__":
    main()
