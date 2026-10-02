"""Compare three frozen-input standard RNN architectures on internal splits."""

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
from models.rnn_architecture import (
    ARCHITECTURE_IDS, SPECIFICATIONS, ArchitectureRNN,
    feature_map_to_cell_sequence, feature_map_to_row_sequence,
)
from representations.cache import HANDCRAFTED, ROOT, current_commit, sha256_file, validate_cache
from training import seed_everything, select_device, write_history, write_json

BATCH_ID = "RNN-ARCH-001"
SEEDS = (42, 123, 2026)
REPRESENTATION_ID = "REP-006-FUSION"
OUTPUT = ROOT / "outputs/rnn_architecture" / BATCH_ID
REPORT = ROOT / "report/rnn_architecture"
ARTIFACTS = ("config.json", "normalization.json", "history.csv", "metrics.json",
             "train_summary.json", "predictions.csv")
RESULT_FIELDS = (
    "experiment_id", "architecture_id", "sequence_representation", "seed",
    "parameter_count", "sequence_length", "rnn_input_size", "hidden_size", "num_layers",
    "bidirectional", "best_epoch", "train_accuracy_at_best_epoch", "final_train_accuracy",
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
    row = feature_map_to_row_sequence(features)
    cell = feature_map_to_cell_sequence(features)
    if row.shape != (8, 7, 1330) or cell.shape != (8, 49, 190):
        raise AssertionError("RNN sequence conversion shape mismatch")
    # Verify exact top-to-bottom row and raster cell ordering independently of learned weights.
    marker = torch.arange(190 * 7 * 7).reshape(1, 190, 7, 7)
    if (marker[0, 3, 2, 4] != feature_map_to_row_sequence(marker)[0, 2, 3 * 7 + 4]
            or marker[0, 3, 2, 4] != feature_map_to_cell_sequence(marker)[0, 2 * 7 + 4, 3]):
        raise AssertionError("RNN sequence order mismatch")
    for architecture_id in ARCHITECTURE_IDS:
        model = ArchitectureRNN(architecture_id)
        if (model.rnn.batch_first is not True or model.rnn.dropout != 0.0
                or sum(isinstance(layer, nn.RNN) for layer in model.modules()) != 1
                or model(features).shape != (8, 2)):
            raise AssertionError(f"RNN model shape or type mismatch: {architecture_id}")
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
    model = ArchitectureRNN("RNN-ARCH-A").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(1, 201):
        epoch_pass(model, train_loader, criterion, device, optimizer)
        measured = epoch_pass(model, inspect_loader, criterion, device)
        if measured["accuracy"] >= 0.95:
            write_json(REPORT / "tiny_overfit_diagnostic.json", {
                "architecture_id": "RNN-ARCH-A", "seed": 42, "samples": 64,
                "epoch": epoch, "training_accuracy_on_subset": measured["accuracy"],
                "formal_result": False, "status": "passed",
            })
            print(f"Tiny overfit diagnostic passed at epoch {epoch}: {measured['accuracy']:.2%}", flush=True)
            return
    raise AssertionError(f"RNN-ARCH-A only fit {measured['accuracy']:.2%} of 64 training images")


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
    spec = SPECIFICATIONS[architecture_id]
    split = prepared["split"]
    train_dataset = make_dataset(prepared["train_indices"], channels, prepared["mean"], prepared["std"])
    validation_dataset = make_dataset(prepared["validation_indices"], channels, prepared["mean"], prepared["std"])
    loader_kwargs = {"batch_size": 32, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = ArchitectureRNN(architecture_id).to(device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config = {
        "batch_id": BATCH_ID, "architecture_id": architecture_id,
        "representation_id": REPRESENTATION_ID, "representation_shape": [190, 7, 7],
        **spec, "seed": seed, "train_samples": 1800, "validation_samples": 200,
        "class_to_idx": {"cat": 0, "dog": 1}, "parameter_count": count,
        "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": sha256_file(HANDCRAFTED / "manifest.json"),
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "batch_size": 32, "max_epochs": 30, "early_stopping_patience": 6,
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
    for epoch in range(1, 31):
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
        if stale >= 6:
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
        "sequence_representation": spec["sequence_representation"], "seed": seed,
        "parameter_count": count, "sequence_length": spec["sequence_length"],
        "rnn_input_size": spec["rnn_input_size"], "hidden_size": spec["hidden_size"],
        "num_layers": 1, "bidirectional": spec["bidirectional"],
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
    figure.suptitle("RNN-ARCH-001: train and internal-validation trajectories")
    figure.tight_layout()
    figure.savefig(REPORT / "training_curves.png", dpi=150)
    plt.close(figure)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def summarize() -> None:
    rows, grouped = aggregate()
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "architecture_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    selection = select(grouped)
    dnn = json.loads((ROOT / "report/dnn_architecture/architecture_decision.json").read_text(encoding="utf-8"))["models"]["DNN-ARCH-C"]
    cnn = json.loads((ROOT / "report/cnn_architecture/architecture_decision.json").read_text(encoding="utf-8"))["models"]["CNN-ARCH-C"]
    t1 = json.loads((ROOT / "report/cnn_training/training_strategy_decision.json").read_text(encoding="utf-8"))["models"]["CNN-TRAIN-T1-FLIP"]
    write_json(REPORT / "architecture_decision.json", {
        "batch_id": BATCH_ID, "models": grouped, **selection,
        "descriptive_internal_references": {
            "DNN-ARCH-C": dnn["mean_validation_accuracy"],
            "CNN-ARCH-C": cnn["mean_validation_accuracy"],
            "CNN-TRAIN-T1-FLIP": t1["mean_validation_accuracy"],
        },
    })
    plot_curves()
    lines = [
        "# RNN-ARCH-001：标准 RNN 架构选择", "",
        "仅使用 `data/train` 的种子 42、123、2026 三份既有 1800/200 划分；输入是冻结的 `REP-006-FUSION` `[190,7,7]`，逐划分归一化只拟合 1800 张训练图，无增强。没有访问 `data/val`。", "",
        "A/B 使用从上到下 7 行序列 `[B,7,1330]`；C 使用 7×7 栅格顺序的 49 单元序列 `[B,49,190]`。A 是单向 tanh RNN(128)，B 是双向 tanh RNN(96)，C 是双向 tanh RNN(128)。均为单层 `nn.RNN(batch_first=True)`，取 `h_n` 的末端方向状态，直接用线性层输出两类 logits。", "",
        "固定 CrossEntropyLoss、AdamW、学习率 3e-4、权重衰减 1e-4、批量 32、最多 30 轮、验证损失早停耐心 6。所有运行在反向传播后、优化器更新前使用梯度裁剪范数 1.0；无 Dropout 或学习率调度器。检查点按最高验证准确率、同分较低验证损失选取。", "",
        "## 三架构汇总", "", "标准差为三种子的样本标准差。", "",
        "| 架构 | 参数量 | 验证准确率均值 ± 标准差 | 最差种子 | 猫/狗均值 | 类别差 | Macro F1 | 最佳轮中位数 | 平均训练秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture_id in ARCHITECTURE_IDS:
        item = grouped[architecture_id]
        lines.append(f"| {architecture_id} | {item['parameter_count']:,} | {pct(item['mean_validation_accuracy'])} ± {pct(item['sample_std_validation_accuracy'])} | {pct(item['worst_seed_accuracy'])} | {pct(item['mean_cat_accuracy'])}/{pct(item['mean_dog_accuracy'])} | {item['absolute_cat_dog_gap'] * 100:.2f} pp | {pct(item['mean_macro_f1'])} | {item['median_best_epoch']} | {item['mean_fit_seconds']:.2f} |")
    lines += ["", "## 逐种子曲线诊断", "",
              "| 架构 | 种子 | 最佳轮/末轮 | 最佳轮验证准确率 | 末轮训练准确率 | 最佳轮→末轮验证损失 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        _, history, _ = record(row["architecture_id"], row["seed"])
        final = history[-1]
        lines.append(f"| {row['architecture_id']} | {row['seed']} | {row['best_epoch']}/{len(history)} | {pct(row['validation_accuracy'])} | {pct(row['final_train_accuracy'])} | {row['validation_loss']:.4f}→{float(final['validation_loss']):.4f} |")
    lines += ["", "| 架构 | 末轮训练准确率均值 | 末轮验证准确率均值 | 末轮训练/验证差距 | 最低验证损失后升高 ≥ 0.10 的运行数 |",
              "|---|---:|---:|---:|---:|"]
    for architecture_id in ARCHITECTURE_IDS:
        histories = [record(architecture_id, seed)[1] for seed in SEEDS]
        mean_train = statistics.mean(float(history[-1]["train_accuracy"]) for history in histories)
        mean_validation = statistics.mean(float(history[-1]["validation_accuracy"]) for history in histories)
        loss_rises = sum(float(history[-1]["validation_loss"]) -
                         min(float(epoch["validation_loss"]) for epoch in history) >= 0.10
                         for history in histories)
        lines.append(f"| {architecture_id} | {pct(mean_train)} | {pct(mean_validation)} | {(mean_train - mean_validation) * 100:.2f} pp | {loss_rises}/3 |")
    lines += ["", "A/B 的末轮训练准确率明显高于内部验证；C 的差距较小，但三个种子的训练准确率仍持续升高。验证损失在部分运行中回升，说明值得另立训练策略阶段检验正则化或增强；本阶段不执行该搜索。", "",
              "![三架构训练和内部验证曲线](training_curves.png)", "",
              "## 预设规则与参照", "",
              f"按预设规则选定 **{selection['selected_architecture']}**；规则代码 `{selection['selection_rule']}`，前两名平均准确率相差 {selection['top_two_mean_accuracy_gap'] * 100:.2f} 个百分点。", "",
              f"A→B（同为行序列）均值变化 {(grouped['RNN-ARCH-B']['mean_validation_accuracy'] - grouped['RNN-ARCH-A']['mean_validation_accuracy']) * 100:+.2f} pp、最差种子变化 {(grouped['RNN-ARCH-B']['worst_seed_accuracy'] - grouped['RNN-ARCH-A']['worst_seed_accuracy']) * 100:+.2f} pp、类别差变化 {(grouped['RNN-ARCH-B']['absolute_cat_dog_gap'] - grouped['RNN-ARCH-A']['absolute_cat_dog_gap']) * 100:+.2f} pp。B→C（双向 RNN，7 行变 49 单元）均值变化 {(grouped['RNN-ARCH-C']['mean_validation_accuracy'] - grouped['RNN-ARCH-B']['mean_validation_accuracy']) * 100:+.2f} pp、最差种子变化 {(grouped['RNN-ARCH-C']['worst_seed_accuracy'] - grouped['RNN-ARCH-B']['worst_seed_accuracy']) * 100:+.2f} pp、类别差变化 {(grouped['RNN-ARCH-C']['absolute_cat_dog_gap'] - grouped['RNN-ARCH-B']['absolute_cat_dog_gap']) * 100:+.2f} pp。", "",
              f"同一内部开发数据上的描述性参照：DNN-ARCH-C {pct(dnn['mean_validation_accuracy'])}，CNN-ARCH-C {pct(cnn['mean_validation_accuracy'])}，CNN-TRAIN-T1-FLIP {pct(t1['mean_validation_accuracy'])}。训练设置不同，且这些参照不参与 RNN 选择；不使用已公布的保留集结果。", "",
              "本阶段仅选择 RNN 架构。是否另立训练策略实验由实际曲线诊断决定；未进行全量 RNN 训练或保留集评估。", "",
    ]
    (REPORT / "architecture_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected {selection['selected_architecture']}", flush=True)


def audit() -> None:
    channels, prepared = context()
    paths = list((REPORT / "experiments").glob("**/metrics.json"))
    if len(paths) != 9 or len(channels) != 190:
        raise AssertionError("Expected exactly nine frozen REP-006 RNN runs")
    for architecture_id in ARCHITECTURE_IDS:
        spec = SPECIFICATIONS[architecture_id]
        model = ArchitectureRNN(architecture_id)
        if (not isinstance(model.rnn, nn.RNN) or model.rnn.num_layers != 1
                or model.rnn.dropout != 0.0 or model.rnn.bidirectional != spec["bidirectional"]
                or model.rnn.input_size != spec["rnn_input_size"]
                or model.rnn.hidden_size != spec["hidden_size"]):
            raise AssertionError("RNN architecture changed")
        expected_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for seed in SEEDS:
            metrics, history, directory = record(architecture_id, seed)
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            normal = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
            split = prepared[seed]["split"]
            if (metrics["status"] != "complete" or metrics["split_sha256"] != split["sha256"]
                    or config["split_sha256"] != split["sha256"]
                    or metrics["feature_cache_manifest_sha256"] != sha256_file(HANDCRAFTED / "manifest.json")
                    or config["representation_id"] != REPRESENTATION_ID
                    or config["representation_shape"] != [190, 7, 7]
                    or config["architecture_id"] != architecture_id or config["seed"] != seed
                    or any(config[key] != value for key, value in spec.items())
                    or config["parameter_count"] != expected_count or metrics["parameter_count"] != expected_count
                    or config["train_samples"] != 1800 or config["validation_samples"] != 200
                    or config["gradient_clip_norm"] != 1.0 or metrics["gradient_clip_norm"] != 1.0
                    or config["rnn_dropout"] != 0.0 or config["scheduler"] is not None
                    or config["augmentation"] != "none" or config["optimizer"] != "AdamW"
                    or config["learning_rate"] != 3e-4 or config["weight_decay"] != 1e-4
                    or config["batch_size"] != 32 or config["max_epochs"] != 30
                    or config["early_stopping_patience"] != 6 or config["loss"] != "CrossEntropyLoss"
                    or normal["split_sha256"] != split["sha256"]
                    or not np.array_equal(np.asarray(normal["mean"], dtype=np.float32), prepared[seed]["mean"])
                    or not np.array_equal(np.asarray(normal["std"], dtype=np.float32), prepared[seed]["std"])):
                raise AssertionError(f"RNN protocol or normalization mismatch: {directory}")
            with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                predictions = list(csv.DictReader(stream))
            if (len(predictions) != 200
                    or [row["filename"] for row in predictions] != split["internal_validation"]
                    or [int(row["true_label"]) for row in predictions].count(0) != 100
                    or [int(row["true_label"]) for row in predictions].count(1) != 100):
                raise AssertionError(f"RNN prediction rows invalid: {directory}")
            matrix = [[0, 0], [0, 0]]
            row_losses = []
            for row in predictions:
                actual, guess = int(row["true_label"]), int(row["predicted_label"])
                logits = np.asarray([float(row["logit_cat"]), float(row["logit_dog"])])
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                if (actual != (0 if row["filename"].startswith("cat.") else 1)
                        or guess not in (0, 1) or int(row["correct"]) != int(actual == guess)
                        or abs(float(row["prob_cat"]) - probs[0]) > 1e-6
                        or abs(float(row["prob_dog"]) - probs[1]) > 1e-6):
                    raise AssertionError(f"RNN prediction probability mismatch: {directory}")
                row_losses.append(-np.log(probs[actual]))
                matrix[actual][guess] += 1
            f1_cat = 2 * matrix[0][0] / (2 * matrix[0][0] + matrix[0][1] + matrix[1][0])
            f1_dog = 2 * matrix[1][1] / (2 * matrix[1][1] + matrix[0][1] + matrix[1][0])
            if (matrix != metrics["confusion_matrix"] or [sum(row) for row in matrix] != [100, 100]
                    or abs((matrix[0][0] + matrix[1][1]) / 200 - metrics["validation_accuracy"]) > 1e-12
                    or abs(float(history[metrics["best_epoch"] - 1]["validation_accuracy"]) - metrics["validation_accuracy"]) > 1e-12
                    or abs(float(np.mean(row_losses)) - metrics["validation_loss"]) > 1e-6
                    or abs((f1_cat + f1_dog) / 2 - metrics["macro_f1"]) > 1e-12):
                raise AssertionError(f"RNN aggregate metrics mismatch: {directory}")
    rows, grouped = aggregate()
    decision = json.loads((REPORT / "architecture_decision.json").read_text(encoding="utf-8"))
    if len(rows) != 9 or decision["models"] != grouped or decision["selected_architecture"] != select(grouped)["selected_architecture"]:
        raise AssertionError("RNN aggregate or selection mismatch")
    print("Audited nine RNN runs, fixed splits/normalization/protocol and 1,800 predictions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "tiny-overfit", "run-all", "summarize", "audit"))
    command = parser.parse_args().command
    {"smoke": smoke, "tiny-overfit": tiny_overfit, "run-all": run_all,
     "summarize": summarize, "audit": audit}[command]()


if __name__ == "__main__":
    main()
