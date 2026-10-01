"""Compare three frozen-input CNN architectures on Stage-I internal splits."""

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

from dnn_architecture_experiments import frozen_representation, make_dataset, prepare_seed
from models.cnn_architecture import ARCHITECTURE_IDS, ArchitectureCNN
from representations.cache import HANDCRAFTED, ROOT, current_commit, sha256_file, validate_cache
from representations.probes import _metrics
from training import run_epoch, seed_everything, select_device, write_history, write_json

BATCH_ID = "CNN-ARCH-001"
SEEDS = (42, 123, 2026)
REPRESENTATION_ID = "REP-006-FUSION"
OUTPUT_ROOT = ROOT / "outputs/cnn_architecture" / BATCH_ID
REPORT_ROOT = ROOT / "report/cnn_architecture"
ARCHIVE_FILES = ("config.json", "normalization.json", "metrics.json", "history.csv", "train_summary.json", "predictions.csv")
RESULT_FIELDS = (
    "architecture_id", "seed", "parameter_count", "best_epoch",
    "train_loss_at_best_epoch", "train_accuracy_at_best_epoch",
    "final_train_loss", "final_train_accuracy", "validation_loss",
    "validation_accuracy", "cat_accuracy", "dog_accuracy", "balanced_accuracy",
    "macro_f1", "fit_seconds", "split_sha256", "feature_cache_manifest_sha256",
    "code_commit", "status", "source_run",
)


def context() -> tuple[tuple[int, ...], dict[int, dict]]:
    validate_cache()
    channels = frozen_representation()
    prepared = {seed: prepare_seed(seed, channels) for seed in SEEDS}
    return channels, prepared


def run_dir(architecture_id: str, seed: int) -> Path:
    return OUTPUT_ROOT / architecture_id / f"seed{seed}"


def archive_run(architecture_id: str, seed: int) -> None:
    source = run_dir(architecture_id, seed)
    target = REPORT_ROOT / "experiments" / architecture_id / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVE_FILES:
        if not (source / name).exists():
            raise FileNotFoundError(f"Incomplete run: {source / name}")
        shutil.copy2(source / name, target / name)


def smoke() -> None:
    channels, prepared = context()
    seed = 42
    dataset = make_dataset(prepared[seed]["train_indices"][:8], channels,
                           prepared[seed]["mean"], prepared[seed]["std"])
    features, labels = next(iter(DataLoader(dataset, batch_size=8)))
    if features.shape != (8, 190, 7, 7) or not torch.isfinite(features).all():
        raise AssertionError("Frozen CNN input has wrong shape or nonfinite values")
    expected = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")[prepared[seed]["train_indices"][:8]]
    if not np.array_equal(labels.numpy(), expected):
        raise AssertionError("CNN smoke labels differ from frozen cache")
    for architecture_id in ARCHITECTURE_IDS:
        model = ArchitectureCNN(architecture_id)
        logits = model(features)
        if logits.shape != (8, 2):
            raise AssertionError(f"Wrong output shape for {architecture_id}: {logits.shape}")
        nn.CrossEntropyLoss()(logits, labels).backward()
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"{architecture_id}: [8,190,7,7] -> [8,2], loss/backward OK, {count:,} parameters", flush=True)


def tiny_overfit() -> None:
    channels, prepared = context()
    indices = prepared[42]["train_indices"]
    labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    subset = [index for index in indices if labels[index] == 0][:32]
    subset += [index for index in indices if labels[index] == 1][:32]
    if len(subset) != 64:
        raise AssertionError("Tiny diagnostic requires 32 training samples per class")
    seed_everything(42)
    device = select_device("auto")
    dataset = make_dataset(subset, channels, prepared[42]["mean"], prepared[42]["std"])
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(42), num_workers=0)
    inspection_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model = ArchitectureCNN("CNN-ARCH-A").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(1, 81):
        run_epoch(model, loader, criterion, device, optimizer)
        measured = run_epoch(model, inspection_loader, criterion, device)
        if measured["accuracy"] >= 0.95:
            print(f"Diagnostic CNN-ARCH-A memorized 64 training examples: {measured['accuracy']:.2%} at epoch {epoch}", flush=True)
            return
    raise AssertionError(f"CNN-ARCH-A failed tiny overfit diagnostic: {measured['accuracy']:.2%} after 80 epochs")


def write_predictions(path: Path, filenames: list[str], labels: np.ndarray,
                      logits: np.ndarray) -> dict:
    probabilities = torch.from_numpy(logits).softmax(dim=1).numpy()
    predictions = logits.argmax(axis=1)
    if not (len(filenames) == len(labels) == len(logits) == 200):
        raise AssertionError("Internal-validation predictions must have 200 aligned rows")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "true_label", "predicted_label", "logit_cat", "logit_dog", "prob_cat", "prob_dog", "correct"))
        for index, name in enumerate(filenames):
            writer.writerow((name, int(labels[index]), int(predictions[index]),
                             float(logits[index, 0]), float(logits[index, 1]),
                             float(probabilities[index, 0]), float(probabilities[index, 1]),
                             int(labels[index] == predictions[index])))
    return _metrics(labels, predictions)


def train_one(architecture_id: str, seed: int, prepared: dict, channels: tuple[int, ...], output: Path) -> None:
    seed_everything(seed)
    device = select_device("auto")
    train_dataset = make_dataset(prepared["train_indices"], channels, prepared["mean"], prepared["std"])
    validation_dataset = make_dataset(prepared["validation_indices"], channels, prepared["mean"], prepared["std"])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0,
                              pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False,
                                   num_workers=0, pin_memory=device.type == "cuda")
    model = ArchitectureCNN(architecture_id).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    split = prepared["split"]
    config = {
        "batch_id": BATCH_ID, "architecture_id": architecture_id,
        "representation_id": REPRESENTATION_ID, "representation_shape": [190, 7, 7],
        "seed": seed, "train_samples": 1800, "validation_samples": 200,
        "class_to_idx": {"cat": 0, "dog": 1}, "parameter_count": parameter_count,
        "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": sha256_file(HANDCRAFTED / "manifest.json"),
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "batch_size": 32, "loss": "CrossEntropyLoss", "max_epochs": 30,
        "early_stopping_patience": 6, "augmentation": "none",
        "checkpoint_selection": "highest internal-validation accuracy; tie lower validation loss",
        "device": str(device), "code_commit": current_commit(),
    }
    write_json(output / "config.json", config)
    write_json(output / "normalization.json", {
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
        write_history(output / "history.csv", history)
        accuracy, loss = validation_metrics["accuracy"], validation_metrics["loss"]
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_accuracy, best_loss = accuracy, loss
            torch.save({"model_state": model.state_dict(), "architecture_id": architecture_id,
                        "epoch": epoch, "validation_metrics": validation_metrics,
                        "config": config}, output / "best_model.pt")
        if loss < lowest_loss - 1e-4:
            lowest_loss, stale = loss, 0
        else:
            stale += 1
        print(f"{architecture_id} seed {seed} epoch {epoch:02d}: train={train_metrics['accuracy']:.3f}, val={accuracy:.3f}, loss={loss:.4f}", flush=True)
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
    logits = torch.cat(logits_all).numpy()
    cache_labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    if not np.array_equal(labels, cache_labels[prepared["validation_indices"]]):
        raise AssertionError("Validation labels differ from frozen split")
    measured = write_predictions(output / "predictions.csv", split["internal_validation"], labels, logits)
    if abs(measured["validation_accuracy"] - checkpoint["validation_metrics"]["accuracy"]) > 1e-9:
        raise AssertionError("Reloaded checkpoint accuracy differs from recorded epoch")
    best_epoch = checkpoint["epoch"]
    metrics = {
        "experiment_id": f"{BATCH_ID}-{architecture_id}-seed{seed}",
        "batch_id": BATCH_ID, "architecture_id": architecture_id,
        "representation_id": REPRESENTATION_ID, "seed": seed,
        "parameter_count": parameter_count, "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_loss_at_best_epoch": history[best_epoch - 1]["train_loss"],
        "train_accuracy_at_best_epoch": history[best_epoch - 1]["train_accuracy"],
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "validation_loss": checkpoint["validation_metrics"]["loss"],
        **measured, "fit_seconds": fit_seconds,
        "split_sha256": split["sha256"],
        "feature_cache_manifest_sha256": config["feature_cache_manifest_sha256"],
        "code_commit": config["code_commit"], "status": "complete",
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "train_summary.json", {
        "architecture_id": architecture_id, "seed": seed,
        "epochs_run": len(history), "best_epoch": best_epoch,
        "parameter_count": parameter_count, "fit_seconds": fit_seconds,
        "best_validation_metrics": checkpoint["validation_metrics"],
        "device": str(device),
    })


def run_one(architecture_id: str, seed: int, prepared: dict, channels: tuple[int, ...]) -> None:
    output = run_dir(architecture_id, seed)
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (metrics.get("status") != "complete" or metrics.get("architecture_id") != architecture_id
                or metrics.get("seed") != seed or metrics.get("split_sha256") != prepared["split"]["sha256"]):
            raise RuntimeError(f"Existing run incompatible: {output}")
        archive_run(architecture_id, seed)
        print(f"Reused completed {architecture_id} seed {seed}", flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Partial run exists; inspect before retrying: {output}")
    output.mkdir(parents=True)
    try:
        train_one(architecture_id, seed, prepared, channels, output)
        archive_run(architecture_id, seed)
    except Exception as error:
        write_json(output / "failure.json", {"architecture_id": architecture_id,
                   "seed": seed, "status": "failed", "error": str(error),
                   "traceback": traceback.format_exc()})
        raise


def run_all() -> None:
    channels, prepared = context()
    for architecture_id in ARCHITECTURE_IDS:
        for seed in SEEDS:
            run_one(architecture_id, seed, prepared[seed], channels)
    print("All nine CNN architecture runs completed", flush=True)


def run_record(architecture_id: str, seed: int) -> tuple[dict, list[dict], Path]:
    directory = REPORT_ROOT / "experiments" / architecture_id / f"seed{seed}"
    path = directory / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if metrics["architecture_id"] != architecture_id or metrics["seed"] != seed or metrics["status"] != "complete":
        raise AssertionError(f"Mismatched archived run: {path}")
    return metrics, history, path


def result_rows() -> list[dict]:
    rows = []
    for architecture_id in ARCHITECTURE_IDS:
        for seed in SEEDS:
            metrics, _, path = run_record(architecture_id, seed)
            rows.append({**{field: metrics[field] for field in RESULT_FIELDS if field != "source_run"},
                         "source_run": str(path.relative_to(ROOT)).replace("\\", "/")})
    return rows


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for architecture_id in ARCHITECTURE_IDS:
        subset = [row for row in rows if row["architecture_id"] == architecture_id]
        if len(subset) != 3 or len({row["parameter_count"] for row in subset}) != 1:
            raise AssertionError(f"Incomplete architecture group: {architecture_id}")
        accuracies = [row["validation_accuracy"] for row in subset]
        result[architecture_id] = {
            "architecture_id": architecture_id,
            "parameter_count": subset[0]["parameter_count"],
            "mean_validation_accuracy": statistics.mean(accuracies),
            "sample_std_validation_accuracy": statistics.stdev(accuracies),
            "worst_seed_accuracy": min(accuracies),
            "mean_cat_accuracy": statistics.mean(row["cat_accuracy"] for row in subset),
            "mean_dog_accuracy": statistics.mean(row["dog_accuracy"] for row in subset),
            "mean_balanced_accuracy": statistics.mean(row["balanced_accuracy"] for row in subset),
            "mean_macro_f1": statistics.mean(row["macro_f1"] for row in subset),
            "median_best_epoch": int(statistics.median(row["best_epoch"] for row in subset)),
            "mean_fit_seconds": statistics.mean(row["fit_seconds"] for row in subset),
            "seed_accuracies": {str(row["seed"]): row["validation_accuracy"] for row in subset},
        }
    return result


def select_architecture(grouped: dict) -> dict:
    ranked = sorted(grouped.values(), key=lambda item: (-item["mean_validation_accuracy"], item["architecture_id"]))
    top, second = ranked[:2]
    gap = top["mean_validation_accuracy"] - second["mean_validation_accuracy"]
    if gap >= 0.0075 - 1e-12:
        selected = top
        reason = "Top mean accuracy exceeds runner-up by at least 0.75 percentage point."
    else:
        worst_gap = top["worst_seed_accuracy"] - second["worst_seed_accuracy"]
        if abs(worst_gap) >= 0.01 - 1e-12:
            selected = top if worst_gap > 0 else second
            reason = "Mean gap below 0.75 point; worst-seed accuracy differs by at least 1.0 point."
        else:
            smaller, larger = sorted((top, second), key=lambda item: (item["parameter_count"], item["architecture_id"]))
            smaller_balance = min(smaller["mean_cat_accuracy"], smaller["mean_dog_accuracy"])
            larger_balance = min(larger["mean_cat_accuracy"], larger["mean_dog_accuracy"])
            if larger_balance - smaller_balance >= 0.02 - 1e-12:
                selected = larger
                reason = "Mean and worst-seed gaps are small; larger model improves minimum mean class accuracy by at least 2.0 points."
            else:
                selected = smaller
                reason = "Mean and worst-seed gaps are small; fewer trainable parameters decide."
    return {"selected_architecture": selected["architecture_id"],
            "top_two_mean_accuracy_gap": gap, "selection_reason": reason,
            "ranked_by_mean_accuracy": [item["architecture_id"] for item in ranked]}


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
    dnn = json.loads((ROOT / "report/dnn_architecture/architecture_decision.json").read_text(encoding="utf-8"))["models"]["DNN-ARCH-C"]
    write_json(REPORT_ROOT / "architecture_decision.json", {
        "batch_id": BATCH_ID, "models": grouped, **selection,
        "diagnostic_dnn_reference": {"architecture_id": "DNN-ARCH-C",
            "mean_validation_accuracy": dnn["mean_validation_accuracy"],
            "sample_std_validation_accuracy": dnn["sample_std_validation_accuracy"]},
    })
    lines = [
        "# CNN 架构选择：CNN-ARCH-001", "",
        "仅使用 `data/train` 的种子 42、123、2026 三份既有 1800/200 划分；输入固定为 Stage I 的 `REP-006-FUSION`，`[190,7,7]` 保留空间结构。归一化逐划分仅由 1800 张训练图拟合。没有使用 `data/val`。", "",
        "三个模型均使用 Conv→BatchNorm→ReLU、全局平均池化和两类原始 logits。A 为 1×1 投影与两层 3×3 卷积；B 为 1×1 投影、两个 64 通道残差块、1×1 升维；C 为 1×1 投影、1×1/3×3/膨胀 3×3 三分支拼接、1×1 升维。均无中途下采样或 Dropout。", "",
        "固定训练协议：CrossEntropyLoss、AdamW、学习率 `3e-4`、权重衰减 `1e-4`、批量 32、最多 30 轮、验证损失早停耐心 6；检查点按最高内部验证准确率、同分较低验证损失选择。", "",
        "## 三架构比较", "", "标准差为三个种子的样本标准差。", "",
        "| 架构 | 参数量 | 验证准确率（均值 ± 标准差） | 最差种子 | 猫/狗均值 | 平衡准确率 | Macro F1 | 最佳轮次中位数 | 平均训练秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture_id in ARCHITECTURE_IDS:
        item = grouped[architecture_id]
        lines.append(f"| {architecture_id} | {item['parameter_count']:,} | "
                     f"{pct(item['mean_validation_accuracy'])} ± {pct(item['sample_std_validation_accuracy'])} | "
                     f"{pct(item['worst_seed_accuracy'])} | {pct(item['mean_cat_accuracy'])}/{pct(item['mean_dog_accuracy'])} | "
                     f"{pct(item['mean_balanced_accuracy'])} | {pct(item['mean_macro_f1'])} | "
                     f"{item['median_best_epoch']} | {item['mean_fit_seconds']:.2f} |")
    lines += ["", "三个种子的逐次验证准确率：" + "；".join(
        f"{architecture_id} = " + ", ".join(f"{seed}: {pct(grouped[architecture_id]['seed_accuracies'][str(seed)])}" for seed in SEEDS)
        for architecture_id in ARCHITECTURE_IDS), "",
        "## 逐次曲线与拟合行为", "",
        "训练准确率是含 BatchNorm 的逐批训练过程统计；验证指标来自 `eval()`。最佳轮后的验证损失变化只在实际观察到后续轮次时计算。", "",
        "| 架构 | 种子 | 最佳轮/末轮 | 最佳轮训练/验证准确率 | 末轮训练/验证准确率 | 最佳轮→末轮验证损失 | 曲线迹象 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        _, history, _ = run_record(row["architecture_id"], row["seed"])
        final = history[-1]
        loss_delta = float(final["validation_loss"]) - row["validation_loss"]
        train_gain = row["final_train_accuracy"] - row["train_accuracy_at_best_epoch"]
        after = len(history) - row["best_epoch"]
        if after >= 2 and loss_delta >= 0.10 and train_gain >= 0.05:
            signal = "训练继续提升而验证损失明显上升"
        elif after and loss_delta >= 0.10 and float(final["validation_accuracy"]) >= row["validation_accuracy"] - 0.03:
            signal = "验证准确率仍接近峰值，损失已上升"
        elif after:
            signal = "有后续轮次，未达到上述明显恶化判据"
        else:
            signal = "最佳轮后无记录"
        lines.append(f"| {row['architecture_id']} | {row['seed']} | {row['best_epoch']}/{len(history)} | "
                     f"{pct(row['train_accuracy_at_best_epoch'])}/{pct(row['validation_accuracy'])} | "
                     f"{pct(row['final_train_accuracy'])}/{pct(float(final['validation_accuracy']))} | "
                     f"{row['validation_loss']:.4f}→{float(final['validation_loss']):.4f} ({loss_delta:+.4f}) | {signal} |")
    lines += ["", "上述曲线迹象只描述已记录轮次，不把训练准确率高于验证准确率本身当作过拟合证明。", ""]
    for architecture_id in ARCHITECTURE_IDS:
        item = grouped[architecture_id]
        bias = abs(item["mean_cat_accuracy"] - item["mean_dog_accuracy"])
        lines.append(f"{architecture_id}：猫/狗平均准确率差 {bias * 100:.2f} 个百分点"
                     + ("，存在明显类别偏向。" if bias >= 0.10 else "，未达到 10 个百分点的类别偏向描述阈值。"))
    lines += ["", "## 固定选择与 DNN 参照", "",
        f"按预设规则选定 **{selection['selected_architecture']}**。{selection['selection_reason']} 前两名平均准确率相差 {selection['top_two_mean_accuracy_gap'] * 100:.2f} 个百分点。", "",
        f"已冻结的 DNN-ARCH-C 内部验证参照为 {pct(dnn['mean_validation_accuracy'])} ± {pct(dnn['sample_std_validation_accuracy'])}。它仅作诊断对照，不参与 CNN 选择；两类模型的学习率及训练协议不同。", "",
        "本阶段只选 CNN 架构。没有运行训练策略搜索、最终全量 CNN 训练或 `data/val` 评估。若之后需要策略优化，应另立实验编号及测试前协议。", "",
    ]
    (REPORT_ROOT / "architecture_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected {selection['selected_architecture']}", flush=True)


def audit() -> None:
    channels, prepared = context()
    if len(channels) != 190:
        raise AssertionError("Wrong frozen input channels")
    directories = list((REPORT_ROOT / "experiments").glob("**/metrics.json"))
    if len(directories) != 9:
        raise AssertionError(f"Expected exactly nine formal CNN runs; found {len(directories)}")
    for architecture_id in ARCHITECTURE_IDS:
        for seed in SEEDS:
            metrics, history, path = run_record(architecture_id, seed)
            directory = path.parent
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            normal = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
            split = prepared[seed]["split"]
            if (metrics["split_sha256"] != split["sha256"]
                    or config["split_sha256"] != split["sha256"]
                    or metrics["feature_cache_manifest_sha256"] != sha256_file(HANDCRAFTED / "manifest.json")
                    or config["representation_shape"] != [190, 7, 7]
                    or config["architecture_id"] != architecture_id
                    or config["seed"] != seed
                    or config["optimizer"] != "AdamW" or config["learning_rate"] != 3e-4
                    or config["weight_decay"] != 1e-4 or config["batch_size"] != 32
                    or config["max_epochs"] != 30 or config["early_stopping_patience"] != 6
                    or config["loss"] != "CrossEntropyLoss" or config["augmentation"] != "none"):
                raise AssertionError(f"Protocol mismatch: {directory}")
            if (normal["split_sha256"] != split["sha256"]
                    or not np.array_equal(np.asarray(normal["mean"], dtype=np.float32), prepared[seed]["mean"])
                    or not np.array_equal(np.asarray(normal["std"], dtype=np.float32), prepared[seed]["std"])):
                raise AssertionError(f"Normalization mismatch: {directory}")
            with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                predictions = list(csv.DictReader(stream))
            if len(predictions) != 200 or [row["filename"] for row in predictions] != split["internal_validation"]:
                raise AssertionError(f"Prediction filenames mismatch: {directory}")
            matrix = [[0, 0], [0, 0]]
            for row in predictions:
                true, predicted = int(row["true_label"]), int(row["predicted_label"])
                if (true != (0 if row["filename"].startswith("cat.") else 1)
                        or predicted not in (0, 1) or int(row["correct"]) != int(true == predicted)
                        or abs(float(row["prob_cat"]) + float(row["prob_dog"]) - 1) > 1e-6):
                    raise AssertionError(f"Invalid prediction: {directory}")
                matrix[true][predicted] += 1
            if (sum(matrix[0]) != 100 or sum(matrix[1]) != 100
                    or matrix != metrics["confusion_matrix"]
                    or abs((matrix[0][0] + matrix[1][1]) / 200 - metrics["validation_accuracy"]) > 1e-12
                    or abs(float(history[metrics["best_epoch"] - 1]["validation_accuracy"]) - metrics["validation_accuracy"]) > 1e-12):
                raise AssertionError(f"Prediction or history metrics mismatch: {directory}")
    print("Audited nine CNN runs, fixed splits/normalization/protocol and 1,800 internal-validation predictions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "tiny-overfit", "run-all", "summarize", "audit"))
    command = parser.parse_args().command
    {"smoke": smoke, "tiny-overfit": tiny_overfit, "run-all": run_all,
     "summarize": summarize, "audit": audit}[command]()


if __name__ == "__main__":
    main()
