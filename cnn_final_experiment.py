"""One frozen 4000-map CNN training run and one held-out evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from cnn_training_strategy_experiments import (
    FLIP_CACHE, STRATEGIES, t1_normalization, validate_flip_cache, verify_base,
)
from dnn_final_experiment import VAL_CACHE, build_val_cache
from models.cnn_architecture import ArchitectureCNN
from representations.cache import HANDCRAFTED, ROOT, current_commit, sha256_file
from representations.dataset import CachedRepresentationDataset
from training import plot_confusion, run_epoch, seed_everything, select_device, write_history, write_json

EXPERIMENT_ID = "CNN-FINAL-001"
ARCHITECTURE_ID = "CNN-ARCH-C"
REPRESENTATION_ID = "REP-006-FUSION"
STRATEGY_ID = STRATEGIES[0]
SEED = 42
EPOCHS = 4
CHANNELS = tuple(range(190))
OUTPUT = ROOT / "outputs/cnn_final" / EXPERIMENT_ID
REPORT = ROOT / "report/cnn_final"
ARCHIVE_FILES = ("config.json", "normalization.json", "train_history.csv",
                 "train_summary.json", "test_metrics.json", "predictions.csv")


def frozen_training_data() -> tuple[np.ndarray, np.ndarray, dict]:
    channels, _ = verify_base()
    flip_manifest = validate_flip_cache()
    decision = json.loads((ROOT / "report/cnn_training/training_strategy_decision.json").read_text(encoding="utf-8"))
    if (channels != CHANNELS or decision["selected_individual_strategy"] != STRATEGY_ID
            or decision["combination_eligible"] or flip_manifest["image_count"] != 2000):
        raise AssertionError("Frozen CNN final representation or strategy mismatch")
    original = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    flipped = np.load(FLIP_CACHE / "labels.npy", mmap_mode="r")
    if (len(original) != 2000 or len(flipped) != 2000
            or Counter(original.tolist()) != {0: 1000, 1: 1000}
            or not np.array_equal(original, flipped)):
        raise AssertionError("Training cache labels or class balance changed")
    mean, std = t1_normalization(list(range(2000)))
    if (mean.shape != (190,) or std.shape != (190,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std < 1e-6)):
        raise AssertionError("Invalid 4000-map training normalization")
    return mean, std, flip_manifest


def training_dataset(mean: np.ndarray, std: np.ndarray) -> ConcatDataset:
    indices = list(range(2000))
    return ConcatDataset((
        CachedRepresentationDataset(HANDCRAFTED / "features.npy", HANDCRAFTED / "labels.npy",
                                    indices, CHANNELS, mean, std),
        CachedRepresentationDataset(FLIP_CACHE / "features.npy", FLIP_CACHE / "labels.npy",
                                    indices, CHANNELS, mean, std),
    ))


def smoke() -> None:
    mean, std, _ = frozen_training_data()
    dataset = training_dataset(mean, std)
    batch = next(iter(DataLoader(dataset, batch_size=8)))
    model = ArchitectureCNN(ARCHITECTURE_ID)
    if (len(dataset) != 4000 or batch[0].shape != (8, 190, 7, 7)
            or model(batch[0]).shape != (8, 2)
            or any(isinstance(layer, nn.Dropout2d) for layer in model.modules())):
        raise AssertionError("Frozen CNN final smoke check failed")
    nn.CrossEntropyLoss()(model(batch[0]), batch[1]).backward()
    print("Frozen caches, 4000-map normalization and CNN forward/backward OK", flush=True)


def train() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"Final run directory already contains artifacts: {OUTPUT}")
    mean, std, _ = frozen_training_data()
    seed_everything(SEED)
    device = select_device("auto")
    model = ArchitectureCNN(ARCHITECTURE_ID).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != 89794 or any(isinstance(layer, nn.Dropout2d) for layer in model.modules()):
        raise AssertionError("CNN-ARCH-C architecture changed")
    dataset = training_dataset(mean, std)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED), pin_memory=device.type == "cuda")
    config = {
        "experiment_id": EXPERIMENT_ID, "model_family": "cnn", "architecture_id": ARCHITECTURE_ID,
        "representation_id": REPRESENTATION_ID, "representation_shape": [190, 7, 7],
        "strategy_id": STRATEGY_ID, "class_to_idx": {"cat": 0, "dog": 1},
        "training_seed": SEED, "original_training_samples": 2000,
        "flipped_training_samples": 2000, "effective_training_samples": 4000,
        "training_class_counts": {"cat": 2000, "dog": 2000},
        "epochs": EPOCHS, "early_stopping": False, "internal_validation_samples": 0,
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "batch_size": 32, "loss": "CrossEntropyLoss", "scheduler": None,
        "augmentation": "deterministic_horizontal_flip", "dropout2d_p": 0.0,
        "checkpoint_selection": "final state after exactly four full-data epochs",
        "parameter_count": parameter_count, "device": str(device), "code_commit": current_commit(),
        "train_cache_manifest_sha256": sha256_file(HANDCRAFTED / "manifest.json"),
        "flipped_cache_manifest_sha256": sha256_file(FLIP_CACHE / "manifest.json"),
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "config.json", config)
    normal = {"source": "all 2000 original and 2000 flipped data/train maps only",
              "channels": 190, "mean": mean.tolist(), "std": std.tolist()}
    write_json(OUTPUT / "normalization.json", normal)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        measured = run_epoch(model, loader, criterion, device, optimizer)
        if measured["samples"] != 4000 or [sum(row) for row in measured["confusion_matrix"]] != [2000, 2000]:
            raise AssertionError("Final training epoch did not use all 4000 balanced examples")
        history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                        "train_loss": measured["loss"], "train_accuracy": measured["accuracy"],
                        "train_cat_accuracy": measured["cat_accuracy"],
                        "train_dog_accuracy": measured["dog_accuracy"]})
        write_history(OUTPUT / "train_history.csv", history)
        print(f"{EXPERIMENT_ID} epoch {epoch}: {history[-1]}", flush=True)
    training_seconds = time.perf_counter() - started
    torch.save({**config, "model_state": model.state_dict(),
                "normalization": {"mean": mean.tolist(), "std": std.tolist()},
                "normalization_sha256": sha256_file(OUTPUT / "normalization.json")},
               OUTPUT / "final_model.pt")
    write_json(OUTPUT / "train_summary.json", {
        "experiment_id": EXPERIMENT_ID, "architecture_id": ARCHITECTURE_ID,
        "representation_id": REPRESENTATION_ID, "parameter_count": parameter_count,
        "original_training_samples": 2000, "flipped_training_samples": 2000,
        "effective_training_samples": 4000, "epochs": EPOCHS, "seed": SEED,
        "training_seconds": training_seconds, "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_train_cat_accuracy": history[-1]["train_cat_accuracy"],
        "final_train_dog_accuracy": history[-1]["train_dog_accuracy"],
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "batch_size": 32, "augmentation": "deterministic_horizontal_flip",
        "checkpoint_sha256": sha256_file(OUTPUT / "final_model.pt"), "status": "training_complete",
    })


def evaluate() -> None:
    if (OUTPUT / "test_metrics.json").exists() or (OUTPUT / "predictions.csv").exists():
        raise RuntimeError("Held-out evaluation already exists; no second evaluation allowed")
    checkpoint = torch.load(OUTPUT / "final_model.pt", map_location="cpu", weights_only=True)
    config = json.loads((OUTPUT / "config.json").read_text(encoding="utf-8"))
    if any(checkpoint[key] != value for key, value in config.items()):
        raise AssertionError("Checkpoint metadata differs from frozen config")
    normal = json.loads((OUTPUT / "normalization.json").read_text(encoding="utf-8"))
    if checkpoint["normalization_sha256"] != sha256_file(OUTPUT / "normalization.json"):
        raise AssertionError("Final normalization file changed")
    mean, std = (np.asarray(normal[key], dtype=np.float32) for key in ("mean", "std"))
    if (not np.array_equal(mean, np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32))
            or not np.array_equal(std, np.asarray(checkpoint["normalization"]["std"], dtype=np.float32))):
        raise AssertionError("Checkpoint normalization mismatch")
    val_manifest, names = build_val_cache()
    device = select_device("auto")
    model = ArchitectureCNN(ARCHITECTURE_ID).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = CachedRepresentationDataset(VAL_CACHE / "features.npy", VAL_CACHE / "labels.npy",
                                          list(range(500)), CHANNELS, mean, std)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    logits_batches, label_batches = [], []
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device, non_blocking=True))
            total_loss += criterion(logits, labels.to(device)).item() * len(labels)
            logits_batches.append(logits.cpu())
            label_batches.append(labels)
    logits = torch.cat(logits_batches)
    labels = torch.cat(label_batches).numpy()
    predictions = logits.argmax(dim=1).numpy()
    probabilities = logits.softmax(dim=1).numpy()
    matrix = [[int(((labels == actual) & (predictions == predicted)).sum())
               for predicted in (0, 1)] for actual in (0, 1)]
    if len(labels) != 500 or [sum(row) for row in matrix] != [250, 250]:
        raise AssertionError("Held-out count or class balance changed")
    cat_acc, dog_acc = matrix[0][0] / 250, matrix[1][1] / 250
    f1_cat = 2 * matrix[0][0] / (2 * matrix[0][0] + matrix[0][1] + matrix[1][0])
    f1_dog = 2 * matrix[1][1] / (2 * matrix[1][1] + matrix[0][1] + matrix[1][0])
    metrics = {
        "experiment_id": EXPERIMENT_ID, "test_dataset": "data/val", "samples": 500,
        "cat_samples": 250, "dog_samples": 250, "loss": total_loss / 500,
        "accuracy": float((predictions == labels).mean()), "cat_accuracy": cat_acc,
        "dog_accuracy": dog_acc, "balanced_accuracy": (cat_acc + dog_acc) / 2,
        "macro_f1": (f1_cat + f1_dog) / 2, "confusion_matrix": matrix,
        "checkpoint_sha256": sha256_file(OUTPUT / "final_model.pt"),
        "train_cache_manifest_sha256": config["train_cache_manifest_sha256"],
        "flipped_cache_manifest_sha256": config["flipped_cache_manifest_sha256"],
        "val_cache_manifest_sha256": sha256_file(VAL_CACHE / "manifest.json"),
        "checkpoint_reloaded_for_evaluation": True, "status": "complete",
    }
    with (OUTPUT / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "true_label", "predicted_label", "logit_cat", "logit_dog",
                         "prob_cat", "prob_dog", "correct"))
        for index, name in enumerate(names):
            writer.writerow((name, int(labels[index]), int(predictions[index]),
                             float(logits[index, 0]), float(logits[index, 1]),
                             float(probabilities[index, 0]), float(probabilities[index, 1]),
                             int(labels[index] == predictions[index])))
    write_json(OUTPUT / "test_metrics.json", metrics)
    plot_confusion(matrix, OUTPUT / "confusion_matrix.png", f"{EXPERIMENT_ID} held-out confusion matrix")
    print(f"One held-out evaluation complete: {json.dumps(metrics)}", flush=True)


def audit() -> None:
    mean, std, _ = frozen_training_data()
    config = json.loads((OUTPUT / "config.json").read_text(encoding="utf-8"))
    normal = json.loads((OUTPUT / "normalization.json").read_text(encoding="utf-8"))
    summary = json.loads((OUTPUT / "train_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUTPUT / "test_metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(OUTPUT / "final_model.pt", map_location="cpu", weights_only=True)
    model = ArchitectureCNN(ARCHITECTURE_ID)
    model.load_state_dict(checkpoint["model_state"])
    with (OUTPUT / "train_history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    with (OUTPUT / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    val_names = json.loads((VAL_CACHE / "filenames.json").read_text(encoding="utf-8"))
    val_labels = np.load(VAL_CACHE / "labels.npy", mmap_mode="r")
    if (config["representation_id"] != REPRESENTATION_ID or config["strategy_id"] != STRATEGY_ID
            or config["epochs"] != 4 or config["training_seed"] != 42
            or config["effective_training_samples"] != 4000 or config["internal_validation_samples"] != 0
            or config["scheduler"] is not None or config["early_stopping"] or config["dropout2d_p"] != 0
            or any(isinstance(layer, nn.Dropout2d) for layer in model.modules())
            or len(history) != 4 or [int(row["epoch"]) for row in history] != [1, 2, 3, 4]
            or any(float(row["learning_rate"]) != 3e-4 for row in history)
            or len(rows) != 500 or [row["filename"] for row in rows] != val_names
            or [int(row["true_label"]) for row in rows] != list(val_labels)
            or not np.array_equal(mean, np.asarray(normal["mean"], dtype=np.float32))
            or not np.array_equal(std, np.asarray(normal["std"], dtype=np.float32))
            or summary["checkpoint_sha256"] != metrics["checkpoint_sha256"]
            or metrics["checkpoint_sha256"] != sha256_file(OUTPUT / "final_model.pt")):
        raise AssertionError("Final CNN training, normalization or evaluation protocol mismatch")
    matrix = [[0, 0], [0, 0]]
    row_losses = []
    for row in rows:
        actual, predicted = int(row["true_label"]), int(row["predicted_label"])
        if int(row["correct"]) != int(actual == predicted):
            raise AssertionError("Prediction correctness mismatch")
        logits = np.asarray([float(row["logit_cat"]), float(row["logit_dog"])], dtype=np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        if (abs(float(row["prob_cat"]) - probs[0]) > 1e-6
                or abs(float(row["prob_dog"]) - probs[1]) > 1e-6):
            raise AssertionError("Prediction probabilities mismatch")
        row_losses.append(-np.log(probs[actual]))
        matrix[actual][predicted] += 1
    f1_cat = 2 * matrix[0][0] / (2 * matrix[0][0] + matrix[0][1] + matrix[1][0])
    f1_dog = 2 * matrix[1][1] / (2 * matrix[1][1] + matrix[0][1] + matrix[1][0])
    if (matrix != metrics["confusion_matrix"] or [sum(row) for row in matrix] != [250, 250]
            or abs((matrix[0][0] + matrix[1][1]) / 500 - metrics["accuracy"]) > 1e-12
            or abs(matrix[0][0] / 250 - metrics["cat_accuracy"]) > 1e-12
            or abs(matrix[1][1] / 250 - metrics["dog_accuracy"]) > 1e-12
            or abs((f1_cat + f1_dog) / 2 - metrics["macro_f1"]) > 1e-12
            or abs(float(np.mean(row_losses)) - metrics["loss"]) > 1e-6):
        raise AssertionError("Saved predictions and test metrics differ")
    print("Audited 4000-map/four-epoch training, reloaded checkpoint and 500 saved predictions", flush=True)


def archive() -> None:
    audit()
    REPORT.mkdir(parents=True, exist_ok=True)
    for filename in ARCHIVE_FILES:
        shutil.copy2(OUTPUT / filename, REPORT / filename)
    shutil.copy2(OUTPUT / "confusion_matrix.png", ROOT / "report/figures/cnn_final_confusion_matrix.png")
    metrics = json.loads((REPORT / "test_metrics.json").read_text(encoding="utf-8"))
    training = json.loads((REPORT / "train_summary.json").read_text(encoding="utf-8"))
    decision = json.loads((ROOT / "report/cnn_training/training_strategy_decision.json").read_text(encoding="utf-8"))
    dnn = json.loads((ROOT / "report/dnn_final/test_metrics.json").read_text(encoding="utf-8"))
    dnn_training = json.loads((ROOT / "report/dnn_final/config.json").read_text(encoding="utf-8"))
    pct = lambda value: f"{100 * value:.2f}%"
    pp = lambda value: f"{100 * value:+.2f} 个百分点"
    lines = [
        f"# {EXPERIMENT_ID}：最终 CNN 实验", "",
        "## 1. 冻结流程", "",
        "原图 → 保留长宽比的 128×128 缩放与反射填充 → HOG + LBP + HSV + RootSIFT → `REP-006-FUSION` `[190,7,7]` → CNN-ARCH-C → 猫/狗两类 logits。", "",
        "## 2. 架构选择", "",
        "CNN-ARCH-001 在三份 `data/train` 内部划分上比较三个冻结候选，选定 CNN-ARCH-C。网络为 190→96 的 1×1 投影，接 1×1 / 3×3 / 膨胀 3×3 三分支（各 32 通道）、96→128 的 1×1 融合、全局平均池化和 `Linear(128,2)`；卷积后有 BatchNorm 与 ReLU。", "",
        "## 3. 训练策略选择", "",
        f"CNN-TRAIN-001 比较 T0 基线和 T1 镜像、T2 Dropout2d、T3 余弦学习率。T1 三种子内部验证为 {pct(decision['models'][STRATEGY_ID]['mean_validation_accuracy'])} ± {pct(decision['models'][STRATEGY_ID]['sample_std_validation_accuracy'])}，按预设规则入选；没有策略达到 +1.0 个百分点的强改进门槛，未进行组合实验。", "",
        "## 4. 最终训练", "",
        f"全部 2000 张训练原图与各自的 2000 张确定性水平镜像图组成 4000 张特征图；归一化只由这 4000 张拟合。随机种子 42，AdamW，学习率 3e-4，权重衰减 1e-4，批量 32，交叉熵，恰好 4 轮，没有内部验证或早停。参数量 {training['parameter_count']:,}，训练耗时 {training['training_seconds']:.2f} 秒，末轮训练准确率 {pct(training['final_train_accuracy'])}。", "",
        "## 5. 500 张保留图结果", "",
        "重载第 4 轮最终检查点，对 500 张未镜像原图应用本次 4000 张训练图的归一化并评估一次；没有测试时增强或阈值调整。", "",
        "| 损失 | 总体准确率 | 猫准确率 | 狗准确率 | 平衡准确率 | Macro F1 |", "|---:|---:|---:|---:|---:|---:|",
        f"| {metrics['loss']:.4f} | {pct(metrics['accuracy'])} | {pct(metrics['cat_accuracy'])} | {pct(metrics['dog_accuracy'])} | {pct(metrics['balanced_accuracy'])} | {pct(metrics['macro_f1'])} |", "",
        f"混淆矩阵（真实行/预测列，猫、狗顺序）：`{metrics['confusion_matrix']}`。", "",
        "![CNN 最终混淆矩阵](../figures/cnn_final_confusion_matrix.png)", "",
        "## 6. 与 DNN-FINAL-001 比较及限制", "",
        f"两者都使用 REP-006-FUSION。DNN-FINAL-001：{dnn_training['parameter_count']:,} 参数、保留集 {pct(dnn['accuracy'])}；CNN-FINAL-001：{training['parameter_count']:,} 参数、{pct(metrics['accuracy'])}。CNN 相对 DNN 的总体准确率变化 {pp(metrics['accuracy'] - dnn['accuracy'])}，猫 {pp(metrics['cat_accuracy'] - dnn['cat_accuracy'])}，狗 {pp(metrics['dog_accuracy'] - dnn['dog_accuracy'])}，Macro F1 {pp(metrics['macro_f1'] - dnn['macro_f1'])}。CNN 使用了确定性水平镜像，DNN 未使用相同增强；训练过程也不同，因此不能把差异仅归因于网络结构。", "",
        "这 500 张保留图此前已用于历史 DNN 评估，但未参与 CNN 架构或训练策略选择；CNN-FINAL-001 的配置在此次评估前冻结。配置、逐轮记录、190 通道归一化和 500 行预测均在本目录；模型检查点保留在忽略的 `outputs/cnn_final/CNN-FINAL-001/`。", "",
    ]
    (REPORT / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "evaluate", "audit", "archive"))
    command = parser.parse_args().command
    {"smoke": smoke, "train": train, "evaluate": evaluate, "audit": audit, "archive": archive}[command]()


if __name__ == "__main__":
    main()
