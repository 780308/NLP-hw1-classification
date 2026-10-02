"""Frozen ten-epoch full-data RNN training and one held-out evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from cnn_training_strategy_experiments import FLIP_CACHE, t1_normalization, validate_flip_cache
from dnn_final_experiment import VAL_CACHE, build_val_cache
from models.rnn2d import ARCHITECTURES, VisionRNN2DClassifier
from representations.cache import HANDCRAFTED, ROOT, current_commit, sha256_file
from representations.dataset import CachedRepresentationDataset
from rnn2d_architecture_experiments import epoch_pass
from training import plot_confusion, seed_everything, select_device, write_history, write_json

EXPERIMENT_ID = "RNN-FINAL-001"
ARCHITECTURE_ID = "RNN2D-ARCH-BASE"
IMPLEMENTATION_ID = "RNN2D-ARCH-L"
REPRESENTATION_ID = "REP-006-FUSION"
SEED = 42
EPOCHS = 10
CHANNELS = tuple(range(190))
OUTPUT = ROOT / "outputs/rnn_final" / EXPERIMENT_ID
REPORT = ROOT / "report/rnn_final"
ARCHIVE_FILES = ("config.json", "normalization.json", "train_history.csv",
                 "train_summary.json", "test_metrics.json", "predictions.csv")


def frozen_training_data() -> tuple[np.ndarray, np.ndarray]:
    flip_manifest = validate_flip_cache()  # Validates original and RGB-flipped training banks.
    decision = json.loads((ROOT / "report/rnn_lr_optimization/lr_decision.json").read_text(encoding="utf-8"))
    if decision["selected_strategy"] != "RNN-LR-T0-CONST-3E4":
        raise AssertionError("Final learning-rate decision changed")
    with (ROOT / "report/rnn_lr_optimization/selected_policy_matched_epochs.csv").open(
            newline="", encoding="utf-8") as stream:
        matched = {int(row["epoch"]): row for row in csv.DictReader(stream)}
    if (not np.isclose(float(matched[10]["mean_validation_accuracy"]), .8266666666666667)
            or not np.isclose(float(matched[11]["mean_validation_accuracy"]), .8183333333333334)):
        raise AssertionError("Frozen epoch-count evidence changed")
    original_labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    flipped_labels = np.load(FLIP_CACHE / "labels.npy", mmap_mode="r")
    if (flip_manifest["feature_shape"] != [2000, 190, 7, 7]
            or Counter(original_labels.tolist()) != {0: 1000, 1: 1000}
            or not np.array_equal(original_labels, flipped_labels)):
        raise AssertionError("Full training cache size, balance or label alignment changed")
    mean, std = t1_normalization(list(range(2000)))
    if (mean.shape != (190,) or std.shape != (190,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std < 1e-6)):
        raise AssertionError("Invalid full-data training normalization")
    return mean, std


def training_dataset(mean: np.ndarray, std: np.ndarray) -> ConcatDataset:
    indices = list(range(2000))
    return ConcatDataset((
        CachedRepresentationDataset(HANDCRAFTED / "features.npy", HANDCRAFTED / "labels.npy",
                                    indices, CHANNELS, mean, std),
        CachedRepresentationDataset(FLIP_CACHE / "features.npy", FLIP_CACHE / "labels.npy",
                                    indices, CHANNELS, mean, std),
    ))


def check_architecture(model: VisionRNN2DClassifier) -> int:
    spec = ARCHITECTURES[IMPLEMENTATION_ID]
    recurrent = [layer for layer in model.modules() if isinstance(layer, nn.RNN)]
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if (spec != {"embed_dim": 320, "rnn_hidden_size": 80, "num_blocks": 3}
            or model.specification != spec or count != 1_992_642 or len(model.blocks) != 3
            or len(recurrent) != 6
            or any(layer.input_size != 320 or layer.hidden_size != 80 or layer.num_layers != 1
                   or layer.nonlinearity != "tanh" or not layer.bidirectional or not layer.batch_first
                   for layer in recurrent)
            or any(isinstance(layer, (nn.Conv2d, nn.LSTM, nn.GRU, nn.Dropout)) for layer in model.modules())):
        raise AssertionError("Frozen two-axis RNN architecture changed")
    return count


def smoke() -> None:
    mean, std = frozen_training_data()
    dataset = training_dataset(mean, std)
    features, labels = next(iter(DataLoader(dataset, batch_size=8)))
    model = VisionRNN2DClassifier(IMPLEMENTATION_ID)
    count = check_architecture(model)
    if len(dataset) != 4000 or features.shape != (8, 190, 7, 7) or model(features).shape != (8, 2):
        raise AssertionError("Full-data RNN tensor or sample count changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(model(features), labels).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print(f"Frozen train caches, 4000-map normalization, forward/backward/clip/step: OK; {count:,} parameters")


def train() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"Final run directory already contains artifacts: {OUTPUT}")
    mean, std = frozen_training_data()
    seed_everything(SEED)
    device = select_device("auto")
    model = VisionRNN2DClassifier(IMPLEMENTATION_ID).to(device)  # Fresh initialization.
    count = check_architecture(model)
    dataset = training_dataset(mean, std)
    if len(dataset) != 4000:
        raise AssertionError("Final training omitted original or flipped maps")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED), pin_memory=device.type == "cuda")
    config = {
        "experiment_id": EXPERIMENT_ID, "model_family": "RNN",
        "architecture_id": ARCHITECTURE_ID, "implementation_id": IMPLEMENTATION_ID,
        "representation_id": REPRESENTATION_ID, "representation_shape": [190, 7, 7],
        **ARCHITECTURES[IMPLEMENTATION_ID], "mlp_expansion_ratio": 2,
        "class_to_idx": {"cat": 0, "dog": 1}, "parameter_count": count,
        "training_seed": SEED, "original_training_samples": 2000,
        "flipped_training_samples": 2000, "effective_training_samples": 4000,
        "training_class_counts": {"cat": 2000, "dog": 2000},
        "epochs": EPOCHS, "internal_validation_samples": 0, "early_stopping": False,
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "batch_size": 32, "gradient_clip_norm": 1.0, "loss": "CrossEntropyLoss",
        "scheduler": None, "dropout_probability": 0.0,
        "augmentation": "deterministic_horizontal_flip",
        "checkpoint_selection": "final state after exactly ten full-data epochs",
        "device": str(device), "code_commit": current_commit(),
        "train_cache_manifest_sha256": sha256_file(HANDCRAFTED / "manifest.json"),
        "flipped_cache_manifest_sha256": sha256_file(FLIP_CACHE / "manifest.json"),
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "normalization.json", {
        "source": "all 2000 original and 2000 RGB-flipped data/train maps only",
        "channels": 190, "fit_samples": 4000, "mean": mean.tolist(), "std": std.tolist(),
    })
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        measured = epoch_pass(model, loader, criterion, device, optimizer)
        if measured["samples"] != 4000 or [sum(row) for row in measured["confusion_matrix"]] != [2000, 2000]:
            raise AssertionError("Final training epoch omitted samples")
        history.append({
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": measured["loss"], "train_accuracy": measured["accuracy"],
            "train_cat_accuracy": measured["cat_accuracy"],
            "train_dog_accuracy": measured["dog_accuracy"],
        })
        write_history(OUTPUT / "train_history.csv", history)
        print(f"{EXPERIMENT_ID} epoch {epoch:02d}: loss={measured['loss']:.4f}, "
              f"accuracy={measured['accuracy']:.4f}", flush=True)
    training_seconds = time.perf_counter() - started
    normalization_hash = sha256_file(OUTPUT / "normalization.json")
    torch.save({**config, "model_state": model.state_dict(),
                "normalization_sha256": normalization_hash,
                "normalization": {"mean": mean.tolist(), "std": std.tolist()}},
               OUTPUT / "final_model.pt")
    checkpoint_hash = sha256_file(OUTPUT / "final_model.pt")
    write_json(OUTPUT / "train_summary.json", {
        "experiment_id": EXPERIMENT_ID, "architecture_id": ARCHITECTURE_ID,
        "representation_id": REPRESENTATION_ID, "parameter_count": count,
        **ARCHITECTURES[IMPLEMENTATION_ID],
        "original_training_samples": 2000, "flipped_training_samples": 2000,
        "effective_training_samples": 4000, "seed": SEED, "epochs": EPOCHS,
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "batch_size": 32, "gradient_clip_norm": 1.0,
        "augmentation": "deterministic_horizontal_flip",
        "training_seconds": training_seconds,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_train_cat_accuracy": history[-1]["train_cat_accuracy"],
        "final_train_dog_accuracy": history[-1]["train_dog_accuracy"],
        "checkpoint_sha256": checkpoint_hash, "status": "training_complete",
    })
    print(f"Final epoch-10 checkpoint saved: SHA-256 {checkpoint_hash}", flush=True)


def evaluate() -> None:
    if (OUTPUT / "test_metrics.json").exists() or (OUTPUT / "predictions.csv").exists():
        raise RuntimeError("RNN held-out evaluation already exists; second evaluation is forbidden")
    checkpoint_path = OUTPUT / "final_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = json.loads((OUTPUT / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((OUTPUT / "train_summary.json").read_text(encoding="utf-8"))
    if (any(checkpoint[key] != value for key, value in config.items())
            or summary["status"] != "training_complete" or summary["epochs"] != EPOCHS
            or summary["checkpoint_sha256"] != sha256_file(checkpoint_path)):
        raise AssertionError("Final checkpoint differs from frozen ten-epoch training")
    normal = json.loads((OUTPUT / "normalization.json").read_text(encoding="utf-8"))
    if checkpoint["normalization_sha256"] != sha256_file(OUTPUT / "normalization.json"):
        raise AssertionError("Final normalization artifact changed")
    mean, std = (np.asarray(normal[key], dtype=np.float32) for key in ("mean", "std"))
    if (not np.array_equal(mean, np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32))
            or not np.array_equal(std, np.asarray(checkpoint["normalization"]["std"], dtype=np.float32))):
        raise AssertionError("Checkpoint normalization does not match training-only artifact")
    val_manifest, names = build_val_cache()  # First held-out access, after final checkpoint is frozen.
    val_features = np.load(VAL_CACHE / "features.npy", mmap_mode="r")
    val_labels = np.load(VAL_CACHE / "labels.npy", mmap_mode="r")
    if (val_manifest["feature_shape"] != [500, 190, 7, 7]
            or val_features.shape != (500, 190, 7, 7)
            or Counter(val_labels.tolist()) != {0: 250, 1: 250}
            or len(names) != 500 or len(set(names)) != 500 or not np.isfinite(val_features).all()):
        raise AssertionError("Held-out REP-006 cache incompatibility")
    device = select_device("auto")
    model = VisionRNN2DClassifier(IMPLEMENTATION_ID).to(device)
    if check_architecture(model) != config["parameter_count"]:
        raise AssertionError("Reload architecture differs from frozen model")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = CachedRepresentationDataset(VAL_CACHE / "features.npy", VAL_CACHE / "labels.npy",
                                          list(range(500)), CHANNELS, mean, std)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    logits_batches, label_batches = [], []
    loss_sum = 0.0
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device, non_blocking=True))
            loss_sum += criterion(logits, labels.to(device)).item() * len(labels)
            logits_batches.append(logits.cpu())
            label_batches.append(labels)
    logits = torch.cat(logits_batches)
    labels = torch.cat(label_batches).numpy()
    if not np.array_equal(labels, val_labels):
        raise AssertionError("Held-out label order changed")
    predictions = logits.argmax(dim=1).numpy()
    probabilities = logits.softmax(dim=1).numpy()
    matrix = [[int(((labels == actual) & (predictions == predicted)).sum())
               for predicted in (0, 1)] for actual in (0, 1)]
    if len(labels) != 500 or [sum(row) for row in matrix] != [250, 250]:
        raise AssertionError("Held-out sample count changed")
    cat_accuracy, dog_accuracy = matrix[0][0] / 250, matrix[1][1] / 250
    f1_cat = 2 * matrix[0][0] / (2 * matrix[0][0] + matrix[0][1] + matrix[1][0])
    f1_dog = 2 * matrix[1][1] / (2 * matrix[1][1] + matrix[0][1] + matrix[1][0])
    metrics = {
        "experiment_id": EXPERIMENT_ID, "test_dataset": "data/val",
        "samples": 500, "cat_samples": 250, "dog_samples": 250,
        "loss": loss_sum / 500, "accuracy": float((predictions == labels).mean()),
        "cat_accuracy": cat_accuracy, "dog_accuracy": dog_accuracy,
        "balanced_accuracy": (cat_accuracy + dog_accuracy) / 2,
        "macro_f1": (f1_cat + f1_dog) / 2, "confusion_matrix": matrix,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_reloaded_for_evaluation": True,
        "train_cache_manifest_sha256": config["train_cache_manifest_sha256"],
        "flipped_cache_manifest_sha256": config["flipped_cache_manifest_sha256"],
        "val_cache_manifest_sha256": sha256_file(VAL_CACHE / "manifest.json"),
        "status": "complete",
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
    print(f"Single held-out evaluation complete: {json.dumps(metrics)}", flush=True)


def audit() -> None:
    mean, std = frozen_training_data()
    config = json.loads((OUTPUT / "config.json").read_text(encoding="utf-8"))
    normal = json.loads((OUTPUT / "normalization.json").read_text(encoding="utf-8"))
    summary = json.loads((OUTPUT / "train_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUTPUT / "test_metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(OUTPUT / "final_model.pt", map_location="cpu", weights_only=True)
    model = VisionRNN2DClassifier(IMPLEMENTATION_ID)
    count = check_architecture(model)
    model.load_state_dict(checkpoint["model_state"])
    with (OUTPUT / "train_history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    with (OUTPUT / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    val_names = json.loads((VAL_CACHE / "filenames.json").read_text(encoding="utf-8"))
    val_labels = np.load(VAL_CACHE / "labels.npy", mmap_mode="r")
    if (config["experiment_id"] != EXPERIMENT_ID or config["architecture_id"] != ARCHITECTURE_ID
            or config["implementation_id"] != IMPLEMENTATION_ID
            or config["representation_id"] != REPRESENTATION_ID
            or config["representation_shape"] != [190, 7, 7]
            or config["parameter_count"] != count or config["training_seed"] != 42
            or config["original_training_samples"] != 2000
            or config["flipped_training_samples"] != 2000
            or config["effective_training_samples"] != 4000
            or config["internal_validation_samples"] != 0
            or config["epochs"] != 10 or config["early_stopping"]
            or config["optimizer"] != "AdamW" or config["learning_rate"] != 3e-4
            or config["weight_decay"] != 1e-4 or config["batch_size"] != 32
            or config["gradient_clip_norm"] != 1.0 or config["scheduler"] is not None
            or config["dropout_probability"] != 0
            or config["augmentation"] != "deterministic_horizontal_flip"
            or config["train_cache_manifest_sha256"] != sha256_file(HANDCRAFTED / "manifest.json")
            or config["flipped_cache_manifest_sha256"] != sha256_file(FLIP_CACHE / "manifest.json")
            or any(checkpoint[key] != value for key, value in config.items())
            or checkpoint["normalization_sha256"] != sha256_file(OUTPUT / "normalization.json")
            or len(history) != 10 or [int(row["epoch"]) for row in history] != list(range(1, 11))
            or any(float(row["learning_rate"]) != 3e-4 for row in history)
            or summary["status"] != "training_complete" or summary["epochs"] != 10
            or summary["checkpoint_sha256"] != sha256_file(OUTPUT / "final_model.pt")
            or metrics["checkpoint_sha256"] != summary["checkpoint_sha256"]
            or not metrics["checkpoint_reloaded_for_evaluation"]
            or not np.array_equal(mean, np.asarray(normal["mean"], dtype=np.float32))
            or not np.array_equal(std, np.asarray(normal["std"], dtype=np.float32))
            or len(rows) != 500 or [row["filename"] for row in rows] != val_names
            or [int(row["true_label"]) for row in rows] != list(val_labels)):
        raise AssertionError("Final RNN training or held-out protocol mismatch")
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
            raise AssertionError("Saved probability differs from logits")
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
        raise AssertionError("Saved RNN predictions and held-out metrics differ")
    print("Audited fresh 4000-map/ten-epoch RNN training, checkpoint and 500 saved predictions")


def archive() -> None:
    audit()
    REPORT.mkdir(parents=True, exist_ok=True)
    for filename in ARCHIVE_FILES:
        shutil.copy2(OUTPUT / filename, REPORT / filename)
    figure = ROOT / "report/figures/rnn_final_confusion_matrix.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT / "confusion_matrix.png", figure)
    metrics = json.loads((REPORT / "test_metrics.json").read_text(encoding="utf-8"))
    training = json.loads((REPORT / "train_summary.json").read_text(encoding="utf-8"))
    rnn_development = json.loads((ROOT / "report/rnn_lr_optimization/lr_decision.json").read_text(encoding="utf-8"))
    rnn_architecture = json.loads((ROOT / "report/rnn2d_architecture/architecture_decision.json").read_text(encoding="utf-8"))
    simple_rnn = json.loads((ROOT / "report/rnn_architecture/architecture_decision.json").read_text(encoding="utf-8"))
    dnn = json.loads((ROOT / "report/dnn_final/test_metrics.json").read_text(encoding="utf-8"))
    cnn = json.loads((ROOT / "report/cnn_final/test_metrics.json").read_text(encoding="utf-8"))
    dnn_config = json.loads((ROOT / "report/dnn_final/config.json").read_text(encoding="utf-8"))
    cnn_config = json.loads((ROOT / "report/cnn_final/config.json").read_text(encoding="utf-8"))
    pct = lambda value: f"{100 * value:.2f}%"
    lines = [
        f"# {EXPERIMENT_ID}：最终 RNN 实验", "",
        "## 1. 冻结流程", "",
        "RGB 原图 → 保长宽比缩放及反射填充至 128×128 → HOG + LBP + HSV + RootSIFT → REP-006 `[190,7,7]` → 逐位置 `Linear(190,320)` 与 LayerNorm → 三块双轴双向 tanh `nn.RNN` 残差块 → 7×7 位置全局均值池化 → `Linear(320,2)` 猫狗分类。每块沿水平和垂直轴使用独立的双向 RNN，保留所有时刻输出；不含卷积、注意力、LSTM、GRU 或 Dropout。", "",
        "## 2. 架构与策略选择依据", "",
        f"简单一维 RNN 开发阶段的最佳三种子均值为 {pct(simple_rnn['models'][simple_rnn['selected_architecture']]['mean_validation_accuracy'])}；双轴 RNN2D 的 L 为 {pct(rnn_architecture['models']['RNN2D-ARCH-L']['mean_validation_accuracy'])}。加入原图 RGB 水平镜像训练后，冻结 BASE 的内部验证均值为 {pct(rnn_development['models']['RNN-LR-T0-CONST-3E4']['mean_validation_accuracy'])}。随后加宽和学习率调度没有达到预设改进门槛，因此保留 BASE 与恒定 `3e-4`。最终 10 轮数取自三种子共同轮次曲线，而非本次留出集。", "",
        "## 3. 最终训练", "",
        f"全部 2000 张训练原图及 2000 张 RGB 水平镜像图构成 4000 张特征图（猫、狗各 2000）；仅用它们拟合 190 通道归一化。新模型从头初始化，种子 42；AdamW，恒定学习率 `3e-4`，权重衰减 `1e-4`，批量 32，梯度范数裁剪 1.0，交叉熵，恰好 10 轮；无验证、早停或调度器。模型参数 {training['parameter_count']:,}，训练耗时 {training['training_seconds']:.2f} 秒，第 10 轮训练损失 {training['final_train_loss']:.4f}，准确率 {pct(training['final_train_accuracy'])}（猫 {pct(training['final_train_cat_accuracy'])}，狗 {pct(training['final_train_dog_accuracy'])}）。仅保存第 10 轮状态，SHA-256：`{training['checkpoint_sha256']}`。", "",
        "## 4. 一次留出集评估", "",
        f"从磁盘重载第 10 轮检查点，用训练集归一化对 `data/val` 的 500 张原图各推理一次。交叉熵损失 **{metrics['loss']:.4f}**；总体准确率 **{pct(metrics['accuracy'])}**；猫 **{pct(metrics['cat_accuracy'])}**；狗 **{pct(metrics['dog_accuracy'])}**；平衡准确率 **{pct(metrics['balanced_accuracy'])}**；Macro F1 **{pct(metrics['macro_f1'])}**。混淆矩阵行是真实猫/狗、列是预测猫/狗：`{metrics['confusion_matrix']}`。", "",
        "![RNN 最终混淆矩阵](../figures/rnn_final_confusion_matrix.png)", "",
        "## 5. DNN / CNN / RNN 最终结果对照", "",
        "| 模型 | 参数量 | 总体准确率 | 猫准确率 | 狗准确率 | 平衡准确率 | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values, count in (("DNN-FINAL-001", dnn, dnn_config["parameter_count"]),
                                ("CNN-FINAL-001", cnn, cnn_config["parameter_count"]),
                                (EXPERIMENT_ID, metrics, training["parameter_count"])):
        lines.append(f"| {name} | {count:,} | {pct(values['accuracy'])} | {pct(values['cat_accuracy'])} | {pct(values['dog_accuracy'])} | {pct(values['balanced_accuracy'])} | {pct(values['macro_f1'])} |")
    lines += ["", "三者均使用 REP-006，但 DNN 最终训练未使用 CNN/RNN 的镜像增强，最终训练轮数与优化动态也不同；这些数字描述实际结果，不能把差距单独归因于网络结构。500 张留出图此前已用于项目 DNN 和 CNN 的最终评估，因此不是全项目意义上的首次未见数据；它没有参与 RNN 的架构、正则化、宽度或学习率选择。RNN 流程在本次评估前已冻结，结果出现后没有再调参或重训。", "",
              "逐轮训练见 [train_history.csv](train_history.csv)，单张预测见 [predictions.csv](predictions.csv)，机器指标见 [test_metrics.json](test_metrics.json)。", ""]
    (REPORT / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Archived {EXPERIMENT_ID} and DNN/CNN/RNN comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "evaluate", "audit", "archive"))
    command = parser.parse_args().command
    {"smoke": smoke, "train": train, "evaluate": evaluate,
     "audit": audit, "archive": archive}[command]()


if __name__ == "__main__":
    main()
