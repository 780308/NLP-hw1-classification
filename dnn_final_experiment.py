"""One frozen full-data DNN run and one held-out evaluation."""

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
from numpy.lib.format import open_memmap
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASS_TO_IDX, list_samples
from models.dnn_architecture import ArchitectureDNN, INPUT_DIM
from representations.cache import (
    HANDCRAFTED, ROOT, cache_config, current_commit, normalization,
    sha256_file, source_samples, stable_sha256, validate_cache,
)
from representations.dataset import CachedRepresentationDataset
from representations.extractors import GROUP_SLICES, extract_all
from representations.preprocess import preprocess_for_handcrafted
from training import plot_confusion, run_epoch, seed_everything, select_device, write_json

EXPERIMENT_ID = "DNN-FINAL-001"
ARCHITECTURE_ID = "DNN-ARCH-C"
REPRESENTATION_ID = "REP-006-FUSION"
EPOCHS = 3
SEED = 42
CHANNELS = tuple(range(190))
OUTPUT = ROOT / "outputs/dnn_final" / EXPERIMENT_ID
REPORT = ROOT / "report/dnn_final"
VAL_CACHE = OUTPUT / "cache_val"
TRAIN_MANIFEST_HASH = "97f9453652a511952fad677d596feb41e33d2c9c9f94024ec616c9916767ecd6"


def check_protocol() -> tuple[dict, list[tuple[Path, int]]]:
    train_manifest = validate_cache()
    if sha256_file(HANDCRAFTED / "manifest.json") != TRAIN_MANIFEST_HASH:
        raise AssertionError("Stage-I train cache manifest hash changed")
    stage1 = json.loads((ROOT / "report/stage1/representation_manifest.json").read_text(encoding="utf-8"))
    fusion = stage1["conditional_fusion"]
    if (stage1["selected_representation_id"] != REPRESENTATION_ID
            or fusion != {"id": REPRESENTATION_ID, "groups": list(GROUP_SLICES),
                          "shape": [190, 7, 7], "flatten_dim": INPUT_DIM}
            or stage1["handcrafted_cache_manifest_sha256"] != TRAIN_MANIFEST_HASH):
        raise AssertionError("Frozen REP-006 definition changed")
    decision = json.loads((ROOT / "report/dnn_architecture/architecture_decision.json").read_text(encoding="utf-8"))
    if decision["selected_architecture"] != ARCHITECTURE_ID:
        raise AssertionError("Frozen DNN architecture changed")
    samples = source_samples()
    if (train_manifest["feature_shape"] != [2000, 190, 7, 7]
            or train_manifest["channel_groups"] != {key: list(bounds) for key, bounds in GROUP_SLICES.items()}):
        raise AssertionError("Train cache geometry or channel ordering changed")
    # The final epoch decision is based solely on already recorded internal-validation curves.
    means = []
    for epoch in range(1, 7):
        accuracies = []
        for seed in (42, 123, 2026):
            path = ROOT / f"report/dnn_architecture/experiments/{ARCHITECTURE_ID}/seed{seed}/history.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            accuracies.append(float(history[epoch - 1]["validation_accuracy"]))
        means.append(sum(accuracies) / 3)
    if max(range(1, 7), key=lambda epoch: means[epoch - 1]) != EPOCHS:
        raise AssertionError("Recorded matched-epoch evidence no longer supports three epochs")
    return train_manifest, samples


def full_normalization() -> tuple[np.ndarray, np.ndarray]:
    mean, std = normalization(list(range(2000)), CHANNELS)
    if mean.shape != (190,) or std.shape != (190,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise AssertionError("Invalid full-training normalization")
    return mean, std


def smoke() -> None:
    check_protocol()
    mean, std = full_normalization()
    dataset = CachedRepresentationDataset(
        HANDCRAFTED / "features.npy", HANDCRAFTED / "labels.npy",
        list(range(8)), CHANNELS, mean, std,
    )
    features, labels = next(iter(DataLoader(dataset, batch_size=8)))
    model = ArchitectureDNN(ARCHITECTURE_ID)
    logits = model(features)
    if features.shape != (8, 190, 7, 7) or logits.shape != (8, 2):
        raise AssertionError("Final DNN tensor shape mismatch")
    nn.CrossEntropyLoss()(logits, labels).backward()
    print("Frozen train cache, normalization, model forward/backward and epoch evidence OK", flush=True)


def train() -> None:
    train_manifest, samples = check_protocol()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"Final run directory already contains artifacts: {OUTPUT}")
    mean, std = full_normalization()
    OUTPUT.mkdir(parents=True)
    seed_everything(SEED)
    device = select_device("auto")
    model = ArchitectureDNN(ARCHITECTURE_ID).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 2417538:
        raise AssertionError("DNN-ARCH-C parameter count changed")
    config = {
        "experiment_id": EXPERIMENT_ID, "architecture_id": ARCHITECTURE_ID,
        "representation_id": REPRESENTATION_ID, "representation_shape": [190, 7, 7],
        "flatten_dim": INPUT_DIM, "class_to_idx": CLASS_TO_IDX,
        "training_seed": SEED, "training_samples": len(samples),
        "training_class_counts": dict(Counter("cat" if label == 0 else "dog" for _, label in samples)),
        "epochs": EPOCHS, "early_stopping": False, "optimizer": "AdamW",
        "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 32,
        "loss": "CrossEntropyLoss", "augmentation": "none",
        "checkpoint_selection": "final state after exactly three full-data epochs",
        "parameter_count": parameter_count, "device": str(device),
        "train_cache_manifest_sha256": sha256_file(HANDCRAFTED / "manifest.json"),
        "code_commit": current_commit(),
    }
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "normalization.json", {
        "source": "all 2000 data/train samples only", "channels": 190,
        "mean": mean.tolist(), "std": std.tolist(),
    })
    dataset = CachedRepresentationDataset(
        HANDCRAFTED / "features.npy", HANDCRAFTED / "labels.npy",
        list(range(2000)), CHANNELS, mean, std,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0,
                        pin_memory=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        measured = run_epoch(model, loader, criterion, device, optimizer)
        history.append({
            "epoch": epoch, "train_loss": measured["loss"],
            "train_accuracy": measured["accuracy"],
            "cat_accuracy": measured["cat_accuracy"],
            "dog_accuracy": measured["dog_accuracy"],
        })
        print(f"{EXPERIMENT_ID} epoch {epoch}: {history[-1]}", flush=True)
    fit_seconds = time.perf_counter() - started
    checkpoint = {**config, "model_type": "dnn", "model_state": model.state_dict(),
                  "normalization": {"mean": mean.tolist(), "std": std.tolist()}}
    torch.save(checkpoint, OUTPUT / "final_model.pt")
    with (OUTPUT / "train_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(OUTPUT / "train_summary.json", {
        "experiment_id": EXPERIMENT_ID, "epochs_completed": len(history),
        "training_samples": len(dataset), "fit_seconds": fit_seconds,
        "final_epoch": history[-1], "checkpoint_sha256": sha256_file(OUTPUT / "final_model.pt"),
        "status": "trained_not_evaluated",
    })


def val_extractor_config(train_manifest: dict) -> dict:
    train_config = cache_config(source_samples())
    if train_manifest["config_sha256"] != stable_sha256(train_config):
        raise AssertionError("Stage-I extractor config changed")
    return {key: value for key, value in train_config.items() if key != "source_filenames"}


def build_val_cache() -> tuple[dict, list[str]]:
    train_manifest, _ = check_protocol()
    if not (OUTPUT / "final_model.pt").exists():
        raise FileNotFoundError("Train the frozen final model before extracting held-out images")
    samples = list_samples(ROOT / "data/val")
    if len(samples) != 500 or Counter(label for _, label in samples) != {0: 250, 1: 250}:
        raise AssertionError("Held-out sample count or class balance changed")
    names = [path.name for path, _ in samples]
    extractor_config = val_extractor_config(train_manifest)
    config_hash = stable_sha256(extractor_config)
    manifest_path = VAL_CACHE / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["feature_extraction_config_sha256"] != config_hash
                or manifest["filenames_sha256"] != stable_sha256(names)
                or manifest["train_cache_manifest_sha256"] != TRAIN_MANIFEST_HASH):
            raise AssertionError("Held-out cache manifest incompatible with frozen protocol")
    else:
        if VAL_CACHE.exists() and any(VAL_CACHE.iterdir()):
            raise RuntimeError("Incomplete held-out cache exists; inspect it before proceeding")
        VAL_CACHE.mkdir(parents=True)
        features = open_memmap(VAL_CACHE / "features.npy", mode="w+", dtype=np.float32, shape=(500, 190, 7, 7))
        started = time.perf_counter()
        for index, (path, _) in enumerate(samples):
            with Image.open(path) as image:
                features[index] = extract_all(preprocess_for_handcrafted(image))
            if (index + 1) % 100 == 0:
                print(f"Held-out features: {index + 1}/500", flush=True)
        features.flush()
        del features
        np.save(VAL_CACHE / "labels.npy", np.asarray([label for _, label in samples], dtype=np.uint8))
        write_json(VAL_CACHE / "filenames.json", names)
        manifest = {
            "experiment_id": EXPERIMENT_ID, "source_dataset": "data/val",
            "image_count": 500, "feature_shape": [500, 190, 7, 7],
            "feature_extraction_config_sha256": config_hash,
            "train_cache_config_sha256": train_manifest["config_sha256"],
            "train_cache_manifest_sha256": TRAIN_MANIFEST_HASH,
            "filenames_sha256": stable_sha256(names), "class_to_idx": CLASS_TO_IDX,
            "dependency_versions": extractor_config["dependency_versions"],
            "code_commit": current_commit(), "extraction_seconds": time.perf_counter() - started,
            "finite_values": True,
        }
        write_json(manifest_path, manifest)
    features = np.load(VAL_CACHE / "features.npy", mmap_mode="r")
    labels = np.load(VAL_CACHE / "labels.npy", mmap_mode="r")
    cached_names = json.loads((VAL_CACHE / "filenames.json").read_text(encoding="utf-8"))
    if (features.shape != (500, 190, 7, 7) or features.dtype != np.float32
            or list(labels) != [label for _, label in samples] or cached_names != names
            or not np.isfinite(features).all()):
        raise AssertionError("Held-out cache failed shape, label, order or finite-value audit")
    return manifest, names


def evaluate() -> None:
    check_protocol()
    if (OUTPUT / "test_metrics.json").exists() or (OUTPUT / "predictions.csv").exists():
        raise RuntimeError("Final held-out evaluation already exists; no second evaluation allowed")
    checkpoint = torch.load(OUTPUT / "final_model.pt", map_location="cpu", weights_only=True)
    config = json.loads((OUTPUT / "config.json").read_text(encoding="utf-8"))
    if any(checkpoint[key] != config[key] for key in config):
        raise AssertionError("Reloaded checkpoint metadata differs from frozen config")
    mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32)
    normal = json.loads((OUTPUT / "normalization.json").read_text(encoding="utf-8"))
    if not np.array_equal(mean, np.asarray(normal["mean"], dtype=np.float32)) or not np.array_equal(std, np.asarray(normal["std"], dtype=np.float32)):
        raise AssertionError("Checkpoint normalization differs from saved full-training statistics")
    manifest, names = build_val_cache()
    device = select_device("auto")
    model = ArchitectureDNN(ARCHITECTURE_ID).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = CachedRepresentationDataset(
        VAL_CACHE / "features.npy", VAL_CACHE / "labels.npy",
        list(range(500)), CHANNELS, mean, std,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    logits_all, labels_all = [], []
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device, non_blocking=True))
            loss_sum += criterion(logits, labels.to(device)).item() * len(labels)
            logits_all.append(logits.cpu())
            labels_all.append(labels)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()
    predictions = logits.argmax(dim=1).numpy()
    probabilities = logits.softmax(dim=1).numpy()
    matrix = [[int(((labels == actual) & (predictions == predicted)).sum())
               for predicted in (0, 1)] for actual in (0, 1)]
    cat_accuracy, dog_accuracy = matrix[0][0] / 250, matrix[1][1] / 250
    precision_cat = matrix[0][0] / (matrix[0][0] + matrix[1][0]) if matrix[0][0] + matrix[1][0] else 0.0
    precision_dog = matrix[1][1] / (matrix[0][1] + matrix[1][1]) if matrix[0][1] + matrix[1][1] else 0.0
    f1_cat = 2 * precision_cat * cat_accuracy / (precision_cat + cat_accuracy) if precision_cat + cat_accuracy else 0.0
    f1_dog = 2 * precision_dog * dog_accuracy / (precision_dog + dog_accuracy) if precision_dog + dog_accuracy else 0.0
    metrics = {
        "experiment_id": EXPERIMENT_ID, "test_dataset": "data/val",
        "checkpoint_sha256": sha256_file(OUTPUT / "final_model.pt"),
        "val_cache_manifest_sha256": sha256_file(VAL_CACHE / "manifest.json"),
        "train_cache_manifest_sha256": TRAIN_MANIFEST_HASH,
        "checkpoint_reloaded_for_evaluation": True,
        "samples": 500, "cat_samples": 250, "dog_samples": 250,
        "loss": loss_sum / 500, "accuracy": float((predictions == labels).mean()),
        "cat_accuracy": cat_accuracy, "dog_accuracy": dog_accuracy,
        "balanced_accuracy": (cat_accuracy + dog_accuracy) / 2,
        "macro_f1": (f1_cat + f1_dog) / 2,
        "confusion_matrix": matrix, "status": "complete",
    }
    write_json(OUTPUT / "test_metrics.json", metrics)
    with (OUTPUT / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "true_label", "predicted_label", "logit_cat", "logit_dog", "prob_cat", "prob_dog", "correct"))
        for index, name in enumerate(names):
            writer.writerow((name, int(labels[index]), int(predictions[index]),
                             float(logits[index, 0]), float(logits[index, 1]),
                             float(probabilities[index, 0]), float(probabilities[index, 1]),
                             int(labels[index] == predictions[index])))
    plot_confusion(matrix, OUTPUT / "confusion_matrix.png", "DNN-FINAL-001 held-out confusion matrix")
    print(f"Final held-out metrics: {json.dumps(metrics, ensure_ascii=False)}", flush=True)


def audit() -> None:
    check_protocol()
    config = json.loads((OUTPUT / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((OUTPUT / "train_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUTPUT / "test_metrics.json").read_text(encoding="utf-8"))
    normal = json.loads((OUTPUT / "normalization.json").read_text(encoding="utf-8"))
    manifest, names = build_val_cache()
    if (config["training_samples"] != 2000 or config["training_class_counts"] != {"cat": 1000, "dog": 1000}
            or config["epochs"] != EPOCHS or config["early_stopping"]
            or summary["epochs_completed"] != EPOCHS or normal["source"] != "all 2000 data/train samples only"
            or manifest["image_count"] != 500 or metrics["samples"] != 500
            or metrics["checkpoint_sha256"] != sha256_file(OUTPUT / "final_model.pt")):
        raise AssertionError("Final training or evaluation protocol mismatch")
    with (OUTPUT / "train_history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in history] != [1, 2, 3]:
        raise AssertionError("Final training history does not contain exactly three epochs")
    with (OUTPUT / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 500 or [row["filename"] for row in rows] != names:
        raise AssertionError("Held-out predictions are incomplete or out of order")
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        true, predicted = int(row["true_label"]), int(row["predicted_label"])
        if true not in (0, 1) or predicted not in (0, 1) or int(row["correct"]) != int(true == predicted):
            raise AssertionError("Invalid prediction row")
        if abs(float(row["prob_cat"]) + float(row["prob_dog"]) - 1) > 1e-6:
            raise AssertionError("Invalid softmax probabilities")
        matrix[true][predicted] += 1
    if (matrix != metrics["confusion_matrix"] or sum(matrix[0]) != 250 or sum(matrix[1]) != 250
            or abs((matrix[0][0] + matrix[1][1]) / 500 - metrics["accuracy"]) > 1e-12):
        raise AssertionError("Held-out predictions and aggregate metrics differ")
    print("Audited frozen full-data training, reloaded checkpoint evaluation and all 500 predictions", flush=True)


def archive() -> None:
    audit()
    REPORT.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "normalization.json", "train_history.csv", "train_summary.json", "test_metrics.json", "predictions.csv"):
        shutil.copy2(OUTPUT / name, REPORT / name)
    shutil.copy2(VAL_CACHE / "manifest.json", REPORT / "val_cache_manifest.json")
    shutil.copy2(OUTPUT / "confusion_matrix.png", ROOT / "report/figures/dnn_final_confusion_matrix.png")
    metrics = json.loads((REPORT / "test_metrics.json").read_text(encoding="utf-8"))
    training = json.loads((REPORT / "train_summary.json").read_text(encoding="utf-8"))
    cache = json.loads((REPORT / "val_cache_manifest.json").read_text(encoding="utf-8"))
    pct = lambda value: f"{value * 100:.2f}%"
    lines = [
        "# DNN-FINAL-001：最终 DNN 实验", "",
        "## 冻结流程", "",
        "原图 → 保留长宽比的 128×128 缩放与反射填充 → HOG+LBP+HSV+RootSIFT → `[190,7,7]` → 展平 9310 维 → DNN-ARCH-C → 猫/狗。网络为 9310→256→128→2，两个隐层均为 Linear→BatchNorm1d→ReLU→Dropout(0.3)。", "",
        "## 选择来源与训练", "",
        "历史 DNN-001 使用原始 RGB。Stage I 只依据 `data/train` 内部划分选定 REP-006-FUSION；DNN-ARCH-001 在相同开发协议下选定 C。最终轮数依据三个 C 架构运行在相同固定轮次上的平均内部验证准确率，在最终评估前由原提议的最佳轮次中位数 6 调整为 3。第 1–6 轮的均值依次为 74.67%、74.00%、75.67%、75.00%、74.17%、73.33%。", "",
        f"随机种子 42；全部 2000 张 `data/train`（猫/狗各 1000）参与训练，190 通道归一化只由这 2000 张图拟合。AdamW、学习率 1e-3、权重衰减 1e-4、批量 32、交叉熵、无增强、恰好 3 轮、无早停。训练耗时 {training['fit_seconds']:.2f} 秒。最终检查点在第 3 轮后保存。", "",
        "## 500 张图评估", "",
        "`data/val` 的每张图独立提取同一冻结特征，使用训练集统计归一化；没有用测试图拟合参数。评估时重载最终检查点，使用 `eval()`，无随机增强或阈值搜索。", "",
        "| 损失 | 总体准确率 | 猫准确率 | 狗准确率 | 平衡准确率 | Macro F1 |", "|---:|---:|---:|---:|---:|---:|",
        f"| {metrics['loss']:.4f} | {pct(metrics['accuracy'])} | {pct(metrics['cat_accuracy'])} | {pct(metrics['dog_accuracy'])} | {pct(metrics['balanced_accuracy'])} | {pct(metrics['macro_f1'])} |", "",
        f"混淆矩阵（真实行/预测列，猫、狗顺序）：`{metrics['confusion_matrix']}`。保留集特征提取耗时 {cache['extraction_seconds']:.2f} 秒。", "",
        "![最终 DNN 混淆矩阵](../figures/dnn_final_confusion_matrix.png)", "",
        "## 历史基线比较与限制", "",
        f"DNN-001：原始 64×64 RGB、12288→256→64→2，历史保留集准确率 62.60%。DNN-FINAL-001：REP-006-FUSION 与 DNN-ARCH-C，准确率 {pct(metrics['accuracy'])}，变化 {(metrics['accuracy'] - 0.626) * 100:+.2f} 个百分点。两次实验同时改变了表示与架构，不能把总差额归因于其中单一因素；内部消融见 Stage I 和 DNN-ARCH-001。", "",
        "这 500 张保留评估图此前曾用于历史 DNN-001 评估，但没有参与 Stage I 表示选择或 DNN 架构选择；本次配置在评估前冻结。因此它不是整个项目中从未查看过的测试集。", "",
        "配置、逐轮历史、指标与 500 行预测见本目录；大检查点与特征数组保存在忽略的 `outputs/dnn_final/DNN-FINAL-001/`。", "",
    ]
    (REPORT / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "evaluate", "audit", "archive"))
    command = parser.parse_args().command
    {"smoke": smoke, "train": train, "evaluate": evaluate, "audit": audit, "archive": archive}[command]()


if __name__ == "__main__":
    main()
