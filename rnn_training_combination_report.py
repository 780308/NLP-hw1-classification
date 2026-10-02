"""Aggregate and audit the one RNN-TRAIN-002 combination experiment."""

from __future__ import annotations

import csv
import json
import statistics

import numpy as np
from matplotlib import pyplot as plt
from torch import nn

import rnn_training_combination_experiments as exp
from cnn_training_strategy_experiments import validate_flip_cache
from models.rnn2d import ARCHITECTURES
from representations.cache import HANDCRAFTED, ROOT, sha256_file
from rnn2d_architecture_experiments import context
from rnn_training_strategy_experiments import model_for, strategy_settings, t1_normalization
from rnn_training_strategy_report import late_loss_rise, record as old_record, rows_and_groups
from training import write_json

FIELDS = (
    "experiment_id", "strategy_id", "seed", "parameter_count", "training_sample_count",
    "best_epoch", "final_epoch", "train_accuracy_at_best_epoch", "final_train_accuracy",
    "validation_loss", "minimum_validation_loss", "validation_accuracy",
    "final_validation_accuracy", "cat_accuracy", "dog_accuracy", "balanced_accuracy",
    "macro_f1", "late_validation_loss_rise", "fit_seconds", "learning_rate",
    "weight_decay", "augmentation", "gradient_clip_norm", "split_sha256",
    "feature_cache_manifest_sha256", "flipped_cache_manifest_sha256", "code_commit",
    "status", "source_run",
)


def record(seed: int) -> tuple[dict, list[dict], dict, dict]:
    directory = exp.REPORT / "experiments" / exp.STRATEGY_ID / f"seed{seed}"
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    normal = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
    with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if not history or metrics["seed"] != seed:
        raise AssertionError(f"Incomplete T4 record: {directory}")
    return metrics, history, config, normal


def aggregate() -> tuple[list[dict], dict]:
    rows = []
    for seed in exp.SEEDS:
        metrics, history, config, _ = record(seed)
        directory = exp.REPORT / "experiments" / exp.STRATEGY_ID / f"seed{seed}"
        rows.append({
            "experiment_id": metrics["experiment_id"], "strategy_id": metrics["strategy_id"],
            "seed": seed, "parameter_count": metrics["parameter_count"],
            "training_sample_count": metrics["training_sample_count"],
            "best_epoch": metrics["best_epoch"], "final_epoch": len(history),
            "train_accuracy_at_best_epoch": metrics["train_accuracy_at_best_epoch"],
            "final_train_accuracy": metrics["final_train_accuracy"],
            "validation_loss": metrics["validation_loss"],
            "minimum_validation_loss": min(float(epoch["validation_loss"]) for epoch in history),
            "validation_accuracy": metrics["validation_accuracy"],
            "final_validation_accuracy": float(history[-1]["validation_accuracy"]),
            **{key: metrics[key] for key in ("cat_accuracy", "dog_accuracy", "balanced_accuracy",
                                                   "macro_f1", "fit_seconds", "learning_rate", "weight_decay",
                                                   "augmentation", "gradient_clip_norm", "split_sha256",
                                                   "feature_cache_manifest_sha256",
                                                   "flipped_cache_manifest_sha256", "code_commit", "status")},
            "late_validation_loss_rise": late_loss_rise(history),
            "source_run": str(directory.relative_to(ROOT)).replace("\\", "/"),
        })
    accuracies = [row["validation_accuracy"] for row in rows]
    cat = statistics.mean(row["cat_accuracy"] for row in rows)
    dog = statistics.mean(row["dog_accuracy"] for row in rows)
    final_train = statistics.mean(row["final_train_accuracy"] for row in rows)
    final_validation = statistics.mean(row["final_validation_accuracy"] for row in rows)
    group = {
        "strategy_id": exp.STRATEGY_ID, "parameter_count": rows[0]["parameter_count"],
        "training_sample_count": rows[0]["training_sample_count"],
        "mean_validation_accuracy": statistics.mean(accuracies),
        "sample_std_validation_accuracy": statistics.stdev(accuracies),
        "worst_seed_accuracy": min(accuracies), "best_seed_accuracy": max(accuracies),
        "mean_cat_accuracy": cat, "mean_dog_accuracy": dog,
        "absolute_cat_dog_gap": abs(cat - dog),
        "mean_balanced_accuracy": statistics.mean(row["balanced_accuracy"] for row in rows),
        "mean_macro_f1": statistics.mean(row["macro_f1"] for row in rows),
        "median_best_epoch": int(statistics.median(row["best_epoch"] for row in rows)),
        "mean_final_train_accuracy": final_train,
        "mean_final_validation_accuracy": final_validation,
        "mean_train_validation_gap": final_train - final_validation,
        "late_validation_loss_rise_runs": sum(row["late_validation_loss_rise"] for row in rows),
        "mean_fit_seconds": statistics.mean(row["fit_seconds"] for row in rows),
        "seed_accuracies": {str(row["seed"]): row["validation_accuracy"] for row in rows},
    }
    return rows, group


def decide(t4: dict, t1: dict) -> dict:
    mean, worst, std = (t4["mean_validation_accuracy"], t4["worst_seed_accuracy"],
                        t4["sample_std_validation_accuracy"])
    worst_gain = worst - t1["worst_seed_accuracy"]
    std_reduction = t1["sample_std_validation_accuracy"] - std
    if mean >= 0.85 - 1e-12:
        selected, reason = exp.STRATEGY_ID, "accuracy_success_mean_at_least_85.0"
    elif mean >= 0.84 - 1e-12 and (worst >= 0.81 - 1e-12 or std <= 0.03 + 1e-12):
        selected, reason = exp.STRATEGY_ID, "stability_success"
    elif (0.8375 - 1e-12 <= mean < 0.84 - 1e-12 and worst_gain >= 0.015 - 1e-12
          and std_reduction >= 0.01 - 1e-12):
        selected, reason = exp.STRATEGY_ID, "close_mean_stability_success"
    elif mean < 0.835 - 1e-12 and worst_gain >= 0.025 - 1e-12 and std_reduction >= 0.015 - 1e-12:
        selected, reason = exp.STRATEGY_ID, "low_mean_exceptional_stability"
    else:
        selected, reason = "RNN-TRAIN-T1-FLIP", "combination_did_not_meet_predeclared_threshold"
    weight_decay = 5e-4 if selected == exp.STRATEGY_ID else 1e-4
    return {
        "selected_training_recipe": selected, "selection_rule": reason,
        "frozen_recipe": {"architecture_id": exp.ARCHITECTURE_ID,
                          "representation_id": "REP-006-FUSION", "horizontal_flip": True,
                          "weight_decay": weight_decay, "learning_rate": 3e-4,
                          "dropout_probability": 0.0},
        "next_stage": "RNN-ARCH-003 controlled width scaling",
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def plot_curves() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 7))
    colors = {42: "#1f77b4", 123: "#ff7f0e", 2026: "#2ca02c"}
    for row, strategy_id in enumerate(("RNN-TRAIN-T1-FLIP", exp.STRATEGY_ID)):
        for seed in exp.SEEDS:
            history = old_record(strategy_id, seed)[1] if row == 0 else record(seed)[1]
            epochs = [int(epoch["epoch"]) for epoch in history]
            for column, metric in enumerate(("accuracy", "loss")):
                axis = axes[row, column]
                axis.plot(epochs, [float(epoch[f"train_{metric}"]) for epoch in history],
                          color=colors[seed], label=f"{seed} train")
                axis.plot(epochs, [float(epoch[f"validation_{metric}"]) for epoch in history],
                          color=colors[seed], linestyle="--", label=f"{seed} validation")
        for axis in axes[row]:
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
        axes[row, 0].set_ylabel(f"{strategy_id}\nAccuracy")
        axes[row, 1].set_ylabel(f"{strategy_id}\nCross-entropy")
    figure.suptitle("RNN-TRAIN-002: flip-only vs flip plus weight decay")
    figure.tight_layout()
    figure.savefig(exp.REPORT / "training_curves.png", dpi=150)
    plt.close(figure)


def summarize() -> None:
    rows, t4 = aggregate()
    _, previous = rows_and_groups()
    references = {key: previous[key] for key in ("RNN-TRAIN-T0-BASE", "RNN-TRAIN-T1-FLIP", "RNN-TRAIN-T3-WD")}
    t1 = references["RNN-TRAIN-T1-FLIP"]
    changes = {
        "mean_accuracy_change": t4["mean_validation_accuracy"] - t1["mean_validation_accuracy"],
        "worst_seed_change": t4["worst_seed_accuracy"] - t1["worst_seed_accuracy"],
        "best_seed_change": t4["best_seed_accuracy"] - t1["best_seed_accuracy"],
        "std_change": t4["sample_std_validation_accuracy"] - t1["sample_std_validation_accuracy"],
        "cat_dog_gap_change": t4["absolute_cat_dog_gap"] - t1["absolute_cat_dog_gap"],
        "train_validation_gap_change": t4["mean_train_validation_gap"] - t1["mean_final_train_validation_gap"],
    }
    choice = decide(t4, t1)
    exp.REPORT.mkdir(parents=True, exist_ok=True)
    with (exp.REPORT / "combination_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(exp.REPORT / "combination_decision.json", {
        "batch_id": exp.BATCH_ID, "strategy_id": exp.STRATEGY_ID, "t4": t4,
        "historical_references": references, "relative_to_t1": changes, **choice,
    })
    plot_curves()
    lines = [
        "# RNN-TRAIN-002：镜像增强与更强权重衰减组合", "",
        "冻结 `REP-006-FUSION` `[190,7,7]`、`RNN2D-ARCH-L` 与种子 42/123/2026 的原 1800/200 划分。每份训练集使用 1800 原图和 1800 张 RGB 水平镜像后重提的特征，归一化仅由这 3600 张图拟合；验证始终为 200 张原图。T4 相对 T1 只将 AdamW 权重衰减由 `1e-4` 改为 `5e-4`，其余固定为学习率 `3e-4`、批量 32、最多 40 轮、早停耐心 8、梯度裁剪 1.0、无 Dropout 和调度器。没有使用 `data/val`。", "",
        "## 与既有策略比较", "", "标准差为三种子的样本标准差。T0/T1/T3 直接读取已归档记录，没有重训。", "",
        "| 策略 | 均值 ± 标准差 | 种子 42/123/2026 | 最差/最好 | 猫/狗均值 | Macro F1 | 末轮训练/验证差距 | 后期损失回升 | 平均训练秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in (*references.items(), (exp.STRATEGY_ID, t4)):
        gap = group.get("mean_train_validation_gap", group.get("mean_final_train_validation_gap"))
        seeds = "/".join(pct(group["seed_accuracies"][str(seed)]) for seed in exp.SEEDS)
        lines.append(f"| {name} | {pct(group['mean_validation_accuracy'])} ± {pct(group['sample_std_validation_accuracy'])} | {seeds} | {pct(group['worst_seed_accuracy'])}/{pct(group['best_seed_accuracy'])} | {pct(group['mean_cat_accuracy'])}/{pct(group['mean_dog_accuracy'])} | {pct(group['mean_macro_f1'])} | {100 * gap:.2f} pp | {group['late_validation_loss_rise_runs']}/3 | {group['mean_fit_seconds']:.2f} |")
    lines += ["", "## T4 逐种子轨迹诊断", "",
              "| 种子 | 最佳/末轮 | 最佳轮训练准确率 | 末轮训练准确率 | 最佳/末轮验证准确率 | 最佳准确率轮验证损失 | 最低验证损失 | 之后回升 ≥0.10 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['best_epoch']}/{row['final_epoch']} | {pct(row['train_accuracy_at_best_epoch'])} | {pct(row['final_train_accuracy'])} | {pct(row['validation_accuracy'])}/{pct(row['final_validation_accuracy'])} | {row['validation_loss']:.4f} | {row['minimum_validation_loss']:.4f} | {'是' if row['late_validation_loss_rise'] else '否'} |")
    lines += ["", "![T1 与 T4 训练曲线](training_curves.png)", "",
              "## 冻结决策", "",
              f"相对 T1：平均准确率变化 {100 * changes['mean_accuracy_change']:+.2f} pp，最差种子变化 {100 * changes['worst_seed_change']:+.2f} pp，最好种子变化 {100 * changes['best_seed_change']:+.2f} pp，样本标准差变化 {100 * changes['std_change']:+.2f} pp，猫狗准确率差变化 {100 * changes['cat_dog_gap_change']:+.2f} pp，末轮训练/验证差距变化 {100 * changes['train_validation_gap_change']:+.2f} pp。", "",
              f"按预设规则选择 **{choice['selected_training_recipe']}**（`{choice['selection_rule']}`）。冻结未来架构比较的配方为 RGB 水平镜像增强、学习率 `3e-4`、AdamW 权重衰减 `{choice['frozen_recipe']['weight_decay']}`、无 Dropout。下一阶段可另立 `RNN-ARCH-003` 受控宽度扩展；本任务没有启动。", "",
              "本阶段仅新增 T4 三个种子的训练；没有访问 `data/val`、运行 RNN 最终训练、额外策略或架构扩展。", ""]
    (exp.REPORT / "combination_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected {choice['selected_training_recipe']} ({choice['selection_rule']})", flush=True)


def audit() -> None:
    channels, prepared = context()
    validate_flip_cache()
    if channels != tuple(range(190)):
        raise AssertionError("REP-006 channel order changed")
    paths = list((exp.REPORT / "experiments").glob("**/metrics.json"))
    if len(paths) != 3:
        raise AssertionError("Expected exactly three T4 runs")
    _, t4 = aggregate()
    _, previous = rows_and_groups()
    model = model_for(exp.STRATEGY_ID)
    rnn_layers = [module for module in model.modules() if isinstance(module, nn.RNN)]
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if (ARCHITECTURES[exp.ARCHITECTURE_ID] != {"embed_dim": 320, "rnn_hidden_size": 80, "num_blocks": 3}
            or count != previous["RNN-TRAIN-T1-FLIP"]["parameter_count"]
            or len(rnn_layers) != 6 or any(layer.nonlinearity != "tanh" or not layer.bidirectional
                                           for layer in rnn_layers)
            or any(isinstance(module, (nn.Dropout, nn.GRU, nn.LSTM, nn.Conv2d))
                   for module in model.modules())
            or strategy_settings(exp.STRATEGY_ID) != {
                "augmentation": "horizontal_flip", "dropout_probability": 0.0, "weight_decay": 5e-4}):
        raise AssertionError("Frozen T4 model or strategy changed")
    for seed in exp.SEEDS:
        metrics, history, config, normal = record(seed)
        split = prepared[seed]["split"]
        expected_mean, expected_std = t1_normalization(prepared[seed]["train_indices"])
        t1_config = old_record("RNN-TRAIN-T1-FLIP", seed)[2]
        t1_normal = old_record("RNN-TRAIN-T1-FLIP", seed)[3]
        directory = exp.REPORT / "experiments" / exp.STRATEGY_ID / f"seed{seed}"
        if (config["batch_id"] != exp.BATCH_ID or metrics["strategy_id"] != exp.STRATEGY_ID
                or config["strategy_id"] != exp.STRATEGY_ID or metrics["status"] != "complete"
                or config["architecture_id"] != exp.ARCHITECTURE_ID
                or config["representation_id"] != "REP-006-FUSION"
                or config["representation_shape"] != [190, 7, 7]
                or config["parameter_count"] != count or metrics["parameter_count"] != count
                or config["training_sample_count"] != 3600 or metrics["training_sample_count"] != 3600
                or config["validation_samples"] != 200 or config["weight_decay"] != 5e-4
                or metrics["weight_decay"] != 5e-4 or config["learning_rate"] != 3e-4
                or metrics["learning_rate"] != 3e-4 or config["dropout_probability"] != 0.0
                or config["augmentation"] != "horizontal_flip" or metrics["augmentation"] != "horizontal_flip"
                or config["scheduler"] is not None or config["optimizer"] != "AdamW"
                or config["loss"] != "CrossEntropyLoss" or config["batch_size"] != 32
                or config["max_epochs"] != 40 or config["early_stopping_patience"] != 8
                or config["gradient_clip_norm"] != 1.0 or metrics["gradient_clip_norm"] != 1.0
                or config["split_sha256"] != split["sha256"] or metrics["split_sha256"] != split["sha256"]
                or normal["split_sha256"] != split["sha256"]
                or config["feature_cache_manifest_sha256"] != sha256_file(HANDCRAFTED / "manifest.json")
                or config["flipped_cache_manifest_sha256"] != sha256_file(exp.FLIP_CACHE / "manifest.json")
                or metrics["feature_cache_manifest_sha256"] != config["feature_cache_manifest_sha256"]
                or metrics["flipped_cache_manifest_sha256"] != config["flipped_cache_manifest_sha256"]
                or config["feature_cache_manifest_sha256"] != t1_config["feature_cache_manifest_sha256"]
                or config["flipped_cache_manifest_sha256"] != t1_config["flipped_cache_manifest_sha256"]
                or not np.array_equal(np.asarray(normal["mean"], dtype=np.float32), expected_mean)
                or not np.array_equal(np.asarray(normal["std"], dtype=np.float32), expected_std)
                or normal["mean"] != t1_normal["mean"] or normal["std"] != t1_normal["std"]):
            raise AssertionError(f"T4 protocol mismatch: {directory}")
        with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        if len(predictions) != 200 or [row["filename"] for row in predictions] != split["internal_validation"]:
            raise AssertionError(f"T4 validation rows differ from frozen split: {directory}")
        matrix = [[0, 0], [0, 0]]
        losses = []
        for row in predictions:
            actual, guess = int(row["true_label"]), int(row["predicted_label"])
            logits = np.asarray([float(row["logit_cat"]), float(row["logit_dog"])])
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            if (actual != (0 if row["filename"].startswith("cat.") else 1)
                    or guess != int(logits.argmax()) or int(row["correct"]) != int(actual == guess)
                    or abs(float(row["prob_cat"]) - probs[0]) > 1e-6
                    or abs(float(row["prob_dog"]) - probs[1]) > 1e-6):
                raise AssertionError(f"T4 prediction mismatch: {directory}")
            matrix[actual][guess] += 1
            losses.append(-np.log(probs[actual]))
        f1_cat = 2 * matrix[0][0] / (2 * matrix[0][0] + matrix[0][1] + matrix[1][0])
        f1_dog = 2 * matrix[1][1] / (2 * matrix[1][1] + matrix[0][1] + matrix[1][0])
        if (matrix != metrics["confusion_matrix"] or [sum(row) for row in matrix] != [100, 100]
                or abs((matrix[0][0] + matrix[1][1]) / 200 - metrics["validation_accuracy"]) > 1e-12
                or abs(float(history[metrics["best_epoch"] - 1]["validation_accuracy"]) -
                       metrics["validation_accuracy"]) > 1e-12
                or abs(float(np.mean(losses)) - metrics["validation_loss"]) > 1e-6
                or abs((f1_cat + f1_dog) / 2 - metrics["macro_f1"]) > 1e-12):
            raise AssertionError(f"T4 metrics mismatch: {directory}")
    decision = json.loads((exp.REPORT / "combination_decision.json").read_text(encoding="utf-8"))
    if (decision["t4"] != t4 or decision["selected_training_recipe"] !=
            decide(t4, previous["RNN-TRAIN-T1-FLIP"])["selected_training_recipe"]):
        raise AssertionError("T4 aggregate or frozen recipe decision changed")
    print("Audited three T4 runs, frozen flip normalization, 600 predictions, and recipe selection", flush=True)
