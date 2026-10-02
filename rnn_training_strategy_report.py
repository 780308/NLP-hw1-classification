"""Aggregate and audit RNN-TRAIN-001 using archived run artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from torch import nn

import rnn_training_strategy_experiments as exp
from cnn_training_strategy_experiments import validate_flip_cache
from models.rnn2d import ARCHITECTURES, VisionRNN2DClassifier
from representations.cache import HANDCRAFTED, ROOT, sha256_file
from training import write_json

FIELDS = (
    "strategy_id", "seed", "parameter_count", "training_sample_count", "best_epoch",
    "train_accuracy_at_best_epoch", "final_train_accuracy", "validation_loss",
    "validation_accuracy", "cat_accuracy", "dog_accuracy", "balanced_accuracy", "macro_f1",
    "fit_seconds", "learning_rate", "weight_decay", "dropout_probability", "augmentation",
    "gradient_clip_norm", "split_sha256", "feature_cache_manifest_sha256",
    "flipped_cache_manifest_sha256", "code_commit", "status", "source_run",
)


def record(strategy_id: str, seed: int) -> tuple[dict, list[dict], dict, dict, Path]:
    if strategy_id == exp.T0:
        directory = exp.BASE_REPORT / f"seed{seed}"
    else:
        directory = exp.REPORT / "experiments" / strategy_id / f"seed{seed}"
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    normal = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
    with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if not history or metrics["seed"] != seed:
        raise AssertionError(f"Incomplete strategy record: {directory}")
    return metrics, history, config, normal, directory


def late_loss_rise(history: list[dict]) -> bool:
    """Detect any >=0.10 rebound after an earlier validation-loss minimum."""
    lowest = float("inf")
    for epoch in history:
        loss = float(epoch["validation_loss"])
        if loss - lowest >= 0.10:
            return True
        lowest = min(lowest, loss)
    return False


def rows_and_groups() -> tuple[list[dict], dict]:
    rows, groups = [], {}
    for strategy_id in exp.ALL_STRATEGIES:
        subset, histories = [], []
        for seed in exp.SEEDS:
            metrics, history, config, _, directory = record(strategy_id, seed)
            row = {
                "strategy_id": strategy_id, "seed": seed,
                "parameter_count": metrics["parameter_count"],
                "training_sample_count": config.get("training_sample_count", config.get("train_samples")),
                "best_epoch": metrics["best_epoch"],
                "train_accuracy_at_best_epoch": metrics["train_accuracy_at_best_epoch"],
                "final_train_accuracy": metrics["final_train_accuracy"],
                "validation_loss": metrics["validation_loss"],
                **{key: metrics[key] for key in ("validation_accuracy", "cat_accuracy", "dog_accuracy",
                                                   "balanced_accuracy", "macro_f1", "fit_seconds",
                                                   "split_sha256", "feature_cache_manifest_sha256",
                                                   "code_commit", "status")},
                "learning_rate": config["learning_rate"], "weight_decay": config["weight_decay"],
                "dropout_probability": config.get("dropout_probability", 0.0),
                "augmentation": config["augmentation"],
                "gradient_clip_norm": config["gradient_clip_norm"],
                "flipped_cache_manifest_sha256": config.get("flipped_cache_manifest_sha256", ""),
                "source_run": str(directory.relative_to(ROOT)).replace("\\", "/"),
            }
            rows.append(row)
            subset.append(row)
            histories.append(history)
        accuracies = [row["validation_accuracy"] for row in subset]
        cat = statistics.mean(row["cat_accuracy"] for row in subset)
        dog = statistics.mean(row["dog_accuracy"] for row in subset)
        final_validation = statistics.mean(float(history[-1]["validation_accuracy"]) for history in histories)
        final_train = statistics.mean(row["final_train_accuracy"] for row in subset)
        groups[strategy_id] = {
            "strategy_id": strategy_id, "parameter_count": subset[0]["parameter_count"],
            "training_sample_count": subset[0]["training_sample_count"],
            "mean_validation_accuracy": statistics.mean(accuracies),
            "sample_std_validation_accuracy": statistics.stdev(accuracies),
            "worst_seed_accuracy": min(accuracies), "best_seed_accuracy": max(accuracies),
            "mean_cat_accuracy": cat, "mean_dog_accuracy": dog,
            "absolute_cat_dog_gap": abs(cat - dog),
            "mean_balanced_accuracy": statistics.mean(row["balanced_accuracy"] for row in subset),
            "mean_macro_f1": statistics.mean(row["macro_f1"] for row in subset),
            "median_best_epoch": int(statistics.median(row["best_epoch"] for row in subset)),
            "mean_final_train_accuracy": final_train,
            "mean_final_validation_accuracy": final_validation,
            "mean_final_train_validation_gap": final_train - final_validation,
            "late_validation_loss_rise_runs": sum(late_loss_rise(history) for history in histories),
            "mean_fit_seconds": statistics.mean(row["fit_seconds"] for row in subset),
            "seed_accuracies": {str(row["seed"]): row["validation_accuracy"] for row in subset},
        }
    baseline = groups[exp.T0]
    for group in groups.values():
        group["relative_to_t0"] = {
            "mean_accuracy_change": group["mean_validation_accuracy"] - baseline["mean_validation_accuracy"],
            "worst_seed_change": group["worst_seed_accuracy"] - baseline["worst_seed_accuracy"],
            "std_change": group["sample_std_validation_accuracy"] - baseline["sample_std_validation_accuracy"],
            "cat_dog_gap_change": group["absolute_cat_dog_gap"] - baseline["absolute_cat_dog_gap"],
        }
    return rows, groups


def improvement_band(group: dict) -> str:
    change = group["relative_to_t0"]
    mean, worst, std = change["mean_accuracy_change"], change["worst_seed_change"], change["std_change"]
    if mean >= 0.01 - 1e-12 or (mean >= 0.005 - 1e-12 and worst >= 0.015 - 1e-12):
        return "strong"
    if mean >= -0.0025 - 1e-12 and worst >= 0.02 - 1e-12 and std <= -0.01 + 1e-12:
        return "stability"
    return "weak_or_none"


def selection(groups: dict) -> dict:
    ranked = sorted(groups.values(), key=lambda group: (
        -group["mean_validation_accuracy"], -group["worst_seed_accuracy"],
        group["sample_std_validation_accuracy"], -group["mean_macro_f1"],
        group["absolute_cat_dog_gap"], group["strategy_id"]))
    top, second = ranked[:2]
    if top["mean_validation_accuracy"] - second["mean_validation_accuracy"] < 0.005 - 1e-12:
        higher_worst, other = sorted((top, second), key=lambda group: -group["worst_seed_accuracy"])
        if (higher_worst["sample_std_validation_accuracy"] - other["sample_std_validation_accuracy"]
                <= 0.01 + 1e-12):
            chosen, rule = higher_worst, "close_mean_higher_worst_seed"
        else:
            chosen, rule = other, "close_mean_worst_seed_std_exception"
    else:
        chosen, rule = top, "highest_mean_accuracy"
    bands = {strategy_id: improvement_band(groups[strategy_id]) for strategy_id in exp.STRATEGIES}
    eligible = sorted((strategy_id for strategy_id, band in bands.items()
                       if band in ("strong", "stability")),
                      key=lambda strategy_id: (-groups[strategy_id]["mean_validation_accuracy"],
                                               -groups[strategy_id]["worst_seed_accuracy"]))
    combination_eligible = len(eligible) >= 2
    baseline = groups[exp.T0]
    if (chosen["mean_validation_accuracy"] >= 0.825 - 1e-12 or
            (chosen["mean_validation_accuracy"] >= baseline["mean_validation_accuracy"] - 1e-12
             and chosen["worst_seed_accuracy"] >= 0.80 - 1e-12
             and chosen["sample_std_validation_accuracy"] <= 0.03 + 1e-12)):
        outcome, next_step = "A", "另立 RNN-ARCH-003 受控架构扩展；本阶段不启动。"
    elif (chosen["strategy_id"] != exp.T0 and
          (bands.get(chosen["strategy_id"]) in ("strong", "stability") or
           chosen["relative_to_t0"]["mean_accuracy_change"] >= 0.005 - 1e-12)):
        outcome = "B"
        next_step = ("可由用户决定是否运行一次组合实验；本阶段不组合。" if combination_eligible
                     else "冻结最佳单项训练策略，再决定是否扩展架构。")
    else:
        outcome, next_step = "C", "过拟合仍是主要瓶颈；重新考虑空间表示分辨率或更强架构归纳偏置。"
    return {
        "selected_strategy": chosen["strategy_id"], "selection_rule": rule,
        "ranked_by_mean_accuracy": [group["strategy_id"] for group in ranked],
        "improvement_bands": bands, "combination_eligible": combination_eligible,
        "combination_candidates": eligible[:2] if combination_eligible else [],
        "outcome": outcome, "recommended_next_step": next_step,
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def plot_curves() -> None:
    figure, axes = plt.subplots(4, 2, figsize=(12, 13))
    colors = {42: "#1f77b4", 123: "#ff7f0e", 2026: "#2ca02c"}
    for row, strategy_id in enumerate(exp.ALL_STRATEGIES):
        for seed in exp.SEEDS:
            _, history, _, _, _ = record(strategy_id, seed)
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
        axes[row, 0].set_ylabel(f"{strategy_id}\nAccuracy")
        axes[row, 1].set_ylabel(f"{strategy_id}\nCross-entropy")
    figure.suptitle("RNN-TRAIN-001: train and internal-validation trajectories")
    figure.tight_layout()
    figure.savefig(exp.REPORT / "training_curves.png", dpi=150)
    plt.close(figure)


def summarize() -> None:
    rows, groups = rows_and_groups()
    exp.REPORT.mkdir(parents=True, exist_ok=True)
    with (exp.REPORT / "training_strategy_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    decision = selection(groups)
    references = {}
    for path, keys in (
        (ROOT / "report/rnn_architecture/architecture_decision.json", ("RNN-ARCH-B", "RNN-ARCH-C")),
        (ROOT / "report/rnn2d_architecture/architecture_decision.json",
         ("RNN2D-ARCH-S", "RNN2D-ARCH-M", "RNN2D-ARCH-L")),
        (ROOT / "report/cnn_training/training_strategy_decision.json", ("CNN-TRAIN-T1-FLIP",)),
    ):
        models = json.loads(path.read_text(encoding="utf-8"))["models"]
        for key in keys:
            references[key] = models[key]["mean_validation_accuracy"]
    write_json(exp.REPORT / "training_strategy_decision.json", {
        "batch_id": exp.BATCH_ID, "baseline_source": str(exp.BASE_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "models": groups, **decision, "descriptive_internal_references": references,
    })
    plot_curves()
    lines = [
        "# RNN-TRAIN-001：双轴 RNN 的有限训练正则化比较", "",
        f"固定 `RNN2D-ARCH-L`（{groups[exp.T0]['parameter_count']:,} 参数）与 `REP-006-FUSION` `[190,7,7]`。沿用种子 42、123、2026 的三份 1800/200 内部划分，只用训练索引拟合归一化；没有访问 `data/val`。T0 直接导入 RNN-ARCH-002 三次记录，没有重训。", "",
        "T1 使用 RGB 水平翻转后重新提取的既有 REP-006 缓存，每份训练集为 1800 原图 + 1800 镜像图，归一化仅由这 3600 张图拟合；验证仍为 200 张原图。T2 只在每块的轴融合线性层后、MLP ReLU 后和 MLP 第二线性层后使用 `nn.Dropout(0.10)`。T3 只将 AdamW 权重衰减从 `1e-4` 提高至 `5e-4`。三者共用学习率 `3e-4`、批量 32、最多 40 轮、早停耐心 8、梯度裁剪 1.0；按最高内部验证准确率及同分较低损失保存检查点。", "",
        "## 三种子结果", "", "标准差为三种子的样本标准差；所有变化相对 T0，单位为百分点。", "",
        "| 策略 | 参数量 | 训练样本 | 平均准确率 ± 标准差 | 种子 42/123/2026 | 最差/最好 | 猫/狗均值 | Macro F1 | 最佳轮中位数 | 平均训练秒数 | 均值/最差/标准差变化 | 改进级别 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for strategy_id in exp.ALL_STRATEGIES:
        group = groups[strategy_id]
        change = group["relative_to_t0"]
        seed_text = "/".join(pct(group["seed_accuracies"][str(seed)]) for seed in exp.SEEDS)
        change_text = "/".join(f"{100 * change[key]:+.2f}" for key in
                               ("mean_accuracy_change", "worst_seed_change", "std_change"))
        band = decision["improvement_bands"].get(strategy_id, "baseline")
        lines.append(f"| {strategy_id} | {group['parameter_count']:,} | {group['training_sample_count']} | {pct(group['mean_validation_accuracy'])} ± {pct(group['sample_std_validation_accuracy'])} | {seed_text} | {pct(group['worst_seed_accuracy'])}/{pct(group['best_seed_accuracy'])} | {pct(group['mean_cat_accuracy'])}/{pct(group['mean_dog_accuracy'])} | {pct(group['mean_macro_f1'])} | {group['median_best_epoch']} | {group['mean_fit_seconds']:.2f} | {change_text} | {band} |")
    lines += ["", "## 过拟合诊断", "",
              "| 策略 | 末轮训练均值 | 末轮验证均值 | 平均差距 | 最低验证损失后升高 ≥0.10 的运行数 | 猫狗准确率差 |",
              "|---|---:|---:|---:|---:|---:|"]
    for strategy_id in exp.ALL_STRATEGIES:
        group = groups[strategy_id]
        lines.append(f"| {strategy_id} | {pct(group['mean_final_train_accuracy'])} | {pct(group['mean_final_validation_accuracy'])} | {100 * group['mean_final_train_validation_gap']:.2f} pp | {group['late_validation_loss_rise_runs']}/3 | {100 * group['absolute_cat_dog_gap']:.2f} pp |")
    baseline = groups[exp.T0]
    flip = groups[exp.STRATEGIES[0]]
    dropout = groups[exp.STRATEGIES[1]]
    decay = groups[exp.STRATEGIES[2]]
    lines += ["", "![训练与内部验证轨迹](training_curves.png)", "",
              f"T1 相对 T0 的末轮训练/验证差距缩小 {100 * (baseline['mean_final_train_validation_gap'] - flip['mean_final_train_validation_gap']):.2f} pp，猫狗准确率差缩小 {100 * (baseline['absolute_cat_dog_gap'] - flip['absolute_cat_dog_gap']):.2f} pp；其三种子标准差下降 {100 * (baseline['sample_std_validation_accuracy'] - flip['sample_std_validation_accuracy']):.2f} pp。T1 的训练样本数翻倍，收益应归为数据增强与有效样本扩充。", "",
              f"T2 的均值相对 T0 变化 {100 * dropout['relative_to_t0']['mean_accuracy_change']:+.2f} pp，虽然标准差下降，未达到预设改进门槛。T3 的均值变化 {100 * decay['relative_to_t0']['mean_accuracy_change']:+.2f} pp、最差种子改善 {100 * decay['relative_to_t0']['worst_seed_change']:+.2f} pp、标准差下降 {-100 * decay['relative_to_t0']['std_change']:.2f} pp，因此达到稳定性改进门槛。各策略至少有 {min(group['late_validation_loss_rise_runs'] for group in groups.values())}/3 次后期验证损失回升，过拟合尚未消除。", "",
              "## 选择和下一步", "",
              f"按冻结的均值优先、近差距重视最差种子的规则，选择 **{decision['selected_strategy']}**（`{decision['selection_rule']}`）。组合资格：**{str(decision['combination_eligible']).lower()}**；候选：{', '.join(decision['combination_candidates']) or '无'}。", "",
              f"结论为 **Outcome {decision['outcome']}**。{decision['recommended_next_step']}", "",
              "历史描述性参照（内部验证均值）：" + "、".join(f"{key} {pct(value)}" for key, value in references.items()) + "。不同阶段的训练协议可能不同，这些参照不参与本阶段选型。", "",
              "本阶段没有运行策略组合、RNN 最终训练、架构扩展、表示改造或 `data/val` 评估。", ""]
    (exp.REPORT / "training_strategy_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected {decision['selected_strategy']}; Outcome {decision['outcome']}; combination_eligible={decision['combination_eligible']}", flush=True)


def audit() -> None:
    channels, prepared = exp.context()
    validate_flip_cache()
    if channels != tuple(range(190)):
        raise AssertionError("REP-006 channel order changed")
    paths = list((exp.REPORT / "experiments").glob("**/metrics.json"))
    if len(paths) != 9:
        raise AssertionError("Expected exactly nine new formal runs")
    baseline_count = json.loads((exp.BASE_REPORT / "seed42/metrics.json").read_text(encoding="utf-8"))["parameter_count"]
    for strategy_id in exp.STRATEGIES:
        settings = exp.strategy_settings(strategy_id)
        model = exp.model_for(strategy_id)
        spec = ARCHITECTURES[exp.ARCHITECTURE_ID]
        recurrent = [layer for layer in model.modules() if isinstance(layer, nn.RNN)]
        if (spec != {"embed_dim": 320, "rnn_hidden_size": 80, "num_blocks": 3}
                or len(model.blocks) != 3 or len(recurrent) != 6
                or any(layer.nonlinearity != "tanh" or not layer.bidirectional or layer.dropout != 0
                       for layer in recurrent)
                or any(isinstance(layer, (nn.Conv2d, nn.LSTM, nn.GRU)) for layer in model.modules())
                or sum(p.numel() for p in model.parameters() if p.requires_grad) != baseline_count):
            raise AssertionError("Frozen RNN architecture changed")
        drops = [layer for layer in model.modules() if isinstance(layer, nn.Dropout)]
        if (len(drops) != (9 if strategy_id == exp.STRATEGIES[1] else 0)
                or any(layer.p != 0.10 for layer in drops)):
            raise AssertionError("Dropout topology changed")
        for block in model.blocks:
            if strategy_id == exp.STRATEGIES[1]:
                if (not isinstance(block.fusion_dropout, nn.Dropout)
                        or [type(layer) for layer in block.channel_mlp] !=
                           [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.Dropout]):
                    raise AssertionError("T2 dropout locations changed")
            elif (not isinstance(block.fusion_dropout, nn.Identity)
                  or [type(layer) for layer in block.channel_mlp] !=
                     [nn.Linear, nn.ReLU, nn.Linear]):
                raise AssertionError("Non-T2 residual path changed")
        for seed in exp.SEEDS:
            metrics, history, config, normal, directory = record(strategy_id, seed)
            split = prepared[seed]["split"]
            expected_mean, expected_std = (exp.t1_normalization(prepared[seed]["train_indices"])
                                           if strategy_id == exp.STRATEGIES[0]
                                           else (prepared[seed]["mean"], prepared[seed]["std"]))
            if (metrics["status"] != "complete" or metrics["strategy_id"] != strategy_id
                    or config["architecture_id"] != exp.ARCHITECTURE_ID
                    or config["representation_id"] != exp.REPRESENTATION_ID
                    or config["representation_shape"] != [190, 7, 7]
                    or any(config[key] != value for key, value in spec.items())
                    or any(config[key] != value or metrics[key] != value for key, value in settings.items())
                    or metrics["parameter_count"] != baseline_count or config["parameter_count"] != baseline_count
                    or config["training_sample_count"] != (3600 if strategy_id == exp.STRATEGIES[0] else 1800)
                    or metrics["training_sample_count"] != config["training_sample_count"]
                    or config["validation_samples"] != 200 or config["learning_rate"] != 3e-4
                    or metrics["learning_rate"] != 3e-4 or config["gradient_clip_norm"] != 1.0
                    or metrics["gradient_clip_norm"] != 1.0 or config["batch_size"] != 32
                    or config["max_epochs"] != 40 or config["early_stopping_patience"] != 8
                    or config["optimizer"] != "AdamW" or config["loss"] != "CrossEntropyLoss"
                    or config["scheduler"] is not None or config["split_sha256"] != split["sha256"]
                    or metrics["split_sha256"] != split["sha256"]
                    or normal["split_sha256"] != split["sha256"]
                    or config["feature_cache_manifest_sha256"] != sha256_file(HANDCRAFTED / "manifest.json")
                    or metrics["feature_cache_manifest_sha256"] != config["feature_cache_manifest_sha256"]
                    or config["flipped_cache_manifest_sha256"] !=
                       (sha256_file(exp.FLIP_CACHE / "manifest.json") if strategy_id == exp.STRATEGIES[0] else "")
                    or metrics["flipped_cache_manifest_sha256"] != config["flipped_cache_manifest_sha256"]
                    or not np.array_equal(np.asarray(normal["mean"], dtype=np.float32), expected_mean)
                    or not np.array_equal(np.asarray(normal["std"], dtype=np.float32), expected_std)):
                raise AssertionError(f"Strategy protocol mismatch: {directory}")
            with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                predictions = list(csv.DictReader(stream))
            matrix = [[0, 0], [0, 0]]
            row_losses = []
            if len(predictions) != 200 or [row["filename"] for row in predictions] != split["internal_validation"]:
                raise AssertionError(f"Prediction rows or split changed: {directory}")
            for row in predictions:
                actual, guess = int(row["true_label"]), int(row["predicted_label"])
                logits = np.asarray([float(row["logit_cat"]), float(row["logit_dog"])])
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                if (actual != (0 if row["filename"].startswith("cat.") else 1)
                        or guess != int(logits.argmax()) or int(row["correct"]) != int(actual == guess)
                        or abs(float(row["prob_cat"]) - probabilities[0]) > 1e-6
                        or abs(float(row["prob_dog"]) - probabilities[1]) > 1e-6):
                    raise AssertionError(f"Prediction details mismatch: {directory}")
                matrix[actual][guess] += 1
                row_losses.append(-np.log(probabilities[actual]))
            f1_cat = 2 * matrix[0][0] / (2 * matrix[0][0] + matrix[0][1] + matrix[1][0])
            f1_dog = 2 * matrix[1][1] / (2 * matrix[1][1] + matrix[0][1] + matrix[1][0])
            if (matrix != metrics["confusion_matrix"] or [sum(row) for row in matrix] != [100, 100]
                    or abs((matrix[0][0] + matrix[1][1]) / 200 - metrics["validation_accuracy"]) > 1e-12
                    or abs(float(history[metrics["best_epoch"] - 1]["validation_accuracy"]) -
                           metrics["validation_accuracy"]) > 1e-12
                    or abs(float(np.mean(row_losses)) - metrics["validation_loss"]) > 1e-6
                    or abs((f1_cat + f1_dog) / 2 - metrics["macro_f1"]) > 1e-12):
                raise AssertionError(f"Strategy metrics differ from predictions: {directory}")
    rows, groups = rows_and_groups()
    decision = json.loads((exp.REPORT / "training_strategy_decision.json").read_text(encoding="utf-8"))
    if len(rows) != 12 or decision["models"] != groups or any(decision[key] != value for key, value in selection(groups).items()):
        raise AssertionError("Strategy aggregate decision mismatch")
    print("Audited T0 import, nine new RNN runs, frozen protocols, and 1,800 new predictions", flush=True)
