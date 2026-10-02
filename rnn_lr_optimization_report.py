"""Aggregate, select, and audit the frozen RNN-TRAIN-003 learning-rate study."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

import rnn_lr_optimization_experiments as exp
from cnn_training_strategy_experiments import validate_flip_cache
from models.rnn2d import ARCHITECTURES, VisionRNN2DClassifier
from representations.cache import HANDCRAFTED, ROOT, sha256_file
from training import write_json

RESULT_FIELDS = (
    "experiment_id", "strategy_id", "seed", "parameter_count", "training_sample_count",
    "initial_learning_rate", "final_learning_rate", "scheduler_type", "weight_decay",
    "batch_size", "gradient_clip_norm", "best_epoch", "final_epoch",
    "train_accuracy_at_best_epoch", "final_train_accuracy", "validation_loss",
    "validation_accuracy", "cat_accuracy", "dog_accuracy", "balanced_accuracy", "macro_f1",
    "fit_seconds", "split_sha256", "feature_cache_manifest_sha256",
    "flipped_cache_manifest_sha256", "code_commit", "status", "source_run",
)


def record(strategy_id: str, seed: int) -> tuple[dict, list[dict], dict, dict, Path]:
    directory = (exp.BASE_REPORT if strategy_id == exp.T0 else
                 exp.REPORT / "experiments" / strategy_id) / f"seed{seed}"
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    normal = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
    with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if not history:
        raise AssertionError(f"Empty history: {directory}")
    return metrics, history, config, normal, directory


def loss_diagnostics(history: list[dict]) -> dict:
    losses = [float(item["validation_loss"]) for item in history]
    accuracies = [float(item["validation_accuracy"]) for item in history]
    minimum_index = min(range(len(losses)), key=losses.__getitem__)
    minimum = losses[minimum_index]
    rise = max(losses[minimum_index:]) - minimum
    return {
        "minimum_validation_loss_epoch": minimum_index + 1,
        "maximum_accuracy_first_epoch": accuracies.index(max(accuracies)) + 1,
        "validation_loss_rise_after_minimum": rise,
        "validation_loss_rise_ge_0p10": rise >= 0.10 - 1e-12,
    }


def rows_and_groups() -> tuple[list[dict], dict, dict]:
    rows, groups, diagnostics = [], {}, {}
    for strategy_id in exp.ALL_STRATEGIES:
        subset, histories, per_seed = [], [], {}
        for seed in exp.SEEDS:
            metrics, history, config, _, directory = record(strategy_id, seed)
            initial_lr = config["learning_rate"]
            scheduler = config.get("scheduler")
            row = {
                "experiment_id": metrics["experiment_id"], "strategy_id": strategy_id,
                "seed": seed, "parameter_count": metrics["parameter_count"],
                "training_sample_count": metrics["training_sample_count"],
                "initial_learning_rate": initial_lr,
                "final_learning_rate": metrics.get("final_learning_rate", initial_lr),
                "scheduler_type": scheduler["type"] if isinstance(scheduler, dict) else "none",
                "weight_decay": config["weight_decay"], "batch_size": config["batch_size"],
                "gradient_clip_norm": config["gradient_clip_norm"],
                "best_epoch": metrics["best_epoch"],
                "final_epoch": metrics.get("final_epoch", metrics["epochs_run"]),
                "train_accuracy_at_best_epoch": metrics["train_accuracy_at_best_epoch"],
                "final_train_accuracy": metrics["final_train_accuracy"],
                **{key: metrics[key] for key in ("validation_loss", "validation_accuracy",
                                                   "cat_accuracy", "dog_accuracy", "balanced_accuracy",
                                                   "macro_f1", "fit_seconds", "split_sha256",
                                                   "feature_cache_manifest_sha256",
                                                   "flipped_cache_manifest_sha256", "code_commit", "status")},
                "source_run": str(directory.relative_to(ROOT)).replace("\\", "/"),
            }
            rows.append(row)
            subset.append(row)
            histories.append(history)
            per_seed[str(seed)] = {
                **loss_diagnostics(history),
                "best_epoch": row["best_epoch"], "final_epoch": row["final_epoch"],
                "final_train_accuracy": row["final_train_accuracy"],
                "final_validation_accuracy": float(history[-1]["validation_accuracy"]),
                "final_train_validation_gap": row["final_train_accuracy"] -
                                              float(history[-1]["validation_accuracy"]),
                "learning_rate_at_best_epoch": float(history[row["best_epoch"] - 1]["learning_rate"]),
                "final_learning_rate": row["final_learning_rate"],
                "number_of_lr_reductions": metrics.get("number_of_lr_reductions"),
                "epochs_of_lr_reductions": metrics.get("epochs_of_lr_reductions", []),
                "learning_rates_after_reduction": metrics.get("learning_rates_after_reduction", []),
            }
            if strategy_id == exp.T3:
                first = per_seed[str(seed)]["epochs_of_lr_reductions"]
                per_seed[str(seed)]["best_accuracy_relative_to_first_reduction"] = (
                    "none" if not first else "before" if row["best_epoch"] <= first[0] else "after")
        accuracies = [row["validation_accuracy"] for row in subset]
        cat = statistics.mean(row["cat_accuracy"] for row in subset)
        dog = statistics.mean(row["dog_accuracy"] for row in subset)
        final_train = statistics.mean(row["final_train_accuracy"] for row in subset)
        final_validation = statistics.mean(float(history[-1]["validation_accuracy"]) for history in histories)
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
            "median_best_epoch": statistics.median(row["best_epoch"] for row in subset),
            "mean_final_train_accuracy": final_train,
            "mean_final_validation_accuracy": final_validation,
            "mean_train_validation_gap": final_train - final_validation,
            "mean_fit_seconds": statistics.mean(row["fit_seconds"] for row in subset),
            "seed_accuracies": {str(row["seed"]): row["validation_accuracy"] for row in subset},
            "late_validation_loss_rise_runs": sum(
                per_seed[str(seed)]["validation_loss_rise_ge_0p10"] for seed in exp.SEEDS),
        }
        diagnostics[strategy_id] = per_seed
    baseline = groups[exp.T0]
    for group in groups.values():
        group["relative_to_t0"] = {
            "mean_accuracy_change": group["mean_validation_accuracy"] - baseline["mean_validation_accuracy"],
            "worst_seed_change": group["worst_seed_accuracy"] - baseline["worst_seed_accuracy"],
            "best_seed_change": group["best_seed_accuracy"] - baseline["best_seed_accuracy"],
            "std_change": group["sample_std_validation_accuracy"] - baseline["sample_std_validation_accuracy"],
            "cat_dog_gap_change": group["absolute_cat_dog_gap"] - baseline["absolute_cat_dog_gap"],
            "train_validation_gap_change": group["mean_train_validation_gap"] -
                                           baseline["mean_train_validation_gap"],
        }
    return rows, groups, diagnostics


def qualifies(group: dict) -> list[str]:
    delta = group["relative_to_t0"]
    reasons = []
    if delta["mean_accuracy_change"] >= .0075 - 1e-12 and delta["worst_seed_change"] >= -.005 - 1e-12:
        reasons.append("clear_accuracy")
    if delta["mean_accuracy_change"] >= .005 - 1e-12 and delta["worst_seed_change"] >= .01 - 1e-12:
        reasons.append("balanced")
    if (group["mean_validation_accuracy"] >= .8375 - 1e-12
            and group["worst_seed_accuracy"] >= .82 - 1e-12
            and group["sample_std_validation_accuracy"] <= .025 + 1e-12):
        reasons.append("stability")
    return reasons


def select(groups: dict) -> dict:
    reasons = {strategy: qualifies(groups[strategy]) for strategy in exp.STRATEGIES}
    eligible = [strategy for strategy in exp.STRATEGIES if reasons[strategy]]
    if not eligible:
        selected, rule = exp.T0, "no_new_strategy_qualified"
    else:
        eligible.sort(key=lambda strategy: -groups[strategy]["mean_validation_accuracy"])
        top_mean = groups[eligible[0]]["mean_validation_accuracy"]
        close = [strategy for strategy in eligible
                 if top_mean - groups[strategy]["mean_validation_accuracy"] < .005 - 1e-12]
        if len(close) > 1:
            complexity = {exp.T1: 0, exp.T2: 1, exp.T3: 2}
            close.sort(key=lambda strategy: (-groups[strategy]["worst_seed_accuracy"],
                                             groups[strategy]["sample_std_validation_accuracy"],
                                             complexity[strategy]))
            selected, rule = close[0], "close_means_worst_seed_then_std_then_simplicity"
        else:
            selected, rule = eligible[0], "highest_qualifying_mean_accuracy"
    return {"selected_strategy": selected, "selection_rule": rule,
            "qualifying_reasons": reasons, "eligible_strategies": eligible,
            "meaningful_lr_policy_improvement": selected != exp.T0,
            "recommended_next_step": "Derive a fixed final epoch from existing three-seed curves, then separately run RNN-FINAL-001."}


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def matched_epochs(strategy_id: str) -> list[dict]:
    histories = {seed: record(strategy_id, seed)[1] for seed in exp.SEEDS}
    common = min(len(history) for history in histories.values())
    rows = []
    for index in range(common):
        at_epoch = [histories[seed][index] for seed in exp.SEEDS]
        accuracy = [float(item["validation_accuracy"]) for item in at_epoch]
        rows.append({
            "epoch": index + 1, "mean_validation_accuracy": statistics.mean(accuracy),
            "sample_std_validation_accuracy": statistics.stdev(accuracy),
            "mean_validation_loss": statistics.mean(float(item["validation_loss"]) for item in at_epoch),
            "mean_cat_accuracy": statistics.mean(float(item["validation_cat_accuracy"]) for item in at_epoch),
            "mean_dog_accuracy": statistics.mean(float(item["validation_dog_accuracy"]) for item in at_epoch),
        })
    return rows


def plot_curves() -> None:
    figure, axes = plt.subplots(4, 2, figsize=(12, 13))
    colors = {42: "#1f77b4", 123: "#ff7f0e", 2026: "#2ca02c"}
    for row, strategy in enumerate(exp.ALL_STRATEGIES):
        for seed in exp.SEEDS:
            history = record(strategy, seed)[1]
            epochs = [int(item["epoch"]) for item in history]
            for col, metric in enumerate(("accuracy", "loss")):
                axis = axes[row, col]
                axis.plot(epochs, [float(item[f"train_{metric}"]) for item in history],
                          color=colors[seed], label=f"{seed} train")
                axis.plot(epochs, [float(item[f"validation_{metric}"]) for item in history],
                          color=colors[seed], linestyle="--", label=f"{seed} validation")
        for axis in axes[row]:
            axis.set_xlabel("Epoch")
            axis.grid(alpha=.25)
            axis.legend(fontsize=7, ncol=2)
        axes[row, 0].set_ylabel(f"{strategy}\nAccuracy")
        axes[row, 1].set_ylabel(f"{strategy}\nCross-entropy")
    figure.suptitle("RNN-TRAIN-003: train and internal-validation trajectories")
    figure.tight_layout()
    figure.savefig(exp.REPORT / "training_curves.png", dpi=150)
    plt.close(figure)


def summarize() -> None:
    rows, groups, diagnostics = rows_and_groups()
    decision = select(groups)
    exp.REPORT.mkdir(parents=True, exist_ok=True)
    with (exp.REPORT / "lr_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    matched = matched_epochs(decision["selected_strategy"])
    with (exp.REPORT / "selected_policy_matched_epochs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matched[0]))
        writer.writeheader()
        writer.writerows(matched)
    write_json(exp.REPORT / "lr_decision.json", {
        "batch_id": exp.BATCH_ID, "baseline_source": str(exp.BASE_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "models": groups, "diagnostics": diagnostics, **decision,
        "matched_epoch_source": "selected_policy_matched_epochs.csv",
    })
    plot_curves()
    lines = ["# RNN-TRAIN-003：冻结双轴 RNN 学习率策略实验", "",
             "仅用原训练集的三份 1800/200 内部划分；`data/val` 未用于本阶段。架构固定为 `RNN2D-ARCH-BASE`（代码标识 `RNN2D-ARCH-L`），1,992,642 参数；REP-006 `[190,7,7]`，原图和 RGB 水平镜像特征共 3600 张参与训练归一化。T0 从既有 Flip-only 记录导入，未重训。", "",
             "## 三种子结果", "",
             "百分比标准差为样本标准差；变化均相对 T0，单位为百分点。", "",
             "| 策略 | 均值 ± 标准差 | 42 / 123 / 2026 | 最差 / 最好 | 猫 / 狗均值 | Macro F1 | 最佳轮中位数 | 末轮训练/验证差 | 晚期损失反弹次数 | 均值/最差/标准差变化 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for strategy in exp.ALL_STRATEGIES:
        group = groups[strategy]
        delta = group["relative_to_t0"]
        seeds = " / ".join(pct(group["seed_accuracies"][str(seed)]) for seed in exp.SEEDS)
        change = "/".join(f"{100 * delta[key]:+.2f}" for key in
                          ("mean_accuracy_change", "worst_seed_change", "std_change"))
        lines.append(f"| {strategy} | {pct(group['mean_validation_accuracy'])} ± {pct(group['sample_std_validation_accuracy'])} | {seeds} | {pct(group['worst_seed_accuracy'])} / {pct(group['best_seed_accuracy'])} | {pct(group['mean_cat_accuracy'])} / {pct(group['mean_dog_accuracy'])} | {pct(group['mean_macro_f1'])} | {group['median_best_epoch']:g} | {100 * group['mean_train_validation_gap']:.2f} pp | {group['late_validation_loss_rise_runs']}/3 | {change} |")
    lines += ["", "## 收敛与调度器诊断", "",
              "下表中的损失反弹为最低验证损失之后的最大增加；最佳准确率轮次沿用检查点选择规则。", "",
              "| 策略 | 种子 | 最佳/末轮 | 最低损失轮 | 最佳轮学习率 | 末轮学习率 | 损失反弹 | 降率轮次 | 最佳准确率相对首次降率 |",
              "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for strategy in exp.ALL_STRATEGIES:
        for seed in exp.SEEDS:
            item = diagnostics[strategy][str(seed)]
            reductions = ", ".join(str(epoch) for epoch in item["epochs_of_lr_reductions"]) or "—"
            relative = {"before": "降率前", "after": "降率后", "none": "无降率"}.get(
                item.get("best_accuracy_relative_to_first_reduction"), "—")
            lines.append(f"| {strategy} | {seed} | {item['best_epoch']}/{item['final_epoch']} | {item['minimum_validation_loss_epoch']} | {item['learning_rate_at_best_epoch']:.8g} | {item['final_learning_rate']:.8g} | {item['validation_loss_rise_after_minimum']:.3f} | {reductions} | {relative} |")
    lines += ["", "T2 在每轮训练完成后步进余弦调度器；表中最佳轮学习率是该轮实际用于训练的值。T3 在每次验证后按验证损失步进；降率轮次表示该轮验证之后改变了下一轮学习率。", "",
              f"与 T0 相比，T1/T2/T3 的晚期损失反弹次数分别为 {groups[exp.T1]['late_validation_loss_rise_runs']}/3、{groups[exp.T2]['late_validation_loss_rise_runs']}/3、{groups[exp.T3]['late_validation_loss_rise_runs']}/3；T0 为 {groups[exp.T0]['late_validation_loss_rise_runs']}/3。反弹次数只描述是否越过 0.10，不代表整体泛化质量。", ""]
    for seed in exp.SEEDS:
        t2_metrics = record(exp.T2, seed)[0]
        t3 = diagnostics[exp.T3][str(seed)]
        reductions = t3["epochs_of_lr_reductions"]
        timing = {"before": "之前", "after": "之后", "none": "（无降率）"}[
            t3["best_accuracy_relative_to_first_reduction"]]
        lines.append(f"- 种子 {seed}：T2 最低实际训练学习率 {t2_metrics['minimum_learning_rate_reached']:.8g}，"
                     f"{'达到' if t2_metrics['minimum_learning_rate_reached'] <= 1e-5 + 1e-12 else '未达到'} eta_min；"
                     f"T3 降率 {len(reductions)} 次，首次降率轮次 {reductions[0] if reductions else '无'}，"
                     f"最佳准确率在首次降率{timing}。")
    lines += ["", "三个新策略均为 3/3 次达到至少 0.10 的晚期验证损失反弹；T1/T2 的反弹幅度没有系统减小，T3 仅在种子 42 上略低于 T0，其他两种子更高。T2 的三个最佳轮都已使用低于初始值的学习率，因此衰减影响了训练，但未改善均值；T3 的三个最佳轮都发生在首次降率后，但均值仍低于 T0。", "",
              "![训练与内部验证曲线](training_curves.png)", "", "## 选择与后续", ""]
    selected = decision["selected_strategy"]
    delta = groups[selected]["relative_to_t0"]
    reasons = ", ".join(decision["qualifying_reasons"].get(selected, [])) or "无新策略达标"
    lines += [f"预设门槛与并列规则选定 **{selected}**（`{decision['selection_rule']}`；{reasons}）。相对 T0：均值 {100 * delta['mean_accuracy_change']:+.2f} pp，最差种子 {100 * delta['worst_seed_change']:+.2f} pp，标准差 {100 * delta['std_change']:+.2f} pp。", "",
              "选定策略的三种子共同轮次诊断见 [selected_policy_matched_epochs.csv](selected_policy_matched_epochs.csv)。该表逐轮列出平均验证准确率、样本标准差、平均验证损失及猫/狗准确率；本阶段不从中直接指定最终轮次。下一阶段应据既有曲线确定固定最终轮次，再另立 `RNN-FINAL-001`。", "",
              "本阶段未做最终全数据训练，也未产生 RNN 留出测试集结果。", ""]
    lines += ["### 选定策略的共同轮次诊断", "",
              "| 轮次 | 平均验证准确率 | 样本标准差 | 平均验证损失 | 猫准确率 | 狗准确率 |",
              "|---:|---:|---:|---:|---:|---:|"]
    for item in matched:
        lines.append(f"| {item['epoch']} | {pct(item['mean_validation_accuracy'])} | "
                     f"{pct(item['sample_std_validation_accuracy'])} | {item['mean_validation_loss']:.4f} | "
                     f"{pct(item['mean_cat_accuracy'])} | {pct(item['mean_dog_accuracy'])} |")
    lines.append("")
    (exp.REPORT / "lr_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected {selected}; qualifying={decision['qualifying_reasons']}")


def audit() -> None:
    channels, prepared = exp.context()
    validate_flip_cache()
    if channels != tuple(range(190)) or ARCHITECTURES[exp.ARCHITECTURE_ID] != {
            "embed_dim": 320, "rnn_hidden_size": 80, "num_blocks": 3}:
        raise AssertionError("Frozen architecture or channels changed")
    model = VisionRNN2DClassifier(exp.ARCHITECTURE_ID)
    recurrent = [layer for layer in model.modules() if isinstance(layer, nn.RNN)]
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if (count != 1_992_642 or len(recurrent) != 6
            or any(layer.nonlinearity != "tanh" or not layer.bidirectional for layer in recurrent)
            or any(isinstance(layer, (nn.Conv2d, nn.LSTM, nn.GRU, nn.Dropout)) for layer in model.modules())):
        raise AssertionError("Frozen RNN topology changed")
    paths = list((exp.REPORT / "experiments").glob("**/metrics.json"))
    if len(paths) != 9:
        raise AssertionError(f"Expected exactly nine new runs, found {len(paths)}")
    expected_feature_hash = sha256_file(HANDCRAFTED / "manifest.json")
    expected_flip_hash = sha256_file(exp.FLIP_CACHE / "manifest.json")
    for strategy in exp.ALL_STRATEGIES:
        initial_lr, scheduler = (3e-4, None) if strategy == exp.T0 else exp.policy(strategy)
        for seed in exp.SEEDS:
            metrics, history, config, normal, directory = record(strategy, seed)
            split_hash = prepared[seed]["split"]["sha256"]
            mean, std = exp.t1_normalization(prepared[seed]["train_indices"])
            if (metrics["status"] != "complete" or metrics["seed"] != seed
                    or config["architecture_id"] != exp.ARCHITECTURE_ID
                    or config["representation_id"] != exp.REPRESENTATION_ID
                    or config["representation_shape"] != [190, 7, 7]
                    or any(config[key] != value for key, value in ARCHITECTURES[exp.ARCHITECTURE_ID].items())
                    or config["augmentation"] != "horizontal_flip"
                    or config["dropout_probability"] != 0 or config["weight_decay"] != 1e-4
                    or config["batch_size"] != 32 or config["gradient_clip_norm"] != 1.0
                    or config["max_epochs"] != 40 or config["early_stopping_patience"] != 8
                    or config["learning_rate"] != initial_lr or config["scheduler"] != scheduler
                    or config["training_sample_count"] != 3600 or config["validation_samples"] != 200
                    or metrics["parameter_count"] != count or metrics["training_sample_count"] != 3600
                    or config["split_sha256"] != split_hash or metrics["split_sha256"] != split_hash
                    or normal["split_sha256"] != split_hash
                    or config["feature_cache_manifest_sha256"] != expected_feature_hash
                    or config["flipped_cache_manifest_sha256"] != expected_flip_hash
                    or not np.array_equal(np.asarray(normal["mean"], dtype=np.float32), mean)
                    or not np.array_equal(np.asarray(normal["std"], dtype=np.float32), std)):
                raise AssertionError(f"Frozen protocol mismatch: {directory}")
            if len(history) != metrics["epochs_run"]:
                raise AssertionError(f"History length mismatch: {directory}")
            audit_optimizer = torch.optim.AdamW(nn.Linear(1, 1).parameters(), lr=initial_lr, weight_decay=1e-4)
            audit_scheduler = exp.make_scheduler(audit_optimizer, scheduler)
            reductions = []
            for index, item in enumerate(history):
                actual_lr = float(item["learning_rate"])
                expected_lr = audit_optimizer.param_groups[0]["lr"]
                if not math.isclose(actual_lr, expected_lr, rel_tol=1e-10, abs_tol=1e-12):
                    raise AssertionError(f"Unexpected epoch LR: {directory}, epoch {index + 1}")
                audit_optimizer.step()
                if strategy == exp.T2:
                    audit_scheduler.step()
                elif strategy == exp.T3:
                    audit_scheduler.step(float(item["validation_loss"]))
                    if audit_optimizer.param_groups[0]["lr"] < actual_lr - 1e-12:
                        reductions.append(index + 1)
            if strategy == exp.T3 and reductions != metrics["epochs_of_lr_reductions"]:
                raise AssertionError(f"Plateau reduction events mismatch: {directory}")
            if strategy != exp.T0:
                checkpoint = torch.load(exp.run_dir(strategy, seed) / "best_model.pt",
                                        map_location="cpu", weights_only=True)
                if (checkpoint["epoch"] != metrics["best_epoch"]
                        or checkpoint["scheduler_config"] != scheduler
                        or (scheduler is not None and checkpoint["scheduler_state"] is None)):
                    raise AssertionError(f"Checkpoint scheduler metadata mismatch: {directory}")
            with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                predictions = list(csv.DictReader(stream))
            if len(predictions) != 200 or len({item["filename"] for item in predictions}) != 200:
                raise AssertionError(f"Prediction count mismatch: {directory}")
            required = {"filename", "true_label", "predicted_label", "logit_cat", "logit_dog",
                        "prob_cat", "prob_dog", "correct"}
            if not required.issubset(predictions[0]):
                raise AssertionError(f"Prediction columns mismatch: {directory}")
            matrix = [[0, 0], [0, 0]]
            for item in predictions:
                truth, predicted = int(item["true_label"]), int(item["predicted_label"])
                matrix[truth][predicted] += 1
                if int(item["correct"]) != int(truth == predicted):
                    raise AssertionError(f"Prediction correctness mismatch: {directory}")
            if matrix != metrics["confusion_matrix"] or sum(map(sum, matrix)) != 200:
                raise AssertionError(f"Confusion matrix mismatch: {directory}")
    rows, groups, _ = rows_and_groups()
    decision = json.loads((exp.REPORT / "lr_decision.json").read_text(encoding="utf-8"))
    if len(rows) != 12 or decision["selected_strategy"] != select(groups)["selected_strategy"]:
        raise AssertionError("Aggregate decision mismatch")
    print("RNN-TRAIN-003 audit OK: 9 new runs, 3 imported baselines, 1800 predictions, frozen protocol")
