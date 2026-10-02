"""Aggregate and audit the final controlled RNN2D width comparison."""

from __future__ import annotations

import csv
import json
import statistics

import numpy as np
from matplotlib import pyplot as plt
from torch import nn

import rnn_architecture_scaling_experiments as exp
from cnn_training_strategy_experiments import validate_flip_cache
from models.rnn2d import ARCHITECTURES, VisionRNN2DClassifier
from representations.cache import HANDCRAFTED, ROOT, sha256_file
from rnn2d_architecture_experiments import context
from rnn_training_strategy_report import late_loss_rise
from rnn_training_strategy_experiments import t1_normalization
from training import write_json

FIELDS = (
    "experiment_id", "architecture_id", "seed", "parameter_count", "embed_dim",
    "rnn_hidden_size", "num_blocks", "training_sample_count", "best_epoch", "final_epoch",
    "train_accuracy_at_best_epoch", "final_train_accuracy", "validation_loss",
    "validation_accuracy", "cat_accuracy", "dog_accuracy", "balanced_accuracy", "macro_f1",
    "fit_seconds", "learning_rate", "weight_decay", "augmentation", "gradient_clip_norm",
    "split_sha256", "feature_cache_manifest_sha256", "flipped_cache_manifest_sha256",
    "code_commit", "status", "source_run",
)


def record(architecture_id: str, seed: int) -> tuple[dict, list[dict], dict, dict, str]:
    directory = (exp.BASE_REPORT / f"seed{seed}" if architecture_id == exp.BASE else
                 exp.REPORT / "experiments" / architecture_id / f"seed{seed}")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    normal = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
    with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    if not history or metrics["seed"] != seed:
        raise AssertionError(f"Incomplete width record: {directory}")
    return metrics, history, config, normal, str(directory.relative_to(ROOT)).replace("\\", "/")


def aggregate() -> tuple[list[dict], dict]:
    rows, groups = [], {}
    for architecture_id in exp.ALL_ARCHITECTURES:
        subset, histories = [], []
        for seed in exp.SEEDS:
            metrics, history, config, _, source = record(architecture_id, seed)
            row = {
                "experiment_id": metrics["experiment_id"], "architecture_id": architecture_id,
                "seed": seed, "parameter_count": metrics["parameter_count"],
                **{key: config[key] for key in ("embed_dim", "rnn_hidden_size", "num_blocks")},
                "training_sample_count": metrics["training_sample_count"],
                "best_epoch": metrics["best_epoch"], "final_epoch": len(history),
                "train_accuracy_at_best_epoch": metrics["train_accuracy_at_best_epoch"],
                "final_train_accuracy": metrics["final_train_accuracy"],
                "validation_loss": metrics["validation_loss"],
                **{key: metrics[key] for key in ("validation_accuracy", "cat_accuracy", "dog_accuracy",
                                                   "balanced_accuracy", "macro_f1", "fit_seconds",
                                                   "learning_rate", "weight_decay", "augmentation",
                                                   "gradient_clip_norm", "split_sha256",
                                                   "feature_cache_manifest_sha256",
                                                   "flipped_cache_manifest_sha256", "code_commit", "status")},
                "source_run": source,
            }
            rows.append(row)
            subset.append(row)
            histories.append(history)
        accuracies = [row["validation_accuracy"] for row in subset]
        cat = statistics.mean(row["cat_accuracy"] for row in subset)
        dog = statistics.mean(row["dog_accuracy"] for row in subset)
        final_train = statistics.mean(row["final_train_accuracy"] for row in subset)
        final_validation = statistics.mean(float(history[-1]["validation_accuracy"])
                                           for history in histories)
        groups[architecture_id] = {
            "architecture_id": architecture_id, "parameter_count": subset[0]["parameter_count"],
            "embed_dim": subset[0]["embed_dim"], "rnn_hidden_size": subset[0]["rnn_hidden_size"],
            "num_blocks": subset[0]["num_blocks"], "training_sample_count": 3600,
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
            "mean_train_validation_gap": final_train - final_validation,
            "late_validation_loss_rise_runs": sum(late_loss_rise(history) for history in histories),
            "mean_fit_seconds": statistics.mean(row["fit_seconds"] for row in subset),
            "seed_accuracies": {str(row["seed"]): row["validation_accuracy"] for row in subset},
        }
    baseline = groups[exp.BASE]
    for group in groups.values():
        group["relative_to_base"] = {
            "mean_accuracy_gain": group["mean_validation_accuracy"] - baseline["mean_validation_accuracy"],
            "worst_seed_gain": group["worst_seed_accuracy"] - baseline["worst_seed_accuracy"],
            "std_change": (group["sample_std_validation_accuracy"] -
                           baseline["sample_std_validation_accuracy"]),
            "parameter_multiplier": group["parameter_count"] / baseline["parameter_count"],
        }
    return rows, groups


def qualifies(group: dict) -> tuple[bool, str]:
    change = group["relative_to_base"]
    mean, worst, std = (change["mean_accuracy_gain"], change["worst_seed_gain"],
                        change["std_change"])
    criterion_a = mean >= 0.0075 - 1e-12 and worst >= -0.005 - 1e-12
    criterion_b = mean >= 0.005 - 1e-12 and worst >= 0.01 - 1e-12
    if not (criterion_a or criterion_b):
        return False, "no_meaningful_gain"
    if std > 0.015 + 1e-12 and not (mean >= 0.015 - 1e-12 and worst >= -1e-12):
        return False, "variance_safeguard"
    return True, "clear_mean_gain" if criterion_a else "balanced_gain"


def decide(groups: dict) -> dict:
    qualified = {key: qualifies(groups[key]) for key in exp.NEW_ARCHITECTURES}
    candidates = [groups[key] for key, (ok, _) in qualified.items() if ok]
    if len(candidates) == 1:
        chosen, selection_rule = candidates[0], "sole_meaningful_width_gain"
    elif len(candidates) == 2:
        candidates.sort(key=lambda group: (-group["mean_validation_accuracy"],
                                           -group["worst_seed_accuracy"],
                                           group["sample_std_validation_accuracy"],
                                           group["parameter_count"]))
        first, second = candidates
        if first["mean_validation_accuracy"] - second["mean_validation_accuracy"] < 0.005 - 1e-12:
            chosen = sorted(candidates, key=lambda group: (-group["worst_seed_accuracy"],
                                                           group["sample_std_validation_accuracy"],
                                                           group["parameter_count"]))[0]
            selection_rule = "close_mean_worst_seed_then_std_then_size"
        else:
            chosen, selection_rule = first, "higher_mean_among_qualifiers"
    else:
        chosen, selection_rule = groups[exp.BASE], "no_meaningful_width_gain"
    if candidates:
        scaling, outcome = "positive scaling", "A"
    else:
        apparent = [groups[key] for key in exp.NEW_ARCHITECTURES
                    if groups[key]["relative_to_base"]["mean_accuracy_gain"] > 0]
        unstable = any(group["relative_to_base"]["std_change"] > 0.015 and
                       group["relative_to_base"]["worst_seed_gain"] <= 0
                       for group in apparent)
        if all(groups[key]["relative_to_base"]["mean_accuracy_gain"] < 0.005 - 1e-12
               for key in exp.NEW_ARCHITECTURES):
            scaling, outcome = "saturation", "B"
        elif unstable:
            scaling, outcome = "unstable scaling", "C"
        elif all(groups[key]["relative_to_base"]["mean_accuracy_gain"] <= 0
                 for key in exp.NEW_ARCHITECTURES):
            scaling, outcome = "negative scaling", "B"
        else:
            scaling, outcome = "saturation", "B"
    return {
        "selected_architecture": chosen["architecture_id"], "selection_rule": selection_rule,
        "qualification": {key: {"qualifies": ok, "reason": reason}
                          for key, (ok, reason) in qualified.items()},
        "scaling_classification": scaling, "outcome": outcome,
        "architecture_optimization_complete": True,
        "frozen_training_recipe": "RNN-TRAIN-T1-FLIP",
        "next_stage": "separate RNN training-setting optimization",
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def plot_curves() -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 10))
    colors = {42: "#1f77b4", 123: "#ff7f0e", 2026: "#2ca02c"}
    for row, architecture_id in enumerate(exp.ALL_ARCHITECTURES):
        for seed in exp.SEEDS:
            history = record(architecture_id, seed)[1]
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
    figure.suptitle("RNN-ARCH-003: frozen flip recipe and controlled width scaling")
    figure.tight_layout()
    figure.savefig(exp.REPORT / "training_curves.png", dpi=150)
    plt.close(figure)


def summarize() -> None:
    rows, groups = aggregate()
    decision = decide(groups)
    exp.REPORT.mkdir(parents=True, exist_ok=True)
    with (exp.REPORT / "architecture_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(exp.REPORT / "architecture_decision.json", {
        "batch_id": exp.BATCH_ID, "baseline_source": str(exp.BASE_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "models": groups, **decision,
    })
    plot_curves()
    lines = [
        "# RNN-ARCH-003：冻结镜像配方的最终宽度比较", "",
        "固定 `REP-006-FUSION` `[190,7,7]`、三份既有 1800/200 内部划分和 `RNN-TRAIN-T1-FLIP` 配方。每份训练集为 1800 原图 + 1800 张 RGB 水平镜像后重提的特征；190 通道归一化只由这 3600 张训练图拟合。固定 AdamW、学习率 `3e-4`、权重衰减 `1e-4`、批量 32、最多 40 轮、验证损失早停耐心 8、梯度裁剪 1.0，无 Dropout 或调度器。BASE 直接导入 T1 三种子记录，未重训；仅新增 WIDE/XWIDE 各三次正式运行，没有访问 `data/val`。", "",
        "三个模型均使用逐位置 Linear 投影、三个双轴双向 tanh `nn.RNN` 残差块、相同的 2 倍通道 MLP 和 49 位置平均池化。只变动嵌入宽度 C 与每方向隐层 H，始终 H=C/4。", "",
        "## 三种子内部验证", "", "标准差为三种子的样本标准差；增益相对 BASE，单位为百分点。", "",
        "| 架构 | C/H/块 | 参数量 | 参数倍率 | 准确率均值 ± 标准差 | 种子 42/123/2026 | 最差/最好 | 猫/狗均值 | Macro F1 | 最佳轮中位数 | 均值/最差增益及标准差变化 | 资格 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for architecture_id in exp.ALL_ARCHITECTURES:
        group = groups[architecture_id]
        change = group["relative_to_base"]
        seeds = "/".join(pct(group["seed_accuracies"][str(seed)]) for seed in exp.SEEDS)
        change_text = "/".join(f"{100 * change[key]:+.2f}" for key in
                               ("mean_accuracy_gain", "worst_seed_gain", "std_change"))
        qualification = decision["qualification"].get(architecture_id, {"reason": "historical baseline"})["reason"]
        lines.append(f"| {architecture_id} | {group['embed_dim']}/{group['rnn_hidden_size']}/{group['num_blocks']} | {group['parameter_count']:,} | {change['parameter_multiplier']:.2f}× | {pct(group['mean_validation_accuracy'])} ± {pct(group['sample_std_validation_accuracy'])} | {seeds} | {pct(group['worst_seed_accuracy'])}/{pct(group['best_seed_accuracy'])} | {pct(group['mean_cat_accuracy'])}/{pct(group['mean_dog_accuracy'])} | {pct(group['mean_macro_f1'])} | {group['median_best_epoch']} | {change_text} | {qualification} |")
    lines += ["", "## 容量与过拟合", "",
              "| 架构 | 末轮训练均值 | 末轮验证均值 | 训练/验证差距 | 后期验证损失回升 ≥0.10 | 平均训练秒数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for architecture_id in exp.ALL_ARCHITECTURES:
        group = groups[architecture_id]
        lines.append(f"| {architecture_id} | {pct(group['mean_final_train_accuracy'])} | {pct(group['mean_final_validation_accuracy'])} | {100 * group['mean_train_validation_gap']:.2f} pp | {group['late_validation_loss_rise_runs']}/3 | {group['mean_fit_seconds']:.2f} |")
    lines += ["", "| 架构 | 种子 | 最佳/末轮 | 最佳轮训练准确率 | 末轮训练/验证准确率 | 最佳验证准确率 | 后期损失回升 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for architecture_id in exp.ALL_ARCHITECTURES:
        for seed in exp.SEEDS:
            metrics, history, _, _, _ = record(architecture_id, seed)
            final = history[-1]
            lines.append(f"| {architecture_id} | {seed} | {metrics['best_epoch']}/{len(history)} | {pct(metrics['train_accuracy_at_best_epoch'])} | {pct(metrics['final_train_accuracy'])}/{pct(float(final['validation_accuracy']))} | {pct(metrics['validation_accuracy'])} | {'是' if late_loss_rise(history) else '否'} |")
    lines += ["", "![三种宽度的训练轨迹](training_curves.png)", "",
              "## 缩放判断与冻结结论", ""]
    base, wide, xwide = (groups[key] for key in exp.ALL_ARCHITECTURES)
    lines += [
        f"BASE→WIDE：均值变化 {100 * wide['relative_to_base']['mean_accuracy_gain']:+.2f} pp，最差种子变化 {100 * wide['relative_to_base']['worst_seed_gain']:+.2f} pp，猫狗差变化 {100 * (wide['absolute_cat_dog_gap'] - base['absolute_cat_dog_gap']):+.2f} pp，标准差变化 {100 * wide['relative_to_base']['std_change']:+.2f} pp。", "",
        f"WIDE→XWIDE：均值变化 {100 * (xwide['mean_validation_accuracy'] - wide['mean_validation_accuracy']):+.2f} pp，最差种子变化 {100 * (xwide['worst_seed_accuracy'] - wide['worst_seed_accuracy']):+.2f} pp，猫狗差变化 {100 * (xwide['absolute_cat_dog_gap'] - wide['absolute_cat_dog_gap']):+.2f} pp，标准差变化 {100 * (xwide['sample_std_validation_accuracy'] - wide['sample_std_validation_accuracy']):+.2f} pp。", "",
        f"BASE、WIDE、XWIDE 的最佳轮中位数分别为 {base['median_best_epoch']}/{wide['median_best_epoch']}/{xwide['median_best_epoch']}，没有随宽度增大而明显提前。末轮训练准确率均值分别为 {pct(base['mean_final_train_accuracy'])}/{pct(wide['mean_final_train_accuracy'])}/{pct(xwide['mean_final_train_accuracy'])}；WIDE 和 XWIDE 的训练/验证差距相对 BASE 分别变化 {100 * (wide['mean_train_validation_gap'] - base['mean_train_validation_gap']):+.2f}/{100 * (xwide['mean_train_validation_gap'] - base['mean_train_validation_gap']):+.2f} pp。三个架构均有 3/3 次后期验证损失回升。XWIDE 的最差种子与标准差改善，但平均准确率仍低于 BASE，因此没有形成可复现的正向均值缩放收益。", "",
        f"根据预设均值、最差种子与方差保护规则，缩放行为判为 **{decision['scaling_classification']}**；选择 **{decision['selected_architecture']}**（`{decision['selection_rule']}`），Outcome **{decision['outcome']}**。", "",
        "架构优化到此结束。下一步是独立的 RNN 训练设置优化阶段；本任务没有进一步加宽、改深、调参、全量训练或评估 `data/val`。", "",
    ]
    (exp.REPORT / "architecture_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected {decision['selected_architecture']}; {decision['scaling_classification']}; Outcome {decision['outcome']}", flush=True)


def audit() -> None:
    channels, prepared = context()
    validate_flip_cache()
    if channels != tuple(range(190)):
        raise AssertionError("REP-006 channel order changed")
    paths = list((exp.REPORT / "experiments").glob("**/metrics.json"))
    if len(paths) != 6:
        raise AssertionError("Expected exactly six new width runs")
    baseline_count = json.loads((exp.BASE_REPORT / "seed42/metrics.json").read_text(encoding="utf-8"))["parameter_count"]
    counts = [baseline_count]
    for architecture_id in exp.NEW_ARCHITECTURES:
        spec = ARCHITECTURES[architecture_id]
        expected = ({"embed_dim": 384, "rnn_hidden_size": 96, "num_blocks": 3}
                    if architecture_id == exp.NEW_ARCHITECTURES[0] else
                    {"embed_dim": 448, "rnn_hidden_size": 112, "num_blocks": 3})
        model = VisionRNN2DClassifier(architecture_id)
        recurrent = [module for module in model.modules() if isinstance(module, nn.RNN)]
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        counts.append(count)
        if (spec != expected or len(model.blocks) != 3 or len(recurrent) != 6
                or any(module.nonlinearity != "tanh" or not module.bidirectional or module.dropout != 0
                       for module in recurrent)
                or any(isinstance(module, (nn.Dropout, nn.Conv2d, nn.LSTM, nn.GRU))
                       for module in model.modules())):
            raise AssertionError("Width model topology changed")
        for seed in exp.SEEDS:
            metrics, history, config, normal, source = record(architecture_id, seed)
            split = prepared[seed]["split"]
            mean, std = t1_normalization(prepared[seed]["train_indices"])
            base_config = record(exp.BASE, seed)[2]
            base_normal = record(exp.BASE, seed)[3]
            directory = ROOT / source
            if (config["batch_id"] != exp.BATCH_ID or config["architecture_id"] != architecture_id
                    or metrics["architecture_id"] != architecture_id or metrics["status"] != "complete"
                    or config["representation_id"] != "REP-006-FUSION"
                    or config["representation_shape"] != [190, 7, 7]
                    or any(config[key] != value for key, value in expected.items())
                    or config["parameter_count"] != count or metrics["parameter_count"] != count
                    or config["training_sample_count"] != 3600 or metrics["training_sample_count"] != 3600
                    or config["validation_samples"] != 200 or config["learning_rate"] != 3e-4
                    or metrics["learning_rate"] != 3e-4 or config["weight_decay"] != 1e-4
                    or metrics["weight_decay"] != 1e-4 or config["dropout_probability"] != 0.0
                    or config["augmentation"] != "horizontal_flip" or metrics["augmentation"] != "horizontal_flip"
                    or config["scheduler"] is not None or config["optimizer"] != "AdamW"
                    or config["loss"] != "CrossEntropyLoss" or config["batch_size"] != 32
                    or config["max_epochs"] != 40 or config["early_stopping_patience"] != 8
                    or config["gradient_clip_norm"] != 1.0 or metrics["gradient_clip_norm"] != 1.0
                    or config["split_sha256"] != split["sha256"] or metrics["split_sha256"] != split["sha256"]
                    or normal["split_sha256"] != split["sha256"]
                    or config["feature_cache_manifest_sha256"] != sha256_file(HANDCRAFTED / "manifest.json")
                    or config["flipped_cache_manifest_sha256"] != sha256_file(exp.FLIP_CACHE / "manifest.json")
                    or config["feature_cache_manifest_sha256"] != base_config["feature_cache_manifest_sha256"]
                    or config["flipped_cache_manifest_sha256"] != base_config["flipped_cache_manifest_sha256"]
                    or not np.array_equal(np.asarray(normal["mean"], dtype=np.float32), mean)
                    or not np.array_equal(np.asarray(normal["std"], dtype=np.float32), std)
                    or normal["mean"] != base_normal["mean"] or normal["std"] != base_normal["std"]):
                raise AssertionError(f"Width protocol mismatch: {directory}")
            with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                predictions = list(csv.DictReader(stream))
            if (len(predictions) != 200 or [row["filename"] for row in predictions] !=
                    split["internal_validation"]):
                raise AssertionError(f"Width validation rows differ from split: {directory}")
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
                    raise AssertionError(f"Prediction mismatch: {directory}")
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
                raise AssertionError(f"Width metrics differ from predictions: {directory}")
    if not counts[0] < counts[1] < counts[2]:
        raise AssertionError("Parameter count ordering changed")
    rows, groups = aggregate()
    decision = json.loads((exp.REPORT / "architecture_decision.json").read_text(encoding="utf-8"))
    if len(rows) != 9 or decision["models"] != groups or any(
            decision[key] != value for key, value in decide(groups).items()):
        raise AssertionError("Width aggregate decision mismatch")
    print("Audited imported BASE, six new width runs, frozen T1 recipe, and 1,200 predictions", flush=True)
