"""Aggregate and audit the nine RNN-ARCH-002 development runs."""

from __future__ import annotations

import csv
import json
import statistics

import numpy as np
from torch import nn

import rnn2d_architecture_experiments as exp
from models.rnn2d import ARCHITECTURES, VisionRNN2DClassifier
from representations.cache import HANDCRAFTED, ROOT, sha256_file
from training import write_json


def category(selected: dict, simple_rnn_b_mean: float) -> tuple[str, str]:
    mean = selected["mean_validation_accuracy"]
    worst = selected["worst_seed_accuracy"]
    # Use the same 0.75 pp effect size as the frozen architecture-selection rule.
    if mean - simple_rnn_b_mean < 0.0075 - 1e-12:
        return "C", "未比最佳简单 RNN 的平均准确率提高至少 0.75 个百分点；建议重新评估 RNN 专用空间表示分辨率。"
    if mean >= 0.795 and worst >= 0.775:
        return "A", "两项竞争力门槛均满足；建议另立有限的 RNN 训练策略阶段。"
    if 0.775 <= mean < 0.795:
        return "B", "2D 递归有进展但尚未达到 A 类门槛；至多建议一个小型训练策略阶段。"
    return "C", "未达到约定的竞争力门槛；建议下一阶段重新评估 RNN 专用空间表示的分辨率，本任务不实施。"


def summarize() -> None:
    rows, grouped = exp.aggregate()
    exp.REPORT.mkdir(parents=True, exist_ok=True)
    with (exp.REPORT / "architecture_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=exp.RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    selection = exp.select(grouped)
    chosen = grouped[selection["selected_architecture"]]
    simple = json.loads((ROOT / "report/rnn_architecture/architecture_decision.json").read_text(encoding="utf-8"))["models"]
    decision_category, recommendation = category(chosen, simple["RNN-ARCH-B"]["mean_validation_accuracy"])
    dnn = json.loads((ROOT / "report/dnn_architecture/architecture_decision.json").read_text(encoding="utf-8"))["models"]["DNN-ARCH-C"]
    cnn = json.loads((ROOT / "report/cnn_architecture/architecture_decision.json").read_text(encoding="utf-8"))["models"]["CNN-ARCH-C"]
    t1 = json.loads((ROOT / "report/cnn_training/training_strategy_decision.json").read_text(encoding="utf-8"))["models"]["CNN-TRAIN-T1-FLIP"]
    references = {
        "RNN-ARCH-B": simple["RNN-ARCH-B"], "RNN-ARCH-C": simple["RNN-ARCH-C"],
        "DNN-ARCH-C": dnn, "CNN-ARCH-C": cnn, "CNN-TRAIN-T1-FLIP": t1,
    }
    write_json(exp.REPORT / "architecture_decision.json", {
        "batch_id": exp.BATCH_ID, "models": grouped, **selection,
        "stage_ii_category": decision_category, "next_step_recommendation": recommendation,
        "descriptive_internal_references": {
            key: {"mean_validation_accuracy": value["mean_validation_accuracy"],
                  "worst_seed_accuracy": value["worst_seed_accuracy"]}
            for key, value in references.items()
        },
    })
    exp.plot_curves()
    pct = exp.pct
    final_trajectories = {}
    for architecture_id in exp.ARCHITECTURE_IDS:
        histories = [exp.record(architecture_id, seed)[1] for seed in exp.SEEDS]
        final_trajectories[architecture_id] = {
            "train": statistics.mean(float(history[-1]["train_accuracy"]) for history in histories),
            "validation": statistics.mean(float(history[-1]["validation_accuracy"]) for history in histories),
            "loss_rises": sum(float(history[-1]["validation_loss"]) -
                               min(float(epoch["validation_loss"]) for epoch in history) >= 0.10
                               for history in histories),
        }
    lines = [
        "# RNN-ARCH-002：双轴标准 RNN 的视觉架构比较", "",
        "仅使用 `data/train` 的种子 42、123、2026 三份既有 1800/200 划分，输入为冻结的 `REP-006-FUSION` `[190,7,7]`。各份归一化只拟合其训练图，无增强；没有使用 `data/val`。", "",
        "每个模型先用共享的逐位置 Linear(190,C) 和 LayerNorm 投影 7×7 网格。每个 BiRNN2D 块对同一预归一化网格分别沿宽度与高度运行参数独立的双向 tanh `nn.RNN`，保留全部位置的输出；两个轴的结果拼接后经线性层融合并残差相加，再通过 Linear→ReLU→Linear 的通道 MLP 和第二次残差。最后对 49 个位置全局平均，再经 LayerNorm 和线性层输出两类 logits。S/M/L 只改变 C、每方向隐层及块数。", "",
        "共同协议：CrossEntropyLoss、AdamW、学习率 3e-4、权重衰减 1e-4、批量 32、最多 40 轮、验证损失早停耐心 8、反传后梯度裁剪范数 1.0。检查点按最高验证准确率、同分较低损失选取。没有 Dropout、调度器或测试时增强。", "",
        "## 三种规模的内部验证", "", "标准差为三种子样本标准差。", "",
        "| 规模 | C / 隐层 / 块数 | 参数量 | 平均准确率 ± 标准差 | 最差种子 | 猫/狗均值 | 类别差 | 平衡准确率 | Macro F1 | 最佳轮中位数 | 平均训练秒数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture_id in exp.ARCHITECTURE_IDS:
        item, spec = grouped[architecture_id], ARCHITECTURES[architecture_id]
        lines.append(f"| {architecture_id} | {spec['embed_dim']} / {spec['rnn_hidden_size']} / {spec['num_blocks']} | {item['parameter_count']:,} | {pct(item['mean_validation_accuracy'])} ± {pct(item['sample_std_validation_accuracy'])} | {pct(item['worst_seed_accuracy'])} | {pct(item['mean_cat_accuracy'])}/{pct(item['mean_dog_accuracy'])} | {item['absolute_cat_dog_gap'] * 100:.2f} pp | {pct(item['mean_balanced_accuracy'])} | {pct(item['mean_macro_f1'])} | {item['median_best_epoch']} | {item['mean_fit_seconds']:.2f} |")
    lines += ["", "## 逐种子训练轨迹", "",
              "| 架构 | 种子 | 最佳轮/末轮 | 最佳验证准确率 | 末轮训练/验证准确率 | 最佳轮→末轮验证损失 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        _, history, _ = exp.record(row["architecture_id"], row["seed"])
        final = history[-1]
        lines.append(f"| {row['architecture_id']} | {row['seed']} | {row['best_epoch']}/{len(history)} | {pct(row['validation_accuracy'])} | {pct(float(final['train_accuracy']))}/{pct(float(final['validation_accuracy']))} | {row['validation_loss']:.4f}→{float(final['validation_loss']):.4f} |")
    lines += ["", "| 架构 | 末轮训练准确率均值 | 末轮验证准确率均值 | 末轮差距 | 最低验证损失后上升 ≥0.10 的运行数 |",
              "|---|---:|---:|---:|---:|"]
    for architecture_id in exp.ARCHITECTURE_IDS:
        trajectory = final_trajectories[architecture_id]
        lines.append(f"| {architecture_id} | {pct(trajectory['train'])} | {pct(trajectory['validation'])} | {(trajectory['train']-trajectory['validation'])*100:.2f} pp | {trajectory['loss_rises']}/3 |")
    lines += ["", "![三规模训练和内部验证曲线](training_curves.png)", "",
              "## 选择、容量与历史参照", "",
              f"按预设规则选定 **{selection['selected_architecture']}**（`{selection['selection_rule']}`）；前两名平均准确率差 {selection['top_two_mean_accuracy_gap'] * 100:.2f} 个百分点。", "",
              f"S→M：平均准确率变化 {(grouped['RNN2D-ARCH-M']['mean_validation_accuracy'] - grouped['RNN2D-ARCH-S']['mean_validation_accuracy'])*100:+.2f} pp；M→L：{(grouped['RNN2D-ARCH-L']['mean_validation_accuracy'] - grouped['RNN2D-ARCH-M']['mean_validation_accuracy'])*100:+.2f} pp。S 并未表现出明显欠拟合：末轮训练准确率均值为 {pct(final_trajectories['RNN2D-ARCH-S']['train'])}，但末轮验证仅 {pct(final_trajectories['RNN2D-ARCH-S']['validation'])}。M 增加容量后均值反而下降；L 的均值及狗类准确率提高，但三种子标准差扩大到 {pct(grouped['RNN2D-ARCH-L']['sample_std_validation_accuracy'])}。三规模的验证损失后期均有回升，显示过拟合和种子敏感性仍然存在。", "",
              f"相对简单 RNN 的 B：入选模型平均准确率变化 {(chosen['mean_validation_accuracy'] - simple['RNN-ARCH-B']['mean_validation_accuracy'])*100:+.2f} pp、最差种子变化 {(chosen['worst_seed_accuracy'] - simple['RNN-ARCH-B']['worst_seed_accuracy'])*100:+.2f} pp；相对简单 RNN 的 C：分别变化 {(chosen['mean_validation_accuracy'] - simple['RNN-ARCH-C']['mean_validation_accuracy'])*100:+.2f} pp 和 {(chosen['worst_seed_accuracy'] - simple['RNN-ARCH-C']['worst_seed_accuracy'])*100:+.2f} pp。", "",
              f"内部开发参照：DNN-ARCH-C {pct(dnn['mean_validation_accuracy'])}、CNN-ARCH-C {pct(cnn['mean_validation_accuracy'])}、CNN-TRAIN-T1-FLIP {pct(t1['mean_validation_accuracy'])}。这些训练协议不同，参照不参与 Stage II 选型，也不使用保留集结果。", "",
              f"## 下一步判定：Category {decision_category}", "", recommendation, "",
              "本阶段没有运行 RNN 最终训练、训练策略搜索、高分辨率表示实验或 `data/val` 评估。", "",
    ]
    (exp.REPORT / "architecture_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected {selection['selected_architecture']}; Category {decision_category}", flush=True)


def audit() -> None:
    channels, prepared = exp.context()
    paths = list((exp.REPORT / "experiments").glob("**/metrics.json"))
    if len(paths) != 9 or channels != tuple(range(190)):
        raise AssertionError("Expected exactly nine frozen REP-006 Stage-II runs")
    for architecture_id in exp.ARCHITECTURE_IDS:
        spec = ARCHITECTURES[architecture_id]
        model = VisionRNN2DClassifier(architecture_id)
        recurrent = [layer for layer in model.modules() if isinstance(layer, nn.RNN)]
        if (len(model.blocks) != spec["num_blocks"] or len(recurrent) != 2 * spec["num_blocks"]
                or any(layer.nonlinearity != "tanh" or not layer.bidirectional
                       or not layer.batch_first or layer.dropout != 0 for layer in recurrent)
                or any(isinstance(layer, (nn.Conv2d, nn.LSTM, nn.GRU, nn.BatchNorm2d))
                       for layer in model.modules())
                or not isinstance(model.projection.linear, nn.Linear)
                or not isinstance(model.head_norm, nn.LayerNorm)):
            raise AssertionError("Stage-II RNN topology changed")
        parameter_ids = [id(p) for layer in recurrent for p in layer.parameters()]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise AssertionError("Horizontal/vertical or block RNN parameters are shared")
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for seed in exp.SEEDS:
            metrics, history, directory = exp.record(architecture_id, seed)
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            normal = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
            split = prepared[seed]["split"]
            if (metrics["status"] != "complete" or metrics["split_sha256"] != split["sha256"]
                    or config["split_sha256"] != split["sha256"]
                    or config["feature_cache_manifest_sha256"] != sha256_file(HANDCRAFTED / "manifest.json")
                    or metrics["feature_cache_manifest_sha256"] != config["feature_cache_manifest_sha256"]
                    or config["representation_id"] != exp.REPRESENTATION_ID
                    or config["representation_shape"] != [190, 7, 7]
                    or config["architecture_id"] != architecture_id or config["seed"] != seed
                    or any(config[key] != value or metrics[key] != value for key, value in spec.items())
                    or config["parameter_count"] != count or metrics["parameter_count"] != count
                    or config["train_samples"] != 1800 or config["validation_samples"] != 200
                    or config["gradient_clip_norm"] != 1.0 or metrics["gradient_clip_norm"] != 1.0
                    or config["rnn_dropout"] != 0.0 or config["scheduler"] is not None
                    or config["augmentation"] != "none" or config["optimizer"] != "AdamW"
                    or config["learning_rate"] != 3e-4 or config["weight_decay"] != 1e-4
                    or config["batch_size"] != 32 or config["max_epochs"] != 40
                    or config["early_stopping_patience"] != 8 or config["loss"] != "CrossEntropyLoss"
                    or normal["split_sha256"] != split["sha256"]
                    or not np.array_equal(np.asarray(normal["mean"], dtype=np.float32), prepared[seed]["mean"])
                    or not np.array_equal(np.asarray(normal["std"], dtype=np.float32), prepared[seed]["std"])):
                raise AssertionError(f"Stage-II protocol or normalization mismatch: {directory}")
            with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                predictions = list(csv.DictReader(stream))
            if (len(predictions) != 200
                    or [row["filename"] for row in predictions] != split["internal_validation"]
                    or [int(row["true_label"]) for row in predictions].count(0) != 100
                    or [int(row["true_label"]) for row in predictions].count(1) != 100):
                raise AssertionError(f"Invalid Stage-II prediction rows: {directory}")
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
                    raise AssertionError(f"Prediction probabilities mismatch: {directory}")
                row_losses.append(-np.log(probs[actual]))
                matrix[actual][guess] += 1
            f1_cat = 2 * matrix[0][0] / (2 * matrix[0][0] + matrix[0][1] + matrix[1][0])
            f1_dog = 2 * matrix[1][1] / (2 * matrix[1][1] + matrix[0][1] + matrix[1][0])
            if (matrix != metrics["confusion_matrix"] or [sum(row) for row in matrix] != [100, 100]
                    or abs((matrix[0][0] + matrix[1][1]) / 200 - metrics["validation_accuracy"]) > 1e-12
                    or abs(float(history[metrics["best_epoch"] - 1]["validation_accuracy"]) - metrics["validation_accuracy"]) > 1e-12
                    or abs(float(np.mean(row_losses)) - metrics["validation_loss"]) > 1e-6
                    or abs((f1_cat + f1_dog) / 2 - metrics["macro_f1"]) > 1e-12):
                raise AssertionError(f"Stage-II prediction or aggregate metrics mismatch: {directory}")
    rows, grouped = exp.aggregate()
    decision = json.loads((exp.REPORT / "architecture_decision.json").read_text(encoding="utf-8"))
    selected = exp.select(grouped)["selected_architecture"]
    if (len(rows) != 9 or decision["models"] != grouped
            or decision["selected_architecture"] != selected
            or decision["stage_ii_category"] != category(grouped[selected],
                json.loads((ROOT / "report/rnn_architecture/architecture_decision.json").read_text(encoding="utf-8"))["models"]["RNN-ARCH-B"]["mean_validation_accuracy"])[0]):
        raise AssertionError("Stage-II aggregate decision mismatch")
    print("Audited nine two-axis RNN runs, frozen splits/normalization and 1,800 predictions", flush=True)
