"""Run Stage-I handcrafted spatial-representation experiments on data/train only."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

from dataset import NORMALIZE_MEAN, NORMALIZE_STD
from representations.cache import (
    HANDCRAFTED, RAW, ROOT, build_cache, current_commit, ensure_split,
    normalization, sha256_file, source_samples, split_indices, validate_cache,
)
from representations.preprocess import preprocess_for_handcrafted
from representations.probes import run_linear, run_mlp, tiny_overfit
from representations.registry import (
    INITIAL_REPRESENTATIONS, RAW_REPRESENTATION_ID, Representation,
    get_representation,
)
from training import write_json

OUTPUT = ROOT / "outputs/stage1"
REPORT = ROOT / "report/stage1"
ARTIFACT_NAMES = ("config.json", "normalization.json", "metrics.json", "history.csv", "train_summary.json", "predictions.csv")
RESULT_COLUMNS = (
    "batch_id", "run_id", "representation_id", "probe", "seed", "channels", "spatial_h",
    "spatial_w", "flatten_dim", "parameter_count", "validation_accuracy", "cat_accuracy",
    "dog_accuracy", "balanced_accuracy", "macro_f1", "best_epoch", "fit_seconds",
    "feature_extraction_seconds", "cache_bytes", "split_sha256", "git_commit", "status",
)


def preprocess_control_letterbox(image: Image.Image, output_size: int = 64) -> Image.Image:
    """Change only resize geometry relative to DNN-001's bilinear RGB input."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(output_size / width, output_size / height)
    new_width = max(1, min(output_size, round(width * scale)))
    new_height = max(1, min(output_size, round(height * scale)))
    resized = np.asarray(rgb.resize((new_width, new_height), Image.Resampling.BILINEAR))
    left = (output_size - new_width) // 2
    right = output_size - new_width - left
    top = (output_size - new_height) // 2
    bottom = output_size - new_height - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    return Image.fromarray(padded)


class LetterboxDataset(Dataset):
    def __init__(self, samples, training: bool) -> None:
        self.samples = samples
        steps = [transforms.RandomHorizontalFlip(p=0.5)] if training else []
        steps.extend((transforms.ToTensor(), transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)))
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            letterboxed = preprocess_control_letterbox(image)
        return self.transform(letterboxed), label


def diagnostic_contact_sheet() -> None:
    """Inspect 16 extreme aspect ratios using training images only."""
    samples = source_samples()
    ratios = []
    for path, _ in samples:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            ratios.append((width / height, path))
    ratios.sort(key=lambda item: item[0])
    selected = ratios[:8] + ratios[-8:]
    sheet = Image.new("RGB", (4 * 180, 4 * 158), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (ratio, path) in enumerate(selected):
        with Image.open(path) as image:
            result = preprocess_for_handcrafted(image)
        if result.shape != (128, 128, 3):
            raise AssertionError(f"Wrong preprocessing shape for {path}")
        x, y = (index % 4) * 180, (index // 4) * 158
        sheet.paste(Image.fromarray(result), (x + 25, y))
        draw.text((x + 4, y + 131), f"{path.name} {ratio:.2f}", fill="black")
    path = OUTPUT / "diagnostics/letterbox_extreme_aspects.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"Decoded {len(samples)} training images; contact sheet: {path}", flush=True)


def _archive(output_dir: Path, batch_id: str, seed: int, representation_id: str, probe: str) -> None:
    target = REPORT / "experiments" / batch_id / f"seed{seed}" / representation_id / probe
    target.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, target / name)


def _output_dir(batch_id: str, seed: int, representation_id: str, probe: str) -> Path:
    return OUTPUT / batch_id / f"seed{seed}" / representation_id / probe


def _run_probe(batch_id: str, representation_id: str, representation: Representation | None, split: dict, probe: str) -> dict:
    seed = split["seed"]
    output_dir = _output_dir(batch_id, seed, representation_id, probe)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("status") != "complete" or metrics.get("split_sha256") != split["sha256"]:
            raise RuntimeError(f"Existing run is incompatible: {output_dir}")
        _archive(output_dir, batch_id, seed, representation_id, probe)
        print(f"Reused completed {metrics['experiment_id']}", flush=True)
        return metrics
    train_indices, validation_indices, _ = split_indices(split)
    labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    mean, std = (None, None) if representation is None else normalization(train_indices, representation.channels)
    try:
        if probe == "LinearSVC":
            metrics = run_linear(batch_id, representation_id, representation, split, train_indices, validation_indices, labels, mean, std, output_dir)
        else:
            metrics = run_mlp(batch_id, representation_id, representation, split, train_indices, validation_indices, labels, mean, std, output_dir)
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "failure.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc(), "code_commit": current_commit()})
        raise
    _archive(output_dir, batch_id, seed, representation_id, probe)
    print(f"Completed {metrics['experiment_id']}: {metrics['validation_accuracy']:.3f}", flush=True)
    return metrics


def _run_representation(batch_id: str, split: dict, representation_id: str, representation: Representation | None) -> None:
    for probe in ("LinearSVC", "MLP"):
        _run_probe(batch_id, representation_id, representation, split, probe)
    rebuild_results()


def run_control() -> None:
    split = ensure_split(42)
    train_indices, validation_indices, _ = split_indices(split)
    sample_lookup = {path.name: (path, label) for path, label in source_samples()}
    train_dataset = LetterboxDataset([sample_lookup[name] for name in split["train"]], training=True)
    validation_dataset = LetterboxDataset([sample_lookup[name] for name in split["internal_validation"]], training=False)
    labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    control_id = "CTRL-LBOX64-GEOM"
    output_dir = _output_dir(control_id, 42, control_id, "MLP")
    if (output_dir / "metrics.json").exists():
        _archive(output_dir, control_id, 42, control_id, "MLP")
        return
    try:
        run_mlp(control_id, control_id, None, split, train_indices, validation_indices, labels, None, None, output_dir, train_dataset, validation_dataset, train_augmentation="RandomHorizontalFlip(p=0.5)", num_workers=2)
    except Exception as error:
        write_json(output_dir / "failure.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        raise
    _archive(output_dir, control_id, 42, control_id, "MLP")
    rebuild_results()


def _fusion_representation(decision: dict) -> Representation:
    return get_representation("REP-006-FUSION", tuple(decision["fusion_groups"]))


def _read_metrics(batch_id: str, seed: int, representation_id: str, probe: str = "MLP") -> dict:
    path = _output_dir(batch_id, seed, representation_id, probe) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Required experiment missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_a_decision() -> dict:
    results = {rep.identifier: _read_metrics("S1A-001", 42, rep.identifier) for rep in INITIAL_REPRESENTATIONS}
    accuracy = {name: metrics["validation_accuracy"] for name, metrics in results.items()}
    lbp_drop = (
        accuracy["REP-001-HOG"] - accuracy["REP-002-HOG-LBP"] > 0.01
        and accuracy["REP-003-ROOTSIFT"] - accuracy["REP-004-ROOTSIFT-LBP"] > 0.01
    )
    hsv_gain = accuracy["REP-005-ROOTSIFT-LBP-HSV"] - accuracy["REP-004-ROOTSIFT-LBP"]
    min_balance = lambda item: min(item["cat_accuracy"], item["dog_accuracy"])
    hsv_balance_gain = min_balance(results["REP-005-ROOTSIFT-LBP-HSV"]) - min_balance(results["REP-004-ROOTSIFT-LBP"])
    hsv_retain = hsv_gain >= 0.005 or hsv_balance_gain >= 0.02
    hog_best = max(("REP-001-HOG", "REP-002-HOG-LBP"), key=lambda name: accuracy[name])
    sift_best = max(("REP-003-ROOTSIFT", "REP-004-ROOTSIFT-LBP", "REP-005-ROOTSIFT-LBP-HSV"), key=lambda name: accuracy[name])
    best_accuracy = max(accuracy.values())
    fusion_trigger = best_accuracy - accuracy[hog_best] <= 0.02 + 1e-12 and best_accuracy - accuracy[sift_best] <= 0.02 + 1e-12
    fusion_groups = ["hog"]
    if not lbp_drop and ("LBP" in hog_best or "LBP" in sift_best):
        fusion_groups.append("lbp")
    if hsv_retain and "HSV" in sift_best:
        fusion_groups.append("hsv")
    fusion_groups.append("rootsift")
    disagreement = None
    if fusion_trigger:
        def read_predictions(name: str) -> dict[str, bool]:
            path = _output_dir("S1A-001", 42, name, "MLP") / "predictions.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                return {row["filename"]: bool(int(row["correct"])) for row in csv.DictReader(stream)}
        hog_predictions, sift_predictions = read_predictions(hog_best), read_predictions(sift_best)
        if hog_predictions.keys() != sift_predictions.keys():
            raise AssertionError("Prediction files use different validation examples")
        counts = {"A_correct_B_wrong": 0, "A_wrong_B_correct": 0, "both_correct": 0, "both_wrong": 0}
        for name in hog_predictions:
            a, b = hog_predictions[name], sift_predictions[name]
            key = "both_correct" if a and b else "both_wrong" if not a and not b else "A_correct_B_wrong" if a else "A_wrong_B_correct"
            counts[key] += 1
        disagreement = {"A": hog_best, "B": sift_best, **counts, "prediction_disagreement_rate": (counts["A_correct_B_wrong"] + counts["A_wrong_B_correct"]) / len(hog_predictions)}
    candidate_ids = list(results)
    fusion_pending = fusion_trigger and not (_output_dir("S1A-001", 42, "REP-006-FUSION", "MLP") / "metrics.json").exists()
    if fusion_trigger and not fusion_pending:
        candidate_ids.append("REP-006-FUSION")
        accuracy["REP-006-FUSION"] = _read_metrics("S1A-001", 42, "REP-006-FUSION")["validation_accuracy"]
    selected = []
    if not fusion_pending:
        dimensions = {rep.identifier: rep.flatten_dim for rep in INITIAL_REPRESENTATIONS}
        if fusion_trigger:
            dimensions["REP-006-FUSION"] = _fusion_representation({"fusion_groups": fusion_groups}).flatten_dim
        ranked = sorted(candidate_ids, key=lambda name: (-accuracy[name], dimensions[name], name))
        selected = [ranked[0]]
        eligible = [name for name in ranked[1:] if accuracy[ranked[0]] - accuracy[name] <= 0.015 + 1e-12 or dimensions[name] <= dimensions[ranked[0]] * 0.75]
        if eligible:
            near_small = [name for name in eligible if accuracy[ranked[0]] - accuracy[name] <= 0.005 + 1e-12 and dimensions[name] < dimensions[eligible[0]] * 0.75]
            selected.append(min(near_small, key=lambda name: dimensions[name]) if near_small else eligible[0])
    decision = {
        "batch_id": "S1A-001", "lbp_removed": lbp_drop, "hsv_retained": hsv_retain,
        "hsv_accuracy_gain": hsv_gain, "hsv_min_class_accuracy_gain": hsv_balance_gain,
        "hog_family_best": hog_best, "rootsift_family_best": sift_best,
        "fusion_triggered": fusion_trigger, "fusion_pending": fusion_pending,
        "fusion_groups": fusion_groups if fusion_trigger else [],
        "prediction_overlap": disagreement, "selected_for_phase_b": selected,
        "phase_a_accuracy": accuracy,
    }
    write_json(REPORT / "phase_a_decision.json", decision)
    return decision


def run_fusion() -> None:
    decision = _phase_a_decision()
    if not decision["fusion_triggered"]:
        print("Fusion criterion was not met; no fusion run", flush=True)
        return
    split = ensure_split(42)
    _run_representation("S1A-001", split, "REP-006-FUSION", _fusion_representation(decision))
    _phase_a_decision()


def run_confirm() -> None:
    decision = _phase_a_decision()
    if decision["fusion_pending"] or len(decision["selected_for_phase_b"]) != 2:
        raise RuntimeError("Phase-A selection is not complete; inspect decision rules")
    for seed in (42, 123, 2026):
        split = ensure_split(seed)
        for name in (RAW_REPRESENTATION_ID, *decision["selected_for_phase_b"]):
            representation = None if name == RAW_REPRESENTATION_ID else _fusion_representation(decision) if name == "REP-006-FUSION" else get_representation(name)
            _run_representation("S1B-001", split, name, representation)


def rebuild_results() -> list[dict]:
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((REPORT / "experiments").glob("**/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rep_id = metrics["representation_id"]
        shape = metrics["representation_shape"]
        manifest_path = (RAW if rep_id == RAW_REPRESENTATION_ID else HANDCRAFTED) / "manifest.json"
        if rep_id.startswith("CTRL-LBOX64"):
            manifest = {"extraction_seconds": 0.0, "bytes_on_disk": 0}
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append({
            "batch_id": metrics["batch_id"], "run_id": metrics["experiment_id"],
            "representation_id": rep_id, "probe": metrics["probe"], "seed": metrics["seed"],
            "channels": shape[0], "spatial_h": shape[1], "spatial_w": shape[2],
            "flatten_dim": metrics["flatten_dim"], "parameter_count": metrics["parameter_count"],
            "validation_accuracy": metrics["validation_accuracy"],
            "cat_accuracy": metrics["cat_accuracy"], "dog_accuracy": metrics["dog_accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"],
            "best_epoch": metrics["best_epoch"], "fit_seconds": metrics["fit_seconds"],
            "feature_extraction_seconds": manifest["extraction_seconds"],
            "cache_bytes": manifest["bytes_on_disk"], "split_sha256": metrics["split_sha256"],
            "git_commit": metrics["code_commit"], "status": metrics["status"],
        })
    rows.sort(key=lambda row: (row["batch_id"], int(row["seed"]), row["representation_id"], row["probe"]))
    with (REPORT / "stage1_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_representation_manifest() -> None:
    handmade = json.loads((HANDCRAFTED / "manifest.json").read_text(encoding="utf-8"))
    raw = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    write_json(REPORT / "representation_manifest.json", {
        "schema_version": 1,
        "source_dataset": "data/train",
        "handcrafted_cache_config_sha256": handmade["config_sha256"],
        "handcrafted_cache_manifest_sha256": sha256_file(HANDCRAFTED / "manifest.json"),
        "handcrafted_cache_bytes": handmade["bytes_on_disk"],
        "handcrafted_extraction_seconds": handmade["extraction_seconds"],
        "raw_cache_config_sha256": raw["config_sha256"],
        "raw_cache_manifest_sha256": sha256_file(RAW / "manifest.json"),
        "channel_groups": handmade["channel_groups"],
        "representations": [
            {"id": RAW_REPRESENTATION_ID, "shape": [3, 64, 64], "flatten_dim": 12288, "groups": ["raw RGB"]},
            *(
                {"id": rep.identifier, "shape": list(rep.shape), "flatten_dim": rep.flatten_dim, "groups": list(rep.groups)}
                for rep in INITIAL_REPRESENTATIONS
            ),
        ],
        "preprocessing": handmade["preprocessing"],
        "dependency_versions": handmade["dependency_versions"],
        "code_commit": handmade["code_commit"],
    })


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def audit_results() -> None:
    """Recompute validation counts from every saved prediction CSV."""
    validate_cache()
    metrics_paths = sorted((REPORT / "experiments").glob("**/metrics.json"))
    if not metrics_paths:
        raise RuntimeError("No archived experiments to audit")
    splits: dict[int, dict] = {}
    for path in metrics_paths:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        seed = metrics["seed"]
        if seed not in splits:
            splits[seed] = ensure_split(seed)
        split = splits[seed]
        if metrics["split_sha256"] != split["sha256"]:
            raise AssertionError(f"Split hash mismatch: {path}")
        with (path.parent / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        if [row["filename"] for row in predictions] != split["internal_validation"]:
            raise AssertionError(f"Prediction filename order mismatch: {path}")
        matrix = [[0, 0], [0, 0]]
        for row in predictions:
            actual = int(row["true_label"])
            predicted = int(row["predicted_label"])
            if actual != (0 if row["filename"].startswith("cat.") else 1):
                raise AssertionError(f"Incorrect prediction label: {path}")
            if predicted not in (0, 1) or int(row["correct"]) != int(actual == predicted):
                raise AssertionError(f"Invalid prediction row: {path}")
            if not np.isfinite(float(row["score_or_margin"])):
                raise AssertionError(f"Non-finite prediction score: {path}")
            matrix[actual][predicted] += 1
        if len(predictions) != 200 or matrix != metrics["confusion_matrix"]:
            raise AssertionError(f"Confusion matrix mismatch: {path}")
        accuracy = (matrix[0][0] + matrix[1][1]) / 200
        if abs(accuracy - metrics["validation_accuracy"]) > 1e-9:
            raise AssertionError(f"Accuracy mismatch: {path}")
        normalization_data = json.loads((path.parent / "normalization.json").read_text(encoding="utf-8"))
        if len(normalization_data["mean"]) != metrics["representation_shape"][0] or len(normalization_data["std"]) != metrics["representation_shape"][0]:
            raise AssertionError(f"Normalization dimension mismatch: {path}")
        if metrics["probe"] == "MLP":
            with (path.parent / "history.csv").open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            best = history[metrics["best_epoch"] - 1]
            if abs(float(best["validation_accuracy"]) - accuracy) > 1e-9:
                raise AssertionError(f"Best epoch differs from predictions: {path}")
    rows = rebuild_results()
    if len(rows) != len(metrics_paths):
        raise AssertionError("Results CSV row count differs from archived runs")
    print(f"Audited {len(metrics_paths)} runs and {len(metrics_paths) * 200} validation predictions", flush=True)


def _phase_b_decision(phase_a: dict) -> dict:
    names = phase_a["selected_for_phase_b"]
    if len(names) != 2:
        raise RuntimeError("Need exactly two handcrafted candidates")
    measurements = {
        name: [_read_metrics("S1B-001", seed, name) for seed in (42, 123, 2026)]
        for name in (RAW_REPRESENTATION_ID, *names)
    }
    def aggregate(records: list[dict]) -> dict:
        accuracy = np.asarray([record["validation_accuracy"] for record in records])
        return {
            "mean_accuracy": float(accuracy.mean()), "std_accuracy": float(accuracy.std(ddof=1)),
            "minimum_accuracy": float(accuracy.min()),
            "mean_cat_accuracy": float(np.mean([record["cat_accuracy"] for record in records])),
            "mean_dog_accuracy": float(np.mean([record["dog_accuracy"] for record in records])),
            "mean_balanced_accuracy": float(np.mean([record["balanced_accuracy"] for record in records])),
            "mean_macro_f1": float(np.mean([record["macro_f1"] for record in records])),
            "seed_accuracies": accuracy.tolist(),
            "mean_linear_svc_accuracy": float(np.mean([_read_metrics("S1B-001", seed, records[0]["representation_id"], "LinearSVC")["validation_accuracy"] for seed in (42, 123, 2026)])),
        }
    summary = {name: aggregate(records) for name, records in measurements.items()}
    sorted_names = sorted(names, key=lambda name: -summary[name]["mean_accuracy"])
    leader, runner = sorted_names
    leader_rep = _fusion_representation(phase_a) if leader == "REP-006-FUSION" else get_representation(leader)
    runner_rep = _fusion_representation(phase_a) if runner == "REP-006-FUSION" else get_representation(runner)
    difference = summary[leader]["mean_accuracy"] - summary[runner]["mean_accuracy"]
    if difference >= 0.01 - 1e-12:
        winner = leader
        rationale = "mean accuracy lead >= 1.0 percentage point"
    else:
        smaller, larger = (leader, runner) if leader_rep.flatten_dim < runner_rep.flatten_dim else (runner, leader)
        larger_balance = min(summary[larger]["mean_cat_accuracy"], summary[larger]["mean_dog_accuracy"])
        smaller_balance = min(summary[smaller]["mean_cat_accuracy"], summary[smaller]["mean_dog_accuracy"])
        winner = larger if larger_balance - smaller_balance >= 0.02 - 1e-12 else smaller
        rationale = "mean accuracy difference < 1.0 point; dimension and class-balance tie break"
    raw_mean = summary[RAW_REPRESENTATION_ID]["mean_accuracy"]
    improvement = summary[winner]["mean_accuracy"] - raw_mean
    better_seeds = sum(a > b for a, b in zip(summary[winner]["seed_accuracies"], summary[RAW_REPRESENTATION_ID]["seed_accuracies"]))
    decision = {
        "batch_id": "S1B-001", "measurements": summary, "winner": winner,
        "winner_shape": list((_fusion_representation(phase_a) if winner == "REP-006-FUSION" else get_representation(winner)).shape),
        "winner_flatten_dim": (_fusion_representation(phase_a) if winner == "REP-006-FUSION" else get_representation(winner)).flatten_dim,
        "selection_rationale": rationale, "raw_control_improvement": improvement,
        "seeds_beating_raw_control": better_seeds,
        "stage1_success_confirmed": improvement >= 0.02 - 1e-12,
    }
    write_json(REPORT / "phase_b_decision.json", decision)
    return decision


def summarize(final: bool = False) -> None:
    rows = rebuild_results()
    if not rows:
        raise RuntimeError("No formal probe results yet")
    phase_a = None
    if all((_output_dir("S1A-001", 42, rep.identifier, "MLP") / "metrics.json").exists() for rep in INITIAL_REPRESENTATIONS):
        phase_a = _phase_a_decision()
    phase_b = _phase_b_decision(phase_a) if final and phase_a else None
    if phase_b:
        manifest_path = REPORT / "representation_manifest.json"
        representation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if phase_a["fusion_triggered"]:
            fusion = _fusion_representation(phase_a)
            representation_manifest["conditional_fusion"] = {
                "id": fusion.identifier,
                "groups": list(fusion.groups),
                "shape": list(fusion.shape),
                "flatten_dim": fusion.flatten_dim,
            }
        representation_manifest["selected_representation_id"] = phase_b["winner"]
        write_json(manifest_path, representation_manifest)
    lines = [
        "# Stage I：手工空间表示实验汇总", "",
        "本文件由 `python stage1.py summarize-screen` 或 `summarize-confirm` 根据实验 JSON/CSV 自动生成。所有选择仅依据 `data/train` 的内部划分；Stage I 未使用 `data/val`。", "",
        "## 历史基线与预处理对照", "",
        "DNN-001：64×64 直接缩放、`12288→256→64→2`、训练时水平翻转；内部验证 69.50%，此前独立测试 62.60%。它是历史作业基线，不与无增强的 REP-000 视作完全相同的训练实验。", "",
    ]
    control_rows = [row for row in rows if row["batch_id"] == "CTRL-LBOX64-GEOM"]
    if control_rows:
        row = control_rows[0]
        lines.append(f"CTRL-LBOX64-GEOM：使用与 DNN-001 相同的 RGB、双线性插值和训练配置，仅保留长宽比并反射填充至 64×64；种子 42 内部验证 {_format_percent(float(row['validation_accuracy']))}，猫 {_format_percent(float(row['cat_accuracy']))}，狗 {_format_percent(float(row['dog_accuracy']))}。")
        earlier = next((item for item in rows if item["batch_id"] == "CTRL-LBOX64"), None)
        if earlier:
            lines.append(f"早期 CTRL-LBOX64 运行得到 {_format_percent(float(earlier['validation_accuracy']))}，但同时使用 EXIF 校正与 LANCZOS 插值，仅保留为有混杂因素的历史记录，不用于几何对照结论。")
    else:
        lines.append("CTRL-LBOX64：尚未运行。")
    lines += ["", "## Phase A：种子 42", "", "| 表示 | LinearSVC | MLP | 猫 / 狗（MLP） | 维度 |", "|---|---:|---:|---:|---:|"]
    for name in (RAW_REPRESENTATION_ID, *(rep.identifier for rep in INITIAL_REPRESENTATIONS), "REP-006-FUSION"):
        subset = [row for row in rows if row["batch_id"] == "S1A-001" and row["representation_id"] == name]
        if not subset:
            continue
        by_probe = {row["probe"]: row for row in subset}
        mlp = by_probe.get("MLP")
        linear = by_probe.get("LinearSVC")
        lines.append(f"| {name} | {_format_percent(float(linear['validation_accuracy'])) if linear else '待测'} | {_format_percent(float(mlp['validation_accuracy'])) if mlp else '待测'} | {_format_percent(float(mlp['cat_accuracy'])) + ' / ' + _format_percent(float(mlp['dog_accuracy'])) if mlp else '待测'} | {mlp['flatten_dim'] if mlp else linear['flatten_dim']} |")
    if phase_a:
        lines += ["", "### Phase A 决策", "",
            f"LBP：{'两条分支均下降超过 1.0 个百分点，后续融合移除' if phase_a['lbp_removed'] else '未同时满足移除条件，保留为可选通道'}。",
            f"HSV：加入 RootSIFT+LBP 后准确率变化 {phase_a['hsv_accuracy_gain'] * 100:+.2f} 个百分点，最低类别准确率变化 {phase_a['hsv_min_class_accuracy_gain'] * 100:+.2f} 个百分点；{'保留' if phase_a['hsv_retained'] else '不带入融合'}。",
            f"HOG 与 RootSIFT 融合条件：{'触发' if phase_a['fusion_triggered'] else '未触发'}。",
        ]
        if phase_a["prediction_overlap"]:
            overlap = phase_a["prediction_overlap"]
            lines.append(f"两分支预测重叠：A 对/B 错 {overlap['A_correct_B_wrong']}，A 错/B 对 {overlap['A_wrong_B_correct']}，都对 {overlap['both_correct']}，都错 {overlap['both_wrong']}，预测正确性分歧率 {_format_percent(overlap['prediction_disagreement_rate'])}。")
        lines.append(f"Phase B 候选：{', '.join(phase_a['selected_for_phase_b']) if phase_a['selected_for_phase_b'] else '等待融合或规则复核'}。")
        hog_lbp_gain = phase_a["phase_a_accuracy"]["REP-002-HOG-LBP"] - phase_a["phase_a_accuracy"]["REP-001-HOG"]
        sift_lbp_gain = phase_a["phase_a_accuracy"]["REP-004-ROOTSIFT-LBP"] - phase_a["phase_a_accuracy"]["REP-003-ROOTSIFT"]
        fusion_accuracy = phase_a["phase_a_accuracy"].get("REP-006-FUSION")
        lines.append(
            f"取舍依据：LBP 在 HOG 和 RootSIFT 分支分别改变 {hog_lbp_gain * 100:+.2f}、{sift_lbp_gain * 100:+.2f} 个百分点；"
            f"HSV 在 RootSIFT+LBP 分支改变 {phase_a['hsv_accuracy_gain'] * 100:+.2f} 个百分点。"
            f"HOG 与 RootSIFT 分支存在互补错误；条件融合后的 MLP 为 {_format_percent(fusion_accuracy) if fusion_accuracy is not None else '待测'}。"
            "最终通道去留仅依据上述预定规则与三种子结果。"
        )
    lines += ["", "## Phase B：三种子确认", ""]
    if phase_b:
        lines += ["标准差为三种子样本标准差。", "", "| 表示 | MLP 平均 ± 标准差 | 最差种子 | 猫 / 狗平均 | LinearSVC 平均 |", "|---|---:|---:|---:|---:|"]
        for name, item in phase_b["measurements"].items():
            lines.append(f"| {name} | {_format_percent(item['mean_accuracy'])} ± {_format_percent(item['std_accuracy'])} | {_format_percent(item['minimum_accuracy'])} | {_format_percent(item['mean_cat_accuracy'])} / {_format_percent(item['mean_dog_accuracy'])} | {_format_percent(item['mean_linear_svc_accuracy'])} |")
        lines += ["", "## 选择与后续接口", "",
            f"选择 **{phase_b['winner']}**，形状 **{phase_b['winner_shape']}**，展平维度 **{phase_b['winner_flatten_dim']}**。规则依据：{phase_b['selection_rationale']}。相对 REP-000 的平均准确率差为 {phase_b['raw_control_improvement'] * 100:+.2f} 个百分点，3 个种子中有 {phase_b['seeds_beating_raw_control']} 个超过原始像素对照。",
            f"按预先定义的阈值，手工表示改进{'已确认' if phase_b['stage1_success_confirmed'] else '未确认；按指南需进入单独的回退实验'}。",
            "DNN：展平 `[C,7,7]`；CNN：直接输入 `[C,7,7]`，使用适合 7×7 的浅层 CNN；RNN：每行为一个时间步，`[7,7*C]`。原始图像的 CNN/RNN 作业基线仍需独立完成。", "",
            "胜出表示包含 128 通道 RootSIFT。可在单独的 S1C-001 中测试只用训练描述子拟合的 PCA64 压缩；仅在三种子平均准确率下降不超过 0.5 个百分点且确实缩减维度时保留。当前未运行 PCA。", "",
            "## 局限", "",
            "仅使用 2000 张开发图像的三个 90/10 内部划分；重复划分存在样本重叠。DNN-001 的 500 张测试图已在早前被查看，本阶段严格未触碰，不能将其当作本阶段独立调参证据。分类器容量随输入维度改变，但层宽、优化器和训练资源固定。", "",
        ]
    else:
        lines += ["尚未完成三种子确认。", ""]
    path = REPORT / "stage1_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary: {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "build-cache", "validate-cache", "tiny-overfit", "run-control", "run-screen", "summarize-screen", "run-fusion", "run-confirm", "summarize-confirm", "audit-results"))
    args = parser.parse_args()
    if args.command == "smoke":
        diagnostic_contact_sheet()
        from representations.extractors import extract_all, extract_hog, extract_lbp, extract_hsv, extract_rootsift
        sample = source_samples()[0][0]
        with Image.open(sample) as image:
            rgb = preprocess_for_handcrafted(image)
        for name, feature, shape in (("hog", extract_hog(rgb), (36, 7, 7)), ("lbp", extract_lbp(rgb), (10, 7, 7)), ("hsv", extract_hsv(rgb), (16, 7, 7)), ("rootsift", extract_rootsift(rgb), (128, 7, 7)), ("all", extract_all(rgb), (190, 7, 7))):
            if feature.shape != shape or not np.isfinite(feature).all():
                raise AssertionError(f"{name} shape/finite check failed")
            print(f"{name}: {feature.shape}")
    elif args.command == "build-cache":
        handmade, raw = build_cache()
        write_representation_manifest()
        print(f"Built cache: {handmade['feature_shape']}, {handmade['bytes_on_disk']} bytes, {handmade['extraction_seconds']:.1f}s")
        print(f"Raw control: {raw['feature_shape']}, {raw['bytes_on_disk']} bytes")
    elif args.command == "validate-cache":
        manifest = validate_cache()
        print(f"Cache valid: {manifest['feature_shape']}")
    elif args.command == "tiny-overfit":
        validate_cache()
        tiny_overfit(ensure_split(42))
    elif args.command == "run-control":
        validate_cache()
        run_control()
    elif args.command == "run-screen":
        validate_cache()
        split = ensure_split(42)
        for name, representation in ((RAW_REPRESENTATION_ID, None), *((rep.identifier, rep) for rep in INITIAL_REPRESENTATIONS)):
            _run_representation("S1A-001", split, name, representation)
    elif args.command == "summarize-screen":
        summarize()
    elif args.command == "run-fusion":
        validate_cache()
        run_fusion()
    elif args.command == "run-confirm":
        validate_cache()
        run_confirm()
    elif args.command == "summarize-confirm":
        summarize(final=True)
    elif args.command == "audit-results":
        audit_results()


if __name__ == "__main__":
    main()
