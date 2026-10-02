"""Run the one predeclared RNN flip plus weight-decay combination."""

from __future__ import annotations

import argparse
import json
import shutil
import traceback

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset

from cnn_training_strategy_experiments import FLIP_CACHE, t1_normalization, validate_flip_cache
from dnn_architecture_experiments import make_dataset
from models.rnn2d import ARCHITECTURES
from representations.cache import HANDCRAFTED, ROOT, sha256_file
from representations.dataset import CachedRepresentationDataset
from rnn2d_architecture_experiments import context
from rnn_training_strategy_experiments import (
    ARCHITECTURE_ID, ARTIFACTS, COMBINATION_STRATEGY, SEEDS,
    model_for, strategy_settings, train_one,
)
from training import write_json

BATCH_ID = "RNN-TRAIN-002"
STRATEGY_ID = COMBINATION_STRATEGY
OUTPUT = ROOT / "outputs/rnn_training" / BATCH_ID / STRATEGY_ID
REPORT = ROOT / "report/rnn_training_combination"
T1_REPORT = ROOT / "report/rnn_training/experiments/RNN-TRAIN-T1-FLIP"


def smoke() -> None:
    channels, prepared = context()
    validate_flip_cache()
    settings = strategy_settings(STRATEGY_ID)
    model = model_for(STRATEGY_ID)
    t1 = json.loads((T1_REPORT / "seed42/config.json").read_text(encoding="utf-8"))
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    recurrent = [module for module in model.modules() if isinstance(module, nn.RNN)]
    if (channels != tuple(range(190)) or settings != {
            "augmentation": "horizontal_flip", "dropout_probability": 0.0, "weight_decay": 5e-4}
            or ARCHITECTURES[ARCHITECTURE_ID] != {"embed_dim": 320, "rnn_hidden_size": 80, "num_blocks": 3}
            or count != t1["parameter_count"] or len(recurrent) != 6
            or any(module.nonlinearity != "tanh" or not module.bidirectional for module in recurrent)
            or any(isinstance(module, (nn.Dropout, nn.Conv2d, nn.GRU, nn.LSTM)) for module in model.modules())):
        raise AssertionError("T4 changed the frozen architecture or strategy")
    if (t1["augmentation"] != "horizontal_flip" or t1["weight_decay"] != 1e-4
            or t1["learning_rate"] != 3e-4):
        raise AssertionError("T1 reference changed")
    original = np.load(HANDCRAFTED / "features.npy", mmap_mode="r")
    flipped = np.load(FLIP_CACHE / "features.npy", mmap_mode="r")
    original_labels = np.load(HANDCRAFTED / "labels.npy", mmap_mode="r")
    flipped_labels = np.load(FLIP_CACHE / "labels.npy", mmap_mode="r")
    indices = prepared[42]["train_indices"]
    if (original.shape != flipped.shape or original.shape != (2000, 190, 7, 7)
            or not np.array_equal(original_labels[indices], flipped_labels[indices])):
        raise AssertionError("T4 original/flip feature alignment changed")
    mean, std = t1_normalization(indices)
    original_train = make_dataset(indices, channels, mean, std)
    flipped_train = CachedRepresentationDataset(FLIP_CACHE / "features.npy", FLIP_CACHE / "labels.npy",
                                                 indices, channels, mean, std)
    validation = make_dataset(prepared[42]["validation_indices"], channels, mean, std)
    if len(ConcatDataset((original_train, flipped_train))) != 3600 or len(validation) != 200:
        raise AssertionError("T4 sample counts changed")
    features = torch.stack([validation[index][0] for index in range(4)])
    labels = torch.as_tensor([validation[index][1] for index in range(4)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()(model(features), labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if optimizer.param_groups[0]["weight_decay"] != 5e-4:
        raise AssertionError("T4 optimizer weight decay changed")
    print(f"T4 smoke OK: 3600 train, 200 original validation, {count:,} parameters", flush=True)


def archive_run(seed: int) -> None:
    source = OUTPUT / f"seed{seed}"
    target = REPORT / "experiments" / STRATEGY_ID / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
        shutil.copy2(source / name, target / name)


def run_all() -> None:
    channels, prepared = context()
    validate_flip_cache()
    if json.loads((ROOT / "report/rnn_training/training_strategy_decision.json").read_text(
            encoding="utf-8"))["combination_candidates"] != ["RNN-TRAIN-T1-FLIP", "RNN-TRAIN-T3-WD"]:
        raise AssertionError("T4 combination was not authorized by RNN-TRAIN-001 results")
    for seed in SEEDS:
        directory = OUTPUT / f"seed{seed}"
        if (directory / "metrics.json").exists():
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            if (metrics.get("status") != "complete" or metrics.get("strategy_id") != STRATEGY_ID
                    or metrics.get("seed") != seed or metrics.get("split_sha256") != prepared[seed]["split"]["sha256"]):
                raise AssertionError(f"Existing T4 run incompatible: {directory}")
            archive_run(seed)
            print(f"Reused complete T4 seed {seed}", flush=True)
            continue
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"Partial T4 run exists: {directory}")
        directory.mkdir(parents=True)
        try:
            train_one(STRATEGY_ID, seed, prepared[seed], channels, directory, batch_id=BATCH_ID)
            archive_run(seed)
        except Exception as error:
            write_json(directory / "failure.json", {"strategy_id": STRATEGY_ID, "seed": seed,
                                                      "error": str(error), "traceback": traceback.format_exc()})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run-all", "summarize", "audit"))
    command = parser.parse_args().command
    if command in ("summarize", "audit"):
        from rnn_training_combination_report import audit, summarize
        {"summarize": summarize, "audit": audit}[command]()
    else:
        {"smoke": smoke, "run-all": run_all}[command]()


if __name__ == "__main__":
    main()
