"""Compare two wider RNN2D models under the frozen flip training recipe."""

from __future__ import annotations

import argparse
import json
import shutil
import traceback

import numpy as np
import torch
from torch import nn

from cnn_training_strategy_experiments import FLIP_CACHE, t1_normalization, validate_flip_cache
from models.rnn2d import ARCHITECTURES, VisionRNN2DClassifier
from representations.cache import ROOT
from rnn2d_architecture_experiments import context
from rnn_training_strategy_experiments import ARTIFACTS, SEEDS, train_one
from training import write_json

BATCH_ID = "RNN-ARCH-003"
BASE = "RNN2D-ARCH-BASE"
NEW_ARCHITECTURES = ("RNN2D-ARCH-WIDE", "RNN2D-ARCH-XWIDE")
ALL_ARCHITECTURES = (BASE, *NEW_ARCHITECTURES)
BASE_REGISTRY_ID = "RNN2D-ARCH-L"
STRATEGY_ID = "RNN-TRAIN-T1-FLIP"
OUTPUT = ROOT / "outputs/rnn_architecture" / BATCH_ID
REPORT = ROOT / "report/rnn_architecture_scaling"
BASE_REPORT = ROOT / "report/rnn_training/experiments" / STRATEGY_ID


def smoke() -> None:
    channels, prepared = context()
    validate_flip_cache()
    source = np.load(FLIP_CACHE / "features.npy", mmap_mode="r")
    if channels != tuple(range(190)) or source.shape != (2000, 190, 7, 7):
        raise AssertionError("Frozen REP-006 flip cache changed")
    specs = {
        BASE_REGISTRY_ID: {"embed_dim": 320, "rnn_hidden_size": 80, "num_blocks": 3},
        "RNN2D-ARCH-WIDE": {"embed_dim": 384, "rnn_hidden_size": 96, "num_blocks": 3},
        "RNN2D-ARCH-XWIDE": {"embed_dim": 448, "rnn_hidden_size": 112, "num_blocks": 3},
    }
    counts = {}
    for architecture_id, spec in specs.items():
        model = VisionRNN2DClassifier(architecture_id)
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        counts[architecture_id] = count
        recurrent = [module for module in model.modules() if isinstance(module, nn.RNN)]
        if (ARCHITECTURES[architecture_id] != spec or spec["rnn_hidden_size"] * 4 != spec["embed_dim"]
                or len(model.blocks) != 3 or len(recurrent) != 6
                or any(module.nonlinearity != "tanh" or not module.bidirectional
                       or not module.batch_first or module.input_size != spec["embed_dim"]
                       or module.hidden_size != spec["rnn_hidden_size"] for module in recurrent)
                or any(isinstance(module, (nn.Dropout, nn.Conv2d, nn.LSTM, nn.GRU))
                       for module in model.modules())):
            raise AssertionError(f"RNN2D width topology changed: {architecture_id}")
        for block in model.blocks:
            if block.horizontal_rnn is block.vertical_rnn:
                raise AssertionError("The two RNN axes share parameters")
        features = torch.randn(2, 190, 7, 7)
        tokens = model.projection(features)
        if tokens.shape != (2, 7, 7, spec["embed_dim"]):
            raise AssertionError("Token projection shape changed")
        for block in model.blocks:
            horizontal, _ = block.horizontal_rnn(torch.randn(2 * 7, 7, spec["embed_dim"]))
            vertical, _ = block.vertical_rnn(torch.randn(2 * 7, 7, spec["embed_dim"]))
            if (horizontal.shape != (14, 7, 2 * spec["rnn_hidden_size"])
                    or vertical.shape != horizontal.shape):
                raise AssertionError("RNN axis output shape changed")
            tokens = block(tokens)
            if tokens.shape != (2, 7, 7, spec["embed_dim"]):
                raise AssertionError("Axis fusion shape changed")
        logits = model(features)
        if logits.shape != (2, 2):
            raise AssertionError("Classifier output shape changed")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        nn.CrossEntropyLoss()(logits, torch.tensor([0, 1])).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    baseline = json.loads((BASE_REPORT / "seed42/metrics.json").read_text(encoding="utf-8"))
    if (counts[BASE_REGISTRY_ID] != baseline["parameter_count"]
            or not counts[BASE_REGISTRY_ID] < counts[NEW_ARCHITECTURES[0]] < counts[NEW_ARCHITECTURES[1]]):
        raise AssertionError("Parameter count ordering or historical BASE changed")
    for seed in SEEDS:
        mean, std = t1_normalization(prepared[seed]["train_indices"])
        baseline_norm = json.loads((BASE_REPORT / f"seed{seed}/normalization.json").read_text(encoding="utf-8"))
        if (not np.array_equal(mean, np.asarray(baseline_norm["mean"], dtype=np.float32))
                or not np.array_equal(std, np.asarray(baseline_norm["std"], dtype=np.float32))):
            raise AssertionError("T1 training-only normalization changed")
    print("WIDTH smoke OK: " + ", ".join(f"{key}={value:,}" for key, value in counts.items()), flush=True)


def archive_run(architecture_id: str, seed: int) -> None:
    source = OUTPUT / architecture_id / f"seed{seed}"
    target = REPORT / "experiments" / architecture_id / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
        shutil.copy2(source / name, target / name)


def run_all() -> None:
    channels, prepared = context()
    validate_flip_cache()
    decision = json.loads((ROOT / "report/rnn_training_combination/combination_decision.json").read_text(
        encoding="utf-8"))
    if (decision["selected_training_recipe"] != STRATEGY_ID
            or decision["frozen_recipe"]["weight_decay"] != 1e-4):
        raise AssertionError("Frozen T1 recipe changed")
    for architecture_id in NEW_ARCHITECTURES:
        for seed in SEEDS:
            directory = OUTPUT / architecture_id / f"seed{seed}"
            if (directory / "metrics.json").exists():
                metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
                if (metrics.get("status") != "complete" or metrics.get("architecture_id") != architecture_id
                        or metrics.get("seed") != seed or metrics.get("split_sha256") !=
                        prepared[seed]["split"]["sha256"]):
                    raise AssertionError(f"Existing width run incompatible: {directory}")
                archive_run(architecture_id, seed)
                print(f"Reused {architecture_id} seed {seed}", flush=True)
                continue
            if directory.exists() and any(directory.iterdir()):
                raise RuntimeError(f"Partial width run exists: {directory}")
            directory.mkdir(parents=True)
            try:
                train_one(STRATEGY_ID, seed, prepared[seed], channels, directory,
                          batch_id=BATCH_ID, architecture_id=architecture_id,
                          experiment_subject=architecture_id)
                archive_run(architecture_id, seed)
            except Exception as error:
                write_json(directory / "failure.json", {"architecture_id": architecture_id, "seed": seed,
                                                          "error": str(error), "traceback": traceback.format_exc()})
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run-all", "summarize", "audit"))
    command = parser.parse_args().command
    if command in ("summarize", "audit"):
        from rnn_architecture_scaling_report import audit, summarize
        {"summarize": summarize, "audit": audit}[command]()
    else:
        {"smoke": smoke, "run-all": run_all}[command]()


if __name__ == "__main__":
    main()
