"""Evaluate a saved checkpoint on the assignment's held-out test directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASS_TO_IDX, CatDogDataset, image_transform, list_samples
from models import create_model
from training import plot_confusion, run_epoch, select_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a cat/dog checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/val"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["class_to_idx"] != CLASS_TO_IDX:
        raise ValueError("Checkpoint class mapping does not match the dataset")

    model = create_model(checkpoint["model"], checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    samples = list_samples(args.data_dir)
    input_size = checkpoint["model_config"]["input_size"]
    loader = DataLoader(
        CatDogDataset(samples, image_transform(input_size, training=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    result = {
        "model": checkpoint["model"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "internal_validation_metrics": checkpoint["validation_metrics"],
        "test_dir": str(args.data_dir.resolve()),
        "test_metrics": metrics,
        "class_to_idx": CLASS_TO_IDX,
    }
    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "test_metrics.json", result)
    summary_path = output_dir / "train_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["test_metrics"] = metrics
        write_json(summary_path, summary)
    plot_confusion(
        metrics["confusion_matrix"],
        output_dir / "confusion_matrix.png",
        f"{checkpoint['model'].upper()} held-out test confusion matrix",
    )
    print(
        f"Test: {metrics['samples']} images, overall={metrics['accuracy']:.4f}, "
        f"cat={metrics['cat_accuracy']:.4f}, dog={metrics['dog_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
