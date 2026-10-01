"""Train an assignment model using only data/train for selection."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASS_TO_IDX, CatDogDataset, image_transform, list_samples, stratified_split
from models import create_model
from training import (
    plot_history,
    run_epoch,
    seed_everything,
    seed_worker,
    select_device,
    write_history,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cat/dog classifier")
    parser.add_argument("--model", choices=["dnn"], default="dnn")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=[256, 64])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, help="Limit batches for a smoke test")
    parser.add_argument("--max-validation-batches", type=int, help="Limit batches for a smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.input_size < 1:
        raise ValueError("Epochs, patience, batch size, and input size must be positive")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("max-train-batches must be positive")
    if args.max_validation_batches is not None and args.max_validation_batches < 1:
        raise ValueError("max-validation-batches must be positive")

    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = args.output_dir or Path("outputs") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = list_samples(args.data_dir / "train")
    train_samples, validation_samples = stratified_split(
        all_samples, args.validation_fraction, args.seed
    )
    write_json(
        output_dir / "split.json",
        {
            "seed": args.seed,
            "class_to_idx": CLASS_TO_IDX,
            "train": [path.name for path, _ in train_samples],
            "internal_validation": [path.name for path, _ in validation_samples],
        },
    )
    pin_memory = device.type == "cuda"
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        CatDogDataset(train_samples, image_transform(args.input_size, training=True)),
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        CatDogDataset(validation_samples, image_transform(args.input_size, training=False)),
        shuffle=False,
        **loader_kwargs,
    )

    model_config = {
        "input_size": args.input_size,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
    }
    model = create_model(args.model, model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {
        "model": args.model,
        "data_dir": str(args.data_dir.resolve()),
        "epochs_max": args.epochs,
        "patience": args.patience,
        "validation_fraction": args.validation_fraction,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "device": str(device),
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss",
        "train_augmentation": "RandomHorizontalFlip(p=0.5)",
        "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
        "model_config": model_config,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
    }
    write_json(output_dir / "config.json", config)
    print(
        f"Model={args.model}, device={device}, parameters={parameter_count:,}, "
        f"train={len(train_samples)}, internal_validation={len(validation_samples)}",
        flush=True,
    )

    history: list[dict] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    lowest_validation_loss = float("inf")
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, args.max_train_batches
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device,
            max_batches=args.max_validation_batches,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_cat_accuracy": validation_metrics["cat_accuracy"],
            "validation_dog_accuracy": validation_metrics["dog_accuracy"],
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)

        accuracy = validation_metrics["accuracy"]
        loss = validation_metrics["loss"]
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_accuracy, best_loss = accuracy, loss
            torch.save(
                {
                    "model": args.model,
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "class_to_idx": CLASS_TO_IDX,
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "training_config": config,
                },
                output_dir / "best_model.pt",
            )

        if loss < lowest_validation_loss - 1e-4:
            lowest_validation_loss = loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"Epoch {epoch:02d}: train loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}; "
            f"internal val loss={loss:.4f}, acc={accuracy:.4f}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs", flush=True)
            break

    duration = time.perf_counter() - started
    checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    summary = {
        "model": args.model,
        "parameter_count": parameter_count,
        "train_samples": len(train_samples),
        "internal_validation_samples": len(validation_samples),
        "epochs_run": len(history),
        "best_epoch": checkpoint["epoch"],
        "best_validation_metrics": checkpoint["validation_metrics"],
        "training_seconds": duration,
        "device": str(device),
        "test_metrics": "pending",
    }
    write_json(output_dir / "train_summary.json", summary)
    plot_history(history, output_dir / "training_curves.png", args.model)
    print(f"Best epoch={summary['best_epoch']}; results saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
