"""
scripts/train.py  — Day 2

Training script for ExamSentinelNet.
Optimised for RTX 3060 6GB:
  - Mixed precision (AMP) → ~2x speedup, ~50% VRAM reduction
  - Gradient accumulation (effective batch = 64 with batch_size=16)
  - OneCycleLR scheduler
  - Gradient clipping
  - Automatic checkpoint saving

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --resume models/weights/best.pt
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from core.model import ExamSentinelNet, ExamSentinelLoss
from core.dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train ExamSentinelNet")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--data", default="data", help="Data root directory")
    parser.add_argument("--output", default="models/weights", help="Output dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--debug", action="store_true", help="Run 2 batches only")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(all_preds: Dict[str, list],
                    all_labels: np.ndarray) -> Dict[str, float]:
    """Compute AUC per head."""
    metrics = {}
    head_map = {
        "deepfake": 0, "recapture": 1, "splicing": 2, "forgery": 3
    }
    for head, col in head_map.items():
        targets = all_labels[:, col]
        if targets.sum() == 0 or targets.sum() == len(targets):
            metrics[f"auc_{head}"] = 0.5
            continue
        try:
            preds = np.array(all_preds[head])
            auc = roc_auc_score(targets, preds)
            metrics[f"auc_{head}"] = auc
        except Exception:
            metrics[f"auc_{head}"] = 0.5

    # Mean AUC
    aucs = [v for k, v in metrics.items() if k.startswith("auc_")]
    metrics["auc_mean"] = np.mean(aucs) if aucs else 0.5
    return metrics


def train_epoch(model, loader, optimizer, loss_fn, scaler, device,
                grad_accum_steps=4, debug=False):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        if debug and step >= 2:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixed precision forward
        with autocast():
            predictions = model(images)
            losses = loss_fn(predictions, labels)
            loss = losses["loss_total"] / grad_accum_steps

        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += losses["loss_total"].item()

        if step % 50 == 0:
            print(f"  Step {step:4d} | loss: {losses['loss_total'].item():.4f} "
                  f"| deepfake: {losses.get('loss_deepfake', torch.tensor(0)).item():.3f} "
                  f"| recapture: {losses.get('loss_recapture', torch.tensor(0)).item():.3f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device, debug=False):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = {k: [] for k in ["deepfake", "recapture", "splicing", "forgery"]}

    for step, (images, labels) in enumerate(loader):
        if debug and step >= 2:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            predictions = model(images)
            losses = loss_fn(predictions, labels)

        total_loss += losses["loss_total"].item()
        all_labels.append(labels.cpu().numpy())

        for head in all_preds:
            if head in predictions:
                probs = torch.sigmoid(predictions[head]).cpu().numpy()
                all_preds[head].extend(probs.flatten().tolist())

    all_labels = np.vstack(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["val_loss"] = total_loss / max(len(loader), 1)
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }, path)
    print(f"  Saved checkpoint → {path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Override epochs if specified
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    os.makedirs(args.output, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device(
        cfg["inference"]["device"]
        if torch.cuda.is_available() else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"ExamSentinel Training")
    print(f"Device:     {device}")
    if device.type == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
        print(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Config:     {args.config}")
    print(f"{'='*60}\n")

    # ── Data
    train_loader, val_loader = get_dataloaders(
        data_root=args.data,
        img_size=cfg["model"]["img_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"]
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # ── Model
    model = ExamSentinelNet(
        img_size=cfg["model"]["img_size"],
        pretrained=True
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}\n")

    # ── Loss
    loss_fn = ExamSentinelLoss()

    # ── Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # ── Scheduler: OneCycleLR
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["training"]["lr"] * 10,
        total_steps=total_steps,
        pct_start=0.1
    )

    # ── Mixed precision scaler
    scaler = GradScaler()

    # ── Resume
    start_epoch = 0
    best_auc = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt["metrics"].get("auc_mean", 0.0)
        print(f"Resumed from epoch {start_epoch}, best AUC: {best_auc:.4f}\n")

    # ── TensorBoard
    writer = SummaryWriter("logs/tensorboard")

    # ── Training loop
    history = []
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        t0 = time.time()
        print(f"\nEpoch {epoch+1}/{cfg['training']['epochs']}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            grad_accum_steps=4, debug=args.debug
        )

        # Validate
        val_metrics = val_epoch(model, val_loader, loss_fn, device,
                                debug=args.debug)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  train_loss: {train_loss:.4f}")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  time: {elapsed:.1f}s")

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Save best
        if val_metrics["auc_mean"] > best_auc:
            best_auc = val_metrics["auc_mean"]
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                f"{args.output}/best.pt"
            )
            print(f"  ★ New best AUC: {best_auc:.4f}")

        # Save latest
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            f"{args.output}/latest.pt"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics
        })

        if args.debug:
            print("\n[DEBUG MODE] Stopping after 1 epoch.")
            break

    # Save history
    with open("logs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete. Best AUC: {best_auc:.4f}")
    print(f"Weights saved to: {args.output}/best.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
