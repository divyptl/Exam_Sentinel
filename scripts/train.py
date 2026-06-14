"""
scripts/train.py — Training loop for ExamSentinelNet.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --debug
"""
import sys, os, time, json, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch, torch.optim as optim
import numpy as np
import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data",   default="data")
    p.add_argument("--output", default="models/weights")
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--debug",  action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    device = torch.device(cfg["inference"]["device"]
                          if torch.cuda.is_available() else "cpu")
    print(f"\nExamSentinel Training\nDevice: {device}")

    from core.model import ExamSentinelNet, ExamSentinelLoss
    from core.dataset import get_dataloaders

    train_loader, val_loader = get_dataloaders(
        args.data, cfg["model"]["img_size"],
        cfg["training"]["batch_size"], cfg["training"]["num_workers"]
    )

    if len(train_loader.dataset) == 0:
        print("\n[!] No training data found. Download datasets per data/README.md")
        print("    Running in debug/dry-run mode with synthetic data...")
        if not args.debug:
            return

    model   = ExamSentinelNet(img_size=cfg["model"]["img_size"], pretrained=True).to(device)
    loss_fn = ExamSentinelLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])

    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        use_amp = True
    except Exception:
        scaler = None
        use_amp = False

    Path(args.output).mkdir(parents=True, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print(f"Model params: {model.count_parameters():,}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print("Starting training...\n")

    best_loss = float("inf")
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            if args.debug and i >= 2:
                break
            imgs   = imgs.to(device)
            labels = labels.to(device)
            if use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    preds  = model(imgs)
                    losses = loss_fn(preds, labels)
                scaler.scale(losses["loss_total"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds  = model(imgs)
                losses = loss_fn(preds, labels)
                losses["loss_total"].backward()
                optimizer.step()
            optimizer.zero_grad()
            total_loss += losses["loss_total"].item()

        avg = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} | loss: {avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "loss": avg}, f"{args.output}/best.pt")
            print(f"  ★ Saved best (loss={best_loss:.4f})")

        if args.debug:
            print("[DEBUG] Stopping after 1 epoch.")
            break

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Weights: {args.output}/best.pt")

if __name__ == "__main__":
    main()
