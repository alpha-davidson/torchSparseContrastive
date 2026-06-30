# Edited: J. Gelina 06/22/26

#!/usr/bin/env python3
"""
train_contrastive.py
====================
Train SparseSimCLR (SimCLR / NT-Xent) on the TorchSparse contrastive ShapeNet dataset.

Usage
-----
# Basic run:
    python train_contrastive.py --data shapenet_simple_large.pt

# Full GPU run:
    python train_contrastive.py \
        --data shapenet_simple_large.pt \
        --epochs 100 \
        --batch-size 16 \
        --lr 3e-4 \
        --temperature 0.1

# Resume from checkpoint:
    python train_contrastive.py --data shapenet_simple_large.pt \
        --resume checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from o16_dataset import make_o16_dataloader
from sparse_simclr import sparse_simclr_21d, SparseSimCLR


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train SparseSimCLR on TorchSparse contrastive ShapeNet data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, required=True,
                   help="Path to O16_w_event_keys.npy")
    p.add_argument("--lens", type=Path, default=Path("data/O16_event_lens.npy"),
                   help="Path to O16_event_lens.npy")
    p.add_argument("--voxel-size",       type=float, default=0.05)
    p.add_argument("--num-workers",      type=int,   default=0)
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch-size",       type=int,   default=16)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight-decay",     type=float, default=1e-4)
    p.add_argument("--grad-clip",        type=float, default=1.0)
    p.add_argument("--in-channels",      type=int,   default=1)
    p.add_argument("--proj-out-dim",     type=int,   default=128)
    p.add_argument("--proj-hidden-dim",  type=int,   default=512)
    p.add_argument("--temperature",      type=float, default=0.1)
    p.add_argument("--final-bn",         action="store_true")
    p.add_argument("--save-dir",         type=Path,  default=Path("checkpoints"))
    p.add_argument("--save-every",       type=int,   default=10)
    p.add_argument("--resume",           type=Path,  default=None)
    return p


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: SparseSimCLR,
    loader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        view_a = batch["view_a"].to(device)
        view_b = batch["view_b"].to(device)

        optimizer.zero_grad()
        loss, _, _ = model(view_a, view_b)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, epoch, model, optimizer, scheduler, history):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history":         history,
    }, path)
    print(f"Saved: {path}")


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"Resumed from epoch {ckpt['epoch']} ({path})")
    return ckpt["epoch"] + 1, ckpt.get("history", [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # dataloader
    loader = make_o16_dataloader(
        data_path=str(args.data),
        lens_path=str(args.lens),
        batch_size=args.batch_size,
        voxel_size=args.voxel_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    print(f"\nDataset: {len(loader.dataset)} events")
    print(f"Train batches: {len(loader)}  "
          f"(batch_size={args.batch_size})\n")

    # model
    model = sparse_simclr_21d(
        in_channels=args.in_channels,
        proj_out_dim=args.proj_out_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        temperature=args.temperature,
        use_final_bn=args.final_bn,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    history: list[dict] = []
    if args.resume is not None:
        start_epoch, history = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )

    best_loss = float("inf")
    print(f"{'Epoch':>6}  {'train_loss':>12}  {'lr':>10}  {'time':>7}")
    print("-" * 44)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, loader, optimizer, device)
        scheduler.step()

        lr_now    = scheduler.get_last_lr()[0]
        elapsed   = time.time() - t0

        print(f"{epoch:3d}/{args.epochs}  "
              f"train_loss={avg_loss:.4f}  "
              f"lr={lr_now:.2e}  "
              f"{elapsed:6.1f}s")

        history.append({"epoch": epoch, "avg_loss": avg_loss, "lr": lr_now})

        is_best     = avg_loss < best_loss
        is_interval = epoch % args.save_every == 0

        if is_best:
            best_loss = avg_loss
            save_checkpoint(args.save_dir / "best.pt",
                            epoch, model, optimizer, scheduler, history)
        if is_interval:
            save_checkpoint(args.save_dir / f"epoch_{epoch:03d}.pt",
                            epoch, model, optimizer, scheduler, history)

    # final checkpoint + loss log
    save_checkpoint(args.save_dir / "final.pt",
                    args.epochs, model, optimizer, scheduler, history)
    (args.save_dir / "loss_history.json").write_text(
        json.dumps(history, indent=2)
    )
    print(f"\nDone. Best avg loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()