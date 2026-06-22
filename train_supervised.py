"""
Supervised shape classification on ShapeNet.

Supports two dataset formats:
  - shapenet_simple.pt      (400 samples, random 90/10 split)
  - shapenet_simple_large.pt (16881 samples, official train/val/test splits)

Usage
-----
# Small dataset (random split)
python train_supervised.py --data-path shapenet_simple.pt --num-classes 4

# Large dataset (official splits)
python train_supervised.py --data-path shapenet_simple_large.pt --num-classes 4

# Fine-tune from SimCLR checkpoint
python train_supervised.py --data-path shapenet_simple_large.pt --num-classes 4 \
    --pretrained checkpoints/checkpoint_best.pt
"""

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# torchsparse shadowing
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

try:
    repo_ts = ROOT_DIR / "torchsparse"
    if repo_ts.exists():
        saved = list(sys.path)
        sys.path = [p for p in sys.path if p and str(ROOT_DIR) not in p]
        try:
            import importlib as _il
            _installed = _il.import_module("torchsparse")
        except Exception:
            _installed = None
        finally:
            sys.path = saved
        if _installed is not None:
            sys.modules["torchsparse"] = _installed
except Exception:
    pass

_s0 = None
if sys.path and sys.path[0] == "":
    _s0 = sys.path.pop(0)
try:
    _spec = importlib.util.spec_from_file_location("local_model", str(ROOT_DIR / "model.py"))
    model_module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = model_module
    _spec.loader.exec_module(model_module)  # type: ignore[attr-defined]
finally:
    if _s0 is not None:
        sys.path.insert(0, _s0)

SparseResNet21D         = model_module.SparseResNet21D
_sparse_global_avg_pool = model_module._sparse_global_avg_pool
_BACKBONE_OUT_CHANNELS  = 128


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShapeNetDataset(Dataset):
    """
    Supports both shapenet_simple.pt and shapenet_simple_large.pt.

    For the large file, use split='train'/'val'/'test' to select the
    official ShapeNet split via the train_idx/val_idx/test_idx tensors.
    For the small file, split is ignored and all samples are returned
    (the train/val split is handled by Subset in build_data_loaders).
    """

    def __init__(
        self,
        path: str,
        voxel_size: float = 0.05,
        use_normals: bool = True,
        split: str = None,          # 'train' | 'val' | 'test' | None
    ):
        from torchsparse import SparseTensor
        from torchsparse.utils.quantize import sparse_quantize

        self._SparseTensor    = SparseTensor
        self._sparse_quantize = sparse_quantize

        raw = torch.load(path, map_location="cpu", weights_only=False)
        self.class_names  = raw["class_names"]
        self.voxel_size   = voxel_size
        self.use_normals  = use_normals
        self.has_official_splits = "train_idx" in raw

        if self.has_official_splits and split is not None:
            key = f"{split}_idx"
            idx = raw[key].long()
        else:
            idx = torch.arange(len(raw["labels"]))

        self.points  = raw["points"][idx]   # (N, 2048, 3)
        self.normals = raw["normals"][idx]  # (N, 2048, 3)
        self.labels  = raw["labels"][idx]   # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        pts = self.points[i]
        nrm = self.normals[i]
        lbl = int(self.labels[i])

        feats  = torch.cat([pts, nrm], dim=1).numpy() if self.use_normals else pts.numpy()
        coords = pts.numpy()

        coords_q, indices = self._sparse_quantize(
            coords, voxel_size=self.voxel_size, return_index=True
        )
        feats_q = feats[indices]

        sparse = self._SparseTensor(
            feats=torch.tensor(feats_q,  dtype=torch.float32),
            coords=torch.tensor(coords_q, dtype=torch.int32),
        )
        return {"x": sparse, "label": torch.tensor(lbl, dtype=torch.long)}


# keep old name as alias so visualize_embeddings.py still works
ShapeNetSimpleDataset = ShapeNetDataset


def collate_fn(batch):
    from torchsparse.utils.collate import sparse_collate
    return {
        "x":     sparse_collate([b["x"]     for b in batch]),
        "label": torch.stack(  [b["label"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Build loaders — auto-detects which file format is being used
# ---------------------------------------------------------------------------

def build_data_loaders(args):
    pin = args.num_workers > 0 and torch.cuda.is_available()

    # Probe the file to see if it has official splits
    raw      = torch.load(args.data_path, map_location="cpu", weights_only=False)
    has_splits = "train_idx" in raw
    del raw

    if has_splits:
        # --- large file: use official ShapeNet splits ---
        train_ds = ShapeNetDataset(args.data_path, args.voxel_size,
                                   use_normals=True, split="train")
        val_ds   = ShapeNetDataset(args.data_path, args.voxel_size,
                                   use_normals=True, split="val")
        test_ds  = ShapeNetDataset(args.data_path, args.voxel_size,
                                   use_normals=True, split="test")
        print(f"Official splits  —  train: {len(train_ds)}  "
              f"val: {len(val_ds)}  test: {len(test_ds)}")
    else:
        # --- small file: random 70/15/15 split ---
        full_ds  = ShapeNetDataset(args.data_path, args.voxel_size, use_normals=True)
        n        = len(full_ds)
        val_size  = max(1, int(n * 0.15))
        test_size = max(1, int(n * 0.15))
        indices   = list(range(n))
        random.shuffle(indices)
        test_ds  = Subset(full_ds, indices[:test_size])
        val_ds   = Subset(full_ds, indices[test_size:test_size + val_size])
        train_ds = Subset(full_ds, indices[test_size + val_size:])
        print(f"Random 70/15/15  —  train: {len(train_ds)}  "
              f"val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=pin)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SparseClassifier(nn.Module):
    def __init__(self, num_classes, in_channels=6, freeze_backbone=False):
        super().__init__()
        self.backbone   = SparseResNet21D(in_channels=in_channels)
        self.classifier = nn.Linear(_BACKBONE_OUT_CHANNELS, num_classes)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        feature_maps = self.backbone(x)
        h = _sparse_global_avg_pool(feature_maps[-1])
        return self.classifier(h)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_simclr_backbone(model, ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    backbone_state = {k[len("backbone."):]: v
                      for k, v in state.items() if k.startswith("backbone.")}
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"  [backbone] missing  : {missing[:5]}")
    if unexpected:
        print(f"  [backbone] unexpected: {unexpected[:5]}")
    print(f"Loaded backbone from '{ckpt_path}'")


def save_checkpoint(state, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / name)
    print(f"Saved: {output_dir / name}")


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    start = ckpt.get("epoch", 0) + 1
    best  = ckpt.get("best_val_acc", 0.0)
    print(f"Resumed from '{path}' (epoch {start}, best_val_acc={best:.4f})")
    return start, best


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, amp_enabled):
    model.eval()
    total_loss = total_correct = total_n = 0
    with torch.no_grad():
        for batch in loader:
            x, labels = batch["x"].to(device), batch["label"].to(device)
            with torch.amp.autocast(device, enabled=amp_enabled):
                logits = model(x)
                loss   = F.cross_entropy(logits, labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n       += labels.size(0)
            total_loss    += loss.item()
    return total_loss / max(1, len(loader)), total_correct / max(1, total_n)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, test_loader = build_data_loaders(args)

    # infer num classes from dataset if not overridden
    sample_ds = train_loader.dataset
    while isinstance(sample_ds, Subset):
        sample_ds = sample_ds.dataset
    class_names = sample_ds.class_names
    num_classes = args.num_classes or len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    model = SparseClassifier(
        num_classes=num_classes,
        in_channels=6,
        freeze_backbone=args.freeze_backbone,
    ).to(args.device)

    if args.pretrained:
        load_simclr_backbone(model, args.pretrained)

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(),   "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler(device=args.device, enabled=args.amp)

    start_epoch  = 1
    best_val_acc = 0.0
    if args.resume:
        start_epoch, best_val_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = epoch_correct = epoch_n = 0

        for batch in train_loader:
            x, labels = batch["x"].to(args.device), batch["label"].to(args.device)
            optimizer.zero_grad()
            with torch.amp.autocast(args.device, enabled=args.amp):
                logits = model(x)
                loss   = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_correct += (logits.detach().argmax(1) == labels).sum().item()
            epoch_n       += labels.size(0)
            epoch_loss    += loss.item()

        scheduler.step()

        train_acc = epoch_correct / max(1, epoch_n)
        avg_loss  = epoch_loss    / max(1, len(train_loader))
        val_loss, val_acc = evaluate(model, val_loader, args.device, args.amp)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={avg_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        is_interval = (epoch % args.save_interval == 0)
        is_best     = (val_acc > best_val_acc)
        if is_interval or is_best:
            if is_best:
                best_val_acc = val_acc
            ckpt = {
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val_acc": best_val_acc,
                "class_names": class_names, "args": vars(args),
            }
            if is_interval:
                save_checkpoint(ckpt, args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            if is_best:
                save_checkpoint(ckpt, args.output_dir, "checkpoint_best.pt")

    # --- final test set evaluation ---
    print("\n--- Test Set Evaluation ---")
    test_loss, test_acc = evaluate(model, test_loader, args.device, args.amp)
    print(f"Test loss: {test_loss:.4f}  Test acc: {test_acc:.4f}")
    print(f"\nDone. Best val acc: {best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",     type=str,   default="shapenet_simple_large.pt")
    p.add_argument("--output-dir",    type=str,   default="./checkpoints_supervised")
    p.add_argument("--num-classes",   type=int,   default=None,
                   help="Inferred from dataset if not set.")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--backbone-lr",   type=float, default=1e-4)
    p.add_argument("--voxel-size",    type=float, default=0.05)
    p.add_argument("--num-workers",   type=int,   default=0)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--save-interval", type=int,   default=10)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp",           action="store_true")
    p.add_argument("--pretrained",    type=str,   default="")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--resume",        type=str,   default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())