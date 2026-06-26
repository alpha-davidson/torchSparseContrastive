# Edited: J. Gelina 06/26/26

#!/usr/bin/env python3
"""
extract_latents.py
===================
Load a trained SparseSimCLR checkpoint and extract latent vectors from the
O16 AT-TPC dataset, saving aligned feature + label arrays for downstream
evaluation with linear_probing.py.

Two extraction modes
--------------------
1. Split mode (default, recommended):
   Loads from the pre-split .npy files produced by O16_downstream_pipeline.py
   (O16_size512_train.npy, _val.npy, _test.npy). Labels are embedded in those
   files, so features and labels are always aligned. Use this for linear probing.

2. Raw mode (fallback):
   Loads directly from O16_w_event_keys.npy via O16Dataset. Used when split
   files don't exist yet. Labels.npy is saved as -1 placeholders.

Usage
-----
# Split mode — uses split files in data/ by default:
    python extract_latents.py --checkpoint checkpoints/best.pt

# Explicit split directory or sample size:
    python extract_latents.py --checkpoint checkpoints/best.pt \\
        --split-dir /path/to/data --sample-size 512

# Raw mode (no split files available):
    python extract_latents.py --checkpoint checkpoints/best.pt \\
        --data data/O16_w_event_keys.npy \\
        --lens data/O16_event_lens.npy \\
        --no-splits

# Custom output location:
    python extract_latents.py --checkpoint checkpoints/best.pt \\
        --output-dir embeddings/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from o16_dataset import O16Dataset
from sparse_simclr import sparse_simclr_21d, SparseSimCLR


# ---------------------------------------------------------------------------
# Split-file dataset
# ---------------------------------------------------------------------------

class _O16SplitDataset(Dataset):
    """
    Load one pre-split O16 .npy file (shape N, 512, 5) from
    O16_downstream_pipeline.py and voxelise each event for inference.

    Applies the same per-event normalisation as O16Dataset (training
    preprocessing) so features are consistent with what the backbone saw
    during training:
        xyz  — per-event min-max to [0, 1]
        q    — per-event min-max to [0, 1]
    """

    def __init__(self, path: Path, voxel_size: float):
        from torchsparse import SparseTensor
        from torchsparse.utils.quantize import sparse_quantize
        self._SparseTensor    = SparseTensor
        self._sparse_quantize = sparse_quantize
        self.voxel_size = voxel_size

        data = np.load(path)                      # (N, 512, 7): x,y,z,q,event_idx,label,ev_len
        self.labels = data[:, 0, 5].astype(np.int64)   # label stored at col 5, row 0
        self.events = data[:, :, :4]              # x, y, z, q  (all 512 rows)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        pts     = self.events[i]                  # (512, 4)
        nonzero = pts[pts[:, 0] != 0]             # strip padding zeros

        xyz = nonzero[:, :3].astype(np.float32)
        q   = nonzero[:, 3:4].astype(np.float32)

        # per-event normalise xyz to [0, 1]  (matches O16Dataset._load_event)
        lo  = xyz.min(axis=0, keepdims=True)
        hi  = xyz.max(axis=0, keepdims=True)
        rng = np.where((hi - lo) > 0, hi - lo, 1.0)
        xyz = (xyz - lo) / rng

        # per-event normalise amplitude to [0, 1]
        q_lo, q_hi = float(q.min()), float(q.max())
        q = (q - q_lo) / max(q_hi - q_lo, 1e-6)

        coords_q, idx = self._sparse_quantize(
            xyz, voxel_size=self.voxel_size, return_index=True
        )
        sparse = self._SparseTensor(
            feats=torch.tensor(q[idx],    dtype=torch.float32),
            coords=torch.tensor(coords_q, dtype=torch.int32),
        )
        return {"x": sparse, "label": torch.tensor(int(self.labels[i]), dtype=torch.long)}


def _collate_split(batch):
    from torchsparse.utils.collate import sparse_collate
    return {
        "x":     sparse_collate([b["x"]     for b in batch]),
        "label": torch.stack(  [b["label"] for b in batch]),
    }


def _collate_raw(batch):
    from torchsparse.utils.collate import sparse_collate
    return {"x": sparse_collate([b["original"] for b in batch])}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract per-event latent vectors from a trained SparseSimCLR checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Checkpoint saved by train_contrastive.py (e.g. checkpoints/best.pt)")
    p.add_argument("--config", type=Path, default=None,
                   help="run_config.json path. Defaults to <checkpoint_dir>/run_config.json.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to save latent_vectors.npy / labels.npy. "
                        "Defaults to <checkpoint_dir>.")

    # Split mode (default)
    p.add_argument("--split-dir", type=Path, default=Path("data"),
                   help="Directory containing O16_size{N}_{train,val,test}.npy files.")
    p.add_argument("--sample-size", type=int, default=None,
                   help="SAMPLE_SIZE used in O16_downstream_pipeline.py (default 512).")

    # Raw mode fallback
    p.add_argument("--no-splits", action="store_true",
                   help="Skip split files and extract from raw O16_w_event_keys.npy instead.")
    p.add_argument("--data", type=Path, default=None,
                   help="Path to O16_w_event_keys.npy (raw mode only).")
    p.add_argument("--lens", type=Path, default=None,
                   help="Path to O16_event_lens.npy (raw mode only).")
    p.add_argument("--min-hits", type=int, default=10,
                   help="Skip events shorter than this (raw mode; must match training).")

    # Model hyperparameters — read from run_config.json if not set
    p.add_argument("--voxel-size",      type=float, default=None)
    p.add_argument("--batch-size",      type=int,   default=None)
    p.add_argument("--num-workers",     type=int,   default=None)
    p.add_argument("--in-channels",     type=int,   default=None)
    p.add_argument("--proj-out-dim",    type=int,   default=None)
    p.add_argument("--proj-hidden-dim", type=int,   default=None)
    p.add_argument("--temperature",     type=float, default=None)
    p.add_argument("--final-bn",        action="store_true", default=None)
    return p


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in any unset args from run_config.json saved during training."""
    config_path = args.config or (args.checkpoint.parent / "run_config.json")

    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        print(f"Loaded training config: {config_path}")
        for key in ["voxel_size", "batch_size", "num_workers", "in_channels",
                    "proj_out_dim", "proj_hidden_dim", "temperature", "final_bn"]:
            if getattr(args, key) is None and key in cfg:
                setattr(args, key, cfg[key])
        if args.data is None and "data" in cfg:
            args.data = Path(cfg["data"])
        if args.lens is None and "lens" in cfg:
            args.lens = Path(cfg["lens"])
    else:
        print(f"No run_config.json at {config_path} — relying on CLI args / defaults.")

    defaults = {
        "voxel_size": 0.05, "batch_size": 16, "num_workers": 0,
        "in_channels": 1, "proj_out_dim": 128, "proj_hidden_dim": 512,
        "temperature": 0.1, "final_bn": False, "sample_size": 512,
    }
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    return args


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_from_splits(
    model: SparseSimCLR,
    split_dir: Path,
    sample_size: int,
    voxel_size: float,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    """Iterate train/val/test split files; return aligned (feats, labels)."""
    all_feats, all_labels = [], []

    for split in ("train", "val", "test"):
        path = split_dir / f"O16_size{sample_size}_{split}.npy"
        if not path.exists():
            print(f"  [{split}] not found at {path} — skipping.")
            continue

        ds     = _O16SplitDataset(path, voxel_size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=_collate_split, num_workers=num_workers)
        print(f"  {split}: {len(ds)} events, {len(loader)} batches")

        with torch.no_grad():
            for batch in loader:
                feats = model.encode(batch["x"].to(device))
                all_feats.append(feats.detach().cpu().numpy())
                all_labels.append(batch["label"].numpy())

    if not all_feats:
        raise FileNotFoundError(
            f"No split files found in {split_dir} matching "
            f"O16_size{sample_size}_{{train,val,test}}.npy.\n"
            f"Run O16_downstream_pipeline.py first, or use --no-splits."
        )

    return np.concatenate(all_feats), np.concatenate(all_labels)


def extract_from_raw(
    model: SparseSimCLR,
    data_path: Path,
    lens_path: Path,
    voxel_size: float,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    min_hits: int,
):
    """Extract from raw O16_w_event_keys.npy; returns feats only (no labels)."""
    ds = O16Dataset(
        data_path=str(data_path),
        lens_path=str(lens_path),
        voxel_size=voxel_size,
        min_hits=min_hits,
        aug_list=[],                              # no-op: use original view
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=_collate_raw,  num_workers=num_workers)
    print(f"Dataset: {len(ds)} valid events, {len(loader)} batches")

    all_feats = []
    with torch.no_grad():
        for batch in loader:
            feats = model.encode(batch["x"].to(device))
            all_feats.append(feats.detach().cpu().numpy())

    feats  = np.concatenate(all_feats)
    labels = np.full(len(feats), -1, dtype=np.int64)
    print("Raw mode — labels.npy saved as -1 placeholders (no split labels available).")
    return feats, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    args = resolve_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError(
            "No GPU detected. TorchSparse sparse downsampling is CUDA-only.\n"
            "Run on a GPU node:  srun --gres=gpu:1 --pty bash"
        )

    model = sparse_simclr_21d(
        in_channels=args.in_channels,
        proj_out_dim=args.proj_out_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        temperature=args.temperature,
        use_final_bn=bool(args.final_bn),
    ).to(device)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded weights from epoch {ckpt.get('epoch', '?')}")

    print("\nExtracting latent vectors...")
    if args.no_splits:
        if args.data is None or args.lens is None:
            raise ValueError("--no-splits requires --data and --lens.")
        feats, labels = extract_from_raw(
            model, args.data, args.lens, args.voxel_size, device,
            args.batch_size, args.num_workers, args.min_hits,
        )
    else:
        feats, labels = extract_from_splits(
            model, args.split_dir, args.sample_size, args.voxel_size, device,
            args.batch_size, args.num_workers,
        )

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    feats_path  = output_dir / "latent_vectors.npy"
    labels_path = output_dir / "labels.npy"
    np.save(feats_path,  feats)
    np.save(labels_path, labels)

    print(f"\nSaved features : {feats_path}   shape={feats.shape}")
    print(f"Saved labels   : {labels_path}  shape={labels.shape}")
    if len(np.unique(labels)) > 1:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution: { {int(k): int(v) for k, v in zip(unique, counts)} }")


if __name__ == "__main__":
    main()
