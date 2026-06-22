#!/usr/bin/env python3
"""
extract_latents.py
===================
Load a trained SparseSimCLR checkpoint and extract a latent vector AND its
class label for every sample in the (supervised) ShapeNet dataset, saved as
two aligned numpy arrays — ready to feed straight into linear_probing.py:

    python linear_probing.py --name O16 latent_vectors.npy labels.npy

Uses the same ShapeNetDataset / collate_fn as f1_contrastive.py (raw,
unaugmented samples with paired "x"/"label"), so features and labels are
guaranteed to stay aligned — both are pulled from the same batch in the
same loop iteration.

Usage
-----
# Uses checkpoints/run_config.json (saved by train_contrastive.py) to
# automatically match model hyperparameters to the checkpoint:
    python extract_latents.py --checkpoint checkpoints/best.pt

# Override the dataset path or any model hyperparameter explicitly:
    python extract_latents.py \
        --checkpoint checkpoints/best.pt \
        --data shapenet_simple_large.pt \
        --proj-out-dim 128 \
        --proj-hidden-dim 512

# Custom output location:
    python extract_latents.py --checkpoint checkpoints/best.pt \
        --output-dir embeddings/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from train_supervised import ShapeNetDataset, collate_fn
from sparse_simclr import sparse_simclr_21d, SparseSimCLR


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract per-sample latent vectors + labels from a trained SparseSimCLR checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a checkpoint saved by train_contrastive.py (e.g. checkpoints/best.pt)")
    p.add_argument("--data", type=Path, default=None,
                   help="Dataset .pt file. If omitted, read from run_config.json next to the checkpoint.")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to run_config.json. Defaults to <checkpoint_dir>/run_config.json.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to save latent_vectors.npy / labels.npy. Defaults to <checkpoint_dir>.")

    # Optional overrides; if not passed, values come from run_config.json
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
    else:
        print(f"No run_config.json found at {config_path} — relying on CLI args / defaults.")

    defaults = {
        "voxel_size": 0.05, "batch_size": 16, "num_workers": 0,
        "in_channels": 3, "proj_out_dim": 128, "proj_hidden_dim": 512,
        "temperature": 0.1, "final_bn": False,
    }
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    if args.data is None:
        raise ValueError("Could not determine dataset path. Pass --data explicitly.")

    return args


# ---------------------------------------------------------------------------
# Latent + label extraction
# ---------------------------------------------------------------------------

def extract_dataset_latents(
    model: SparseSimCLR,
    data_path: Path,
    voxel_size: float,
    use_normals: bool,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """
    Run the trained encoder over every sample across the train/val/test
    splits of ShapeNetDataset (raw, unaugmented samples) and return:
        feats      : (num_samples, embed_dim) numpy array
        labels     : (num_samples,) numpy array of integer class labels
        class_names: list of class name strings
    Features and labels are extracted from the same batch in the same loop
    iteration, so they stay aligned by construction.
    """
    model.eval()

    all_feats, all_labels = [], []
    class_names = None

    for split in ("train", "val", "test"):
        ds = ShapeNetDataset(data_path, voxel_size=voxel_size,
                             split=split, use_normals=use_normals)
        if class_names is None:
            class_names = ds.class_names
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)
        print(f"  {split}: {len(ds)} samples, {len(loader)} batches")

        with torch.no_grad():
            for batch in loader:
                x      = batch["x"].to(device)
                labels = batch["label"]

                # model.encode() runs the backbone and applies the correct
                # global average pooling over the last stage's feature maps,
                # returning one (backbone_out_channels,) vector per sample.
                feats = model.encode(x)

                all_feats.append(feats.detach().cpu().numpy())
                all_labels.append(labels.numpy())

    feats  = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return feats, labels, class_names


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
            "No GPU detected (torch.cuda.is_available() is False).\n"
            "This TorchSparse build only implements sparse downsampling on CUDA, "
            "so backbone inference will fail with a NotImplementedError on CPU.\n"
            "Run this on a GPU node, the same way you launch training jobs, e.g.:\n"
            "  srun --gres=gpu:1 --pty bash\n"
            "  python extract_latents.py --checkpoint checkpoints/best.pt\n"
            "or submit it via sbatch."
        )

    # rebuild model architecture, then load trained weights
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

    # in_channels==3 -> xyz only; in_channels==6 -> xyz + normals.
    # Adjust here if your backbone was trained with a different convention.
    use_normals = args.in_channels == 6

    print("\nExtracting latent vectors + labels for the full dataset...")
    feats, labels, class_names = extract_dataset_latents(
        model, args.data, args.voxel_size, use_normals, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    feats_path  = output_dir / "latent_vectors.npy"
    labels_path = output_dir / "labels.npy"
    np.save(feats_path, feats)
    np.save(labels_path, labels)

    print(f"\nSaved features: {feats_path}  shape={feats.shape}")
    print(f"Saved labels:   {labels_path}  shape={labels.shape}")
    print(f"Classes ({len(class_names)}): {class_names}")


if __name__ == "__main__":
    main()