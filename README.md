# SparseSimCLR — Contrastive Learning for 3D Point Clouds

Self-supervised representation learning on voxelized 3D point clouds using SimCLR and a sparse 3D ResNet backbone ([TorchSparse](https://github.com/mit-han-lab/torchsparse)). Trained without labels; representations are evaluated via linear probing and supervised fine-tuning.

---

## Model Architecture

```
Point Cloud (N × 3)
        │
        ├── augmentation pipeline A ──→ voxelize ──→ SparseTensor (view_a)
        └── augmentation pipeline B ──→ voxelize ──→ SparseTensor (view_b)

view_a ──→ SparseResNet21D backbone ──→ global avg pool ──→ h_a (B × 128)
                                                                 │
                                                         ProjectionHead (MLP)
                                                                 │
                                                         z_a  (B × proj_out_dim)

view_b  [same path] ──→ z_b (B × proj_out_dim)

NT-Xent loss(z_a, z_b)
```

**Forward pass (`SparseSimCLR.forward`):**
1. Both views are passed independently through the same backbone (`SparseResNet21D`).
2. The last stage's sparse feature map is pooled to a dense vector via `_sparse_global_avg_pool`.
3. Each pooled vector is projected to a lower-dimensional space by `ProjectionHead` and L2-normalised.
4. `NTXentLoss` (normalised temperature-scaled cross-entropy) is computed over the batch of positive pairs.

---

## File Reference

### Core Model — `model.py`
Defines the full SparseSimCLR architecture.

| Symbol | Purpose |
|---|---|
| `_sparse_global_avg_pool` | Scatter-averages all occupied voxel features into one `(B, C)` dense tensor |
| `ProjectionHead` | Two-layer MLP (Linear → BN → ReLU → Linear → optional BN) that maps backbone features to the contrastive embedding space |
| `NTXentLoss` | NT-Xent loss: builds a `(2B, 2B)` cosine-similarity matrix, masks self-pairs, and computes cross-entropy with positive-pair targets |
| `SparseSimCLR` | Top-level module; owns the backbone, projector, and criterion; exposes `encode()` (backbone + pool) and `project()` (encode + project + normalise) |
| `sparse_simclr_21d()` | Convenience constructor: builds `SparseResNet21D` + `SparseSimCLR` with one call |

### Re-export Shim — `sparse_simclr.py`
Thin module that loads `model.py` via `importlib` and re-exports `SparseSimCLR` and `sparse_simclr_21d`. Allows training scripts to do `from sparse_simclr import ...` regardless of whether `model.py` is on the Python path.

### Augmentations — `augmentations.py`
All augmentations operate on raw `(N, 3)` numpy point clouds and return a copy. Applied twice independently to the same point cloud to produce the two contrastive views.

| Class | Effect |
|---|---|
| `RandomRotation` | Random rotation around one or more axes; `axes='y'` for upright objects, `'xyz'` for SO(3) |
| `RandomJitter` | Per-point Gaussian noise (clipped) — simulates sensor noise |
| `RandomScale` | Uniform scale factor — simulates distance variation |
| `RandomFlip` | Random axis flip with probability `p` |
| `RandomPointDropout` | Drops a fraction of points and resamples replacements — simulates occlusion |
| `RandomShift` | Random global translation |
| `Compose` | Chains a list of augmentations sequentially |
| `simclr_augmentation()` | Default preset: rotation → scale → jitter → flip → dropout → shift |
| `attpc_augmentation()` | AT-TPC preset: stronger jitter and dropout for detector noise; no flip (track direction is meaningful) |

### Dataset — `contrastive_dataset.py`
Wraps the ShapeNet `.pt` files for contrastive training.

| Symbol | Purpose |
|---|---|
| `ContrastiveShapeNetDataset` | `Dataset` that loads a `.pt` file, applies the augmentation pipeline twice per sample, and voxelizes each view into a `SparseTensor`; returns `{view_a, view_b, original, label}` |
| `collate_contrastive_batch` | Collate function: calls `sparse_collate` on the two views and stacks labels into a batch |
| `make_contrastive_dataloader` | Convenience factory used by `train_contrastive.py` |
| `load_metadata` | Returns a summary dict (class names, counts, split sizes) printed at training start |

### Contrastive Training — `train_contrastive.py`
Main self-supervised training loop.

- Builds a `SparseSimCLR` model via `sparse_simclr_21d()`.
- Uses Adam + cosine annealing LR with gradient clipping.
- Saves `best.pt` on every loss improvement and `epoch_NNN.pt` every `--save-every` epochs.
- Writes `run_config.json` alongside checkpoints so that downstream scripts can automatically match model hyperparameters.

```bash
python train_contrastive.py --data shapenet_simple_large.pt --epochs 100 --batch-size 16
python train_contrastive.py --data shapenet_simple_large.pt --resume checkpoints/best.pt
```

### Latent Extraction — `extract_latents.py`
Runs a trained encoder over all splits of the dataset (no augmentation) and saves the resulting features and integer class labels as aligned `.npy` arrays.

- Reads `run_config.json` automatically to reconstruct the exact model that was trained; CLI flags override any entry.
- Calls `model.encode()` — backbone + global pool only, no projection head.
- Outputs `latent_vectors.npy` `(N, backbone_out_channels)` and `labels.npy` `(N,)`.
- For the purpose of being compatible with the ATTPC Latent Repo

```bash
python extract_latents.py --checkpoint checkpoints/best.pt
```

## Quick Start

```bash
# 1. Self-supervised training
python train_contrastive.py --data shapenet_simple_large.pt --epochs 100

# 2. Extract frozen backbone representations
python extract_latents.py --checkpoint checkpoints/best.pt

# 3. Evaluate with sklearn linear probe (From ATTPC Latent) + learning curves
python linear_probing.py --name run1 checkpoints/latent_vectors.npy checkpoints/labels.npy
```

## Checkpoints Directory Layout

```
checkpoints/
  run_config.json        # hyperparameters saved by train_contrastive.py
  best.pt                # lowest avg NT-Xent loss across all epochs
  epoch_010.pt           # periodic checkpoints (every --save-every epochs)
  final.pt               # state at the end of training
  loss_history.json      # per-epoch {epoch, avg_loss, lr} log
  latent_vectors.npy     # produced by extract_latents.py
  labels.npy             # aligned class labels (same order as latent_vectors)
```

---

## Dependencies

- PyTorch ≥ 2.0
- [TorchSparse](https://github.com/mit-han-lab/torchsparse) (CUDA required for backbone inference)
- scikit-learn, numpy, matplotlib, seaborn (evaluation + visualisation)
- click (linear_probing.py CLI)
