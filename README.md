# SparseSimCLR — Contrastive Learning for AT-TPC Particle Track Data

Self-supervised representation learning on voxelized 3D particle tracks from the Active Target Time Projection Chamber (AT-TPC), using SimCLR and a sparse 3D ResNet backbone ([TorchSparse](https://github.com/mit-han-lab/torchsparse)). Trained without labels; representations are evaluated via linear probing (see [ATTPCLatent](https://github.com/your-org/ATTPCLatent)).

The primary dataset is O16 data from `O16_run160.h5`. Each event is a 3D point cloud of pad hits with fields `x, y, z, time_bucket, amplitude`. The model learns representations that capture track topology without supervision.

---

## Model Architecture

```
AT-TPC Event (N hits × {x,y,z,amplitude})
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

**Input feature:** amplitude only (`in_channels=1`). xyz coordinates are used for voxelization; amplitude is the per-voxel feature fed to the network.

---

## Data Pipeline

Raw data lives in `O16_run160.h5`. Run `O16_downstream_pipeline.py` once to produce the numpy arrays consumed by training:

```
O16_run160.h5
    │
    ├─ convert_data()         →  O16_w_event_keys.npy   (n_events, max_hits, 6)
    │                             cols: x, y, z, time_bucket, amplitude, event_idx
    │
    ├─ add_num_tracks()       →  O16_dataset.npy        (n_labelled, max_hits, 7)
    │                             adds track-count label from O16_labels.csv
    │
    ├─ simplify_class()       →  (in-place) re-maps track counts → 3 classes
    │                             0,1,2 → 0  |  3 → 1  |  4,5 → 2
    │
    ├─ random_sample()        →  O16_size512_sampled.npy   (n_events, 512, 5)
    ├─ scale_data()           →  O16_size512_scaled.npy    (normalised [0,1])
    ├─ split_train_val_test() →  O16_size512_{train,val,test}.npy
    └─ generate_trials()      →  O16_size512_{N}train_{features,labels}/trial_k.npy
```

For contrastive training specifically, only the first step is needed:

```bash
python O16_downstream_pipeline.py   # or: sbatch run_O16_downstream_pipeline.sh
# produces: data/O16_w_event_keys.npy, data/O16_event_lens.npy
```

---

## File Reference

### Core Model — `model.py`

| Symbol | Purpose |
|---|---|
| `_sparse_global_avg_pool` | Scatter-averages all occupied voxel features into one `(B, C)` dense tensor |
| `ProjectionHead` | Two-layer MLP (Linear → BN → ReLU → Linear → optional BN) that maps backbone features to the contrastive embedding space |
| `NTXentLoss` | NT-Xent loss: builds a `(2B, 2B)` cosine-similarity matrix, masks self-pairs, and computes cross-entropy with positive-pair targets |
| `SparseSimCLR` | Top-level module; owns the backbone, projector, and criterion; exposes `encode()` (backbone + pool) and `project()` (encode + project + normalise) |
| `sparse_simclr_21d()` | Convenience constructor: builds `SparseResNet21D` + `SparseSimCLR` with one call |

### Re-export Shim — `sparse_simclr.py`
Thin module that loads `model.py` via `importlib` and re-exports `SparseSimCLR` and `sparse_simclr_21d`. Allows training scripts to do `from sparse_simclr import ...` regardless of Python path.

### Augmentations — `augmentations.py`
All augmentations operate on raw `(N, 3)` numpy point clouds and return a copy. Applied twice independently to the same event to produce the two contrastive views.

| Class | Effect |
|---|---|
| `RandomRotation` | Random rotation around one or more axes |
| `RandomJitter` | Per-point Gaussian noise (clipped) — simulates pad noise and gain variation |
| `RandomScale` | Uniform scale factor |
| `RandomFlip` | Random axis flip with probability `p` |
| `RandomPointDropout` | Drops a fraction of hits and resamples replacements — simulates missing pads |
| `RandomShift` | Random global translation |
| `Compose` | Chains a list of augmentations sequentially |
| `attpc_augmentation()` | AT-TPC preset: stronger jitter (`σ=0.02`) and dropout (`p=0.2`); no flip since track direction is physically meaningful |

### O16 Dataset — `o16_dataset.py`
Loads O16 AT-TPC events from the numpy arrays produced by `O16_downstream_pipeline.py`.

| Symbol | Purpose |
|---|---|
| `O16Dataset` | `Dataset` that loads events from mmap'd `.npy` files, normalises xyz and amplitude per-event to `[0,1]`, applies augmentations independently twice, and voxelizes each view into a `SparseTensor`; returns `{view_a, view_b, original}` |
| `collate_o16_batch` | Collate function: calls `sparse_collate` on both views |
| `make_o16_dataloader` | Convenience factory used by `train_contrastive.py` |
| `attpc_aug_list()` | Returns the default list of AT-TPC augmentation transforms |

**Column layout of `O16_w_event_keys.npy` (axis-2):**
```
0  x            mm
1  y            mm
2  z            mm
3  t            time bucket
4  A            amplitude (charge)
5  event_idx
```

### Data Processing Pipeline — `O16_downstream_pipeline.py`
Sequential pipeline that converts `O16_run160.h5` to train/val/test numpy arrays with track-count labels. Each step is a standalone function; run `main()` to execute the full pipeline. See the Data Pipeline section above for the full step-by-step flow.

### Contrastive Training — `train_contrastive.py`
Main self-supervised training loop.

- Builds a `SparseSimCLR` model via `sparse_simclr_21d()`.
- Loads O16 events via `make_o16_dataloader` (`in_channels=1`, amplitude only).
- Uses Adam + cosine annealing LR with gradient clipping.
- Saves `best.pt` on every loss improvement and `epoch_NNN.pt` every `--save-every` epochs.
- Writes `run_config.json` alongside checkpoints so downstream scripts can automatically match model hyperparameters.

```bash
# Direct run
python train_contrastive.py \
    --data data/O16_w_event_keys.npy \
    --lens data/O16_event_lens.npy \
    --in-channels 1 \
    --epochs 100 --batch-size 16

# Resume from checkpoint
python train_contrastive.py \
    --data data/O16_w_event_keys.npy \
    --lens data/O16_event_lens.npy \
    --resume checkpoints/best.pt

# SLURM submit (recommended on cluster)
sbatch run_train_O16_contrastive.sh
```

### Latent Extraction — `extract_latents.py`
Runs a trained encoder over the dataset (no augmentation) and saves per-sample feature vectors and integer class labels as aligned `.npy` arrays, ready for downstream evaluation with [ATTPCLatent](https://github.com/your-org/ATTPCLatent).

- Reads `run_config.json` automatically to reconstruct the exact model that was trained; CLI flags override any entry.
- Calls `model.encode()` — backbone + global pool only, no projection head.
- Outputs `latent_vectors.npy` `(N, 128)` and `labels.npy` `(N,)`.
- **Requires GPU** — TorchSparse sparse downsampling is CUDA-only.
- If `--labels` is not provided, `labels.npy` is saved as `-1` placeholders so the output directory is always complete.

```bash
python extract_latents.py \
    --checkpoint checkpoints/best.pt \
    --data data/O16_w_event_keys.npy \
    --lens data/O16_event_lens.npy \
    --labels data/O16_size512_labels.npy   # optional
```

---

## Quick Start

```bash
# 1. Process raw data (one time only)
python O16_downstream_pipeline.py
# produces: data/O16_w_event_keys.npy, data/O16_event_lens.npy

# 2. Self-supervised contrastive training
sbatch run_train_O16_contrastive.sh
# or directly:
python train_contrastive.py \
    --data data/O16_w_event_keys.npy \
    --lens data/O16_event_lens.npy \
    --in-channels 1 --epochs 100

# 3. Extract frozen backbone representations
python extract_latents.py --checkpoint checkpoints/best.pt

# 4. Evaluate with linear probing (ATTPCLatent repo)
python linear_probing.py --name O16 checkpoints/latent_vectors.npy checkpoints/labels.npy
```

---

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
- [TorchSparse](https://github.com/mit-han-lab/torchsparse) — **must be installed from source; CUDA required**
- scikit-learn, numpy, matplotlib, umap-learn (evaluation + visualisation)
- h5py, pandas, tqdm (data processing pipeline)

Install all non-TorchSparse dependencies:
```bash
pip install -r requirements.txt
```
