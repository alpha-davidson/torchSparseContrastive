# Source tree guide

This file explains the project-owned Python source under `src/`. The canonical
code path now lives here.

---

## Package map

```text
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── O16_downstream_pipeline.py
│   ├── convert-data.py
│   ├── legacy_files/
│   │   ├── __init__.py
│   │   └── contrastive_dataset.py
│   └── o16_dataset.py
├── evaluation/
│   ├── __init__.py
│   ├── extract_latents.py
│   └── linear_probe.py
├── models/
│   ├── __init__.py
│   ├── model.py
│   └── sparse_simclr.py
├── training/
│   ├── __init__.py
│   ├── model.py
│   ├── train_contrastive.py
│   └── train_supervised.py
└── utils/
    ├── __init__.py
    └── augmentations.py
```

---

## `src/data/`

### `src/data/convert-data.py`

Standalone raw-data converter for `data/O16_run160.h5`.

Inputs:

```text
data/O16_run160.h5
```

Outputs:

```text
data/O16_event_lens.npy
data/O16_w_event_keys.npy
```

`O16_w_event_keys.npy` stores each event as:

```text
x, y, z, time_bucket, amplitude, event_index
```

### `src/data/O16_downstream_pipeline.py`

Full labeled downstream-data preparation pipeline.

Main steps:

1. Convert raw HDF5 to numpy arrays.
2. Attach event-level track-count labels from `data/O16_labels.csv`.
3. Simplify raw track counts into 3 classes.
4. Resample each event to 512 hits.
5. Scale features.
6. Create train/val/test split files.
7. Generate small-train-size trial files for downstream benchmark studies.

Primary outputs include:

```text
data/O16_dataset.npy
data/O16_size512_sampled.npy
data/O16_size512_scaled.npy
data/O16_size512_train.npy
data/O16_size512_val.npy
data/O16_size512_test.npy
```

### `src/data/o16_dataset.py`

The active O16 dataset used for contrastive pretraining and raw latent
extraction.

Important objects:

| Symbol | Purpose |
|---|---|
| `O16Dataset` | Loads O16 events from `O16_w_event_keys.npy` using mmap, filters short events, normalizes each event, creates two augmented views, and voxelizes to TorchSparse `SparseTensor`s. |
| `attpc_aug_list()` | Default AT-TPC augmentation list. |
| `_augment()` | Applies spatial transforms while carrying amplitude features along. |
| `collate_o16_batch()` | Sparse-collates `view_a`, `view_b`, and `original`. |
| `make_o16_dataloader()` | Convenience `DataLoader` factory for training. |

This is the dataset used by `src/training/train_contrastive.py`.

### `src/data/legacy_files/contrastive_dataset.py`

Legacy/general contrastive dataset for ShapeNet-style `.pt` files.

Important objects:

| Symbol | Purpose |
|---|---|
| `ContrastiveShapeNetDataset` | Returns two augmented sparse views of each ShapeNet-style point cloud. |
| `collate_contrastive_batch()` | Sparse collate function for the two views plus labels. |
| `make_contrastive_dataloader()` | Convenience DataLoader factory. |
| `load_metadata()` | Reads basic dataset metadata from the `.pt` file. |

This file is not part of the current O16 training run. It was not used in the
successful O16 pipeline; it is retained for ShapeNet-style experiments or
reference.

---

## `src/utils/`

### `src/utils/augmentations.py`

Shared point-cloud augmentation utilities.

Important classes/functions:

| Symbol | Purpose |
|---|---|
| `RandomRotation` | Randomly rotates xyz coordinates. |
| `RandomJitter` | Adds clipped Gaussian coordinate noise. |
| `RandomScale` | Applies random global scaling. |
| `RandomFlip` | Randomly flips one axis. |
| `RandomPointDropout` | Drops a fraction of points and resamples replacements. |
| `RandomShift` | Applies random global translation. |
| `Compose` | Chains transforms. |
| `simclr_augmentation()` | Generic SimCLR point-cloud augmentation preset. |
| `attpc_augmentation()` | AT-TPC-specific augmentation preset. |

---

## `src/models/`

### `src/models/model.py`

Core SparseSimCLR implementation.

Important objects:

| Symbol | Purpose |
|---|---|
| `_sparse_global_avg_pool()` | Pools sparse voxel features into one dense vector per event. Supports an explicit `batch_size` to preserve empty final-layer samples. |
| `ProjectionHead` | MLP projection head used for SimCLR contrastive training. |
| `NTXentLoss` | Normalized temperature-scaled cross-entropy loss. |
| `SparseSimCLR` | Full contrastive model wrapper: backbone, projector, and loss. |
| `sparse_simclr_21d()` | Convenience constructor using `SparseResNet21D`. |

The explicit batch-size handling in `_sparse_global_avg_pool()` is important:
it prevents one augmented view from returning fewer embeddings than the other
when a sample has zero occupied voxels at the deepest sparse layer.

### `src/models/sparse_simclr.py`

Small re-export shim:

```python
from src.models.model import SparseSimCLR, sparse_simclr_21d
```

This keeps training/evaluation imports short and stable.

---

## `src/training/`

### `src/training/train_contrastive.py`

Main O16 SimCLR pretraining entrypoint.

Responsibilities:

1. Parse training/config/checkpoint arguments.
2. Set TorchSparse `hash_rsv_ratio`.
3. Build the O16 dataloader.
4. Build `SparseSimCLR`.
5. Train with Adam, cosine annealing, and gradient clipping.
6. Save `best.pt`, periodic checkpoints, `final.pt`, `run_config.json`, and `loss_history.json`.

Recommended invocation:

```bash
python -u -m src.training.train_contrastive \
  --data data/O16_w_event_keys.npy \
  --lens data/O16_event_lens.npy \
  --voxel-size 0.025 \
  --hash-rsv-ratio 8 \
  --in-channels 1
```

Cluster wrapper:

```bash
sbatch scripts/run_train_O16_contrastive.sh
```

### `src/training/train_supervised.py`

ShapeNet-oriented supervised classifier training script.

It builds a sparse ResNet classifier and can optionally initialize the backbone
from a SimCLR checkpoint. This is not the main O16 contrastive path.

### `src/training/model.py`

Currently an unused duplicate of `src/models/model.py`.

It is safe to remove later once the repo cleanup is committed and imports are
confirmed stable. For now, treat `src/models/model.py` as canonical.

---

## `src/evaluation/`

### `src/evaluation/extract_latents.py`

Loads a trained SparseSimCLR checkpoint and extracts one latent vector per
event.

Modes:

| Mode | Description |
|---|---|
| Split mode | Uses `data/O16_size512_{train,val,test}.npy`; saves aligned labels. Recommended for linear probing. |
| Raw mode | Uses `data/O16_w_event_keys.npy`; labels are `-1` placeholders. |

Outputs:

```text
embeddings/O16_simclr_best/latent_vectors.npy
embeddings/O16_simclr_best/labels.npy
```

Cluster wrapper:

```bash
sbatch scripts/run_extract_latents.sh
```

### `src/evaluation/linear_probe.py`

Sklearn/matplotlib-based linear probe for extracted latent vectors.

Inputs:

```text
latent_vectors.npy
labels.npy
```

Outputs:

```text
linear_probe_results/<name>_learning_curve/
  learning_curve_results.json
  learning_curve.png
  performance_table.png
  final_confusion_matrix.png
  classification_report.csv
  classification_report.png
```

It trains logistic-regression probes over logarithmically spaced training set
sizes, reports mean/std accuracy over folds, and trains a final model using the
full training split.

Cluster wrapper:

```bash
sbatch scripts/linear_probe_copy.sh
```

---

## Current canonical paths

Use these going forward:

```text
src/data/o16_dataset.py
src/data/O16_downstream_pipeline.py
src/data/convert-data.py
src/models/model.py
src/models/sparse_simclr.py
src/training/train_contrastive.py
src/evaluation/extract_latents.py
src/evaluation/linear_probe.py
src/utils/augmentations.py
```

Avoid editing root-level legacy copies if any reappear. The source of truth is
`src/`.
