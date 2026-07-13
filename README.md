# torchSparseContrastive

Self-supervised contrastive learning for O16 AT-TPC particle-track events using
TorchSparse, a sparse 3D ResNet backbone, and a SimCLR / NT-Xent objective.

The repo has been reorganized around a package-style `src/` layout. The shell
entrypoints in `scripts/` are the recommended way to run jobs on the cluster.

---

## Current project layout

```text
torchSparseContrastive/
├── src/
│   ├── data/                  # O16 conversion, O16 datasets, legacy ShapeNet dataset
│   ├── models/                # SparseSimCLR model and constructor shim
│   ├── training/              # contrastive and supervised training entrypoints
│   ├── evaluation/            # latent extraction and linear probing
│   └── utils/                 # shared augmentation utilities
├── scripts/                   # SLURM wrappers
├── data/                      # raw/processed arrays and labels; usually not committed
├── checkpoints/               # trained model checkpoints; usually not committed
├── embeddings/                # extracted latent vectors; usually not committed
├── linear_probe_results/      # linear-probe metrics, plots, reports
├── logs/                      # SLURM/job logs
└── torchsparse/               # local TorchSparse source/backend
```

For a file-by-file explanation of `src/`, see [src/README.md](src/README.md).

---

## Main O16 workflow

### 1. Convert raw O16 data

```bash
sbatch scripts/run_convert_data.sh
```

This runs:

```bash
python -u src/data/convert-data.py
```

and produces:

```text
data/O16_event_lens.npy
data/O16_w_event_keys.npy
```

### 2. Build labeled downstream splits

```bash
sbatch scripts/run_O16_downstream_pipeline.sh
```

This runs:

```bash
python -u -m src.data.O16_downstream_pipeline
```

and produces split files such as:

```text
data/O16_size512_train.npy
data/O16_size512_val.npy
data/O16_size512_test.npy
```

The downstream pipeline expects `data/O16_labels.csv` to contain event-level
track-count labels.

### 3. Train SparseSimCLR

```bash
sbatch scripts/run_train_O16_contrastive.sh
```

This runs:

```bash
python -u -m src.training.train_contrastive
```

Important current settings:

```text
voxel_size       = 0.025 (0 to 40 on all axes)
hash_rsv_ratio   = 8
in_channels      = 1   # amplitude only
```

The smaller voxel size prevents the final sparse downsampling stage from
collapsing to zero output coordinates. The larger TorchSparse hash reservation
prevents hash-map capacity errors during kernel-map construction.

### 4. Extract latent vectors

```bash
sbatch scripts/run_extract_latents.sh
```

This runs:

```bash
python -u -m src.evaluation.extract_latents
```

and writes:

```text
embeddings/O16_simclr_best/latent_vectors.npy
embeddings/O16_simclr_best/labels.npy
```

### 5. Linear-probe evaluation

```bash
sbatch scripts/linear_probe_copy.sh
```

This runs:

```bash
python -u src/evaluation/linear_probe.py
```

and writes learning curves, a confusion matrix, and classification reports to:

```text
linear_probe_results/O16_simclr_best_learning_curve/
```

---

## Model summary

```text
O16 event hits: x, y, z, amplitude
        │
        ├── augmentation A → voxelize → SparseTensor view_a
        └── augmentation B → voxelize → SparseTensor view_b

view_a → SparseResNet21D → sparse global avg pool → ProjectionHead → z_a
view_b → SparseResNet21D → sparse global avg pool → ProjectionHead → z_b

loss = NT-Xent(z_a, z_b)
```

Coordinates are used for sparse voxelization. The model feature channel is
amplitude only.

---

## Notes on `contrastive_dataset.py`

`src/data/legacy_files/contrastive_dataset.py` is a legacy/general ShapeNet-style
contrastive dataset for `.pt` files containing point clouds, normals, labels,
and optional train/val/test indices.

It is not used by the current O16 training path. The O16 path uses
`src/data/o16_dataset.py`.

As far as the O16 work here goes, we did not run `contrastive_dataset.py`; it is
kept as reference/legacy support for ShapeNet-style experiments.

---

## Important generated directories

These are runtime artifacts and are normally not source code:

```text
data/
checkpoints/
embeddings/
linear_probe_results/
logs/
```

They may be large and should generally stay out of git unless a small metadata
file is intentionally committed.

---

## Sanity checks

Useful quick checks after refactors:

```bash
PYTHONNOUSERSITE=1 /home/DAVIDSON/tomallenntiador/.conda/envs/contrastive/bin/python \
  -m py_compile $(find src -name '*.py' -not -path '*/__pycache__/*')

for f in scripts/*.sh; do bash -n "$f" || exit 1; done

PYTHONNOUSERSITE=1 /home/DAVIDSON/tomallenntiador/.conda/envs/contrastive/bin/python \
  -c "from src.models.sparse_simclr import sparse_simclr_21d; print('imports ok')"
```

---

## Current cleanup caution

The canonical source code now lives under `src/`. Some generated directories,
TorchSparse vendor files, logs, checkpoints, and old sanity-check outputs may
still appear in `git status`. Use `git status --short` before moving/deleting
anything, and prefer `git add -A` only once the structure is intentionally
settled.
