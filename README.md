# torchSparseContrastive

Self-supervised contrastive learning for AT-TPC sparse 3D point-cloud events
using TorchSparse, SparseResNet21D, and the SimCLR/NT-Xent objective.

Project-owned Python code is organized under [`src/`](src/README.md). Slurm job
entrypoints are organized by workflow under [`scripts/`](scripts/README.md).
Large datasets, checkpoints, embeddings, logs, and results are kept outside the
source packages.

## Repository layout

```text
torchSparseContrastive/
├── src/
│   ├── data/
│   │   ├── data_conversion_files/
│   │   ├── dataset_loaders/
│   │   ├── downstream_data_loader/
│   │   └── legacy/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   │   ├── probing_files/
│   │   └── legacy/
│   ├── downstream/
│   └── utils/
├── scripts/
│   ├── data_conversion_scripts/
│   ├── smoke_test/
│   ├── training_scripts/
│   ├── downstream_dataloader_script/
│   ├── latent_extraction/
│   ├── probing_scripts/
│   ├── downstream_eval_scripts/
│   └── legacy/
├── data/
├── checkpoints/
├── embeddings/
├── results/
├── linear_probe_results/
├── logs/
└── torchsparse/
```

## Current workflow

Run commands from the repository root.

```bash
# 1. Convert each raw isotope that will be used for pretraining.
sbatch scripts/data_conversion_scripts/run_convert_data.sh
sbatch scripts/data_conversion_scripts/run_convert_ar46.sh
sbatch scripts/data_conversion_scripts/run_convert_c16.sh
sbatch scripts/data_conversion_scripts/run_convert_mg22.sh

# 2. Optional O16 sparse-loader smoke test.
sbatch scripts/smoke_test/run_O16_dataset.sh

# 3. Train O16-only or combined-isotope SparseSimCLR.
sbatch scripts/training_scripts/run_train_O16_contrastive.sh

# 4. Build labeled, variable-length O16 train/validation/test splits.
sbatch scripts/downstream_dataloader_script/run_O16_downstream_no_resample.sh

# 5a. Frozen-representation evaluation.
sbatch scripts/latent_extraction/run_extract_latents_no_resample.sh
sbatch scripts/probing_scripts/run_linear_probe_copy.sh
# or: sbatch scripts/probing_scripts/run_rbf_probe.sh

# 5b. Dynamic supervised checkpoint comparison and random baseline.
sbatch scripts/downstream_eval_scripts/run_downstream_checkpoint_classification.sh
sbatch scripts/downstream_eval_scripts/run_downstream_random_classification.sh
```

Data conversion only needs to be rerun when its raw input or desired converted
output changes. The smoke test exercises the same O16 loader imported by the
trainer but does not prepare a separate training dataset.

## Model path

```text
event hits (x, y, z, amplitude)
    ├─ augmentation A → voxelization → SparseTensor view A
    └─ augmentation B → voxelization → SparseTensor view B

each view → SparseResNet21D → global average pool → projection MLP
two normalized projections → NT-Xent loss
```

`model.encode` exposes the 128-dimensional globally pooled backbone feature for
latent extraction and downstream heads. The SimCLR projection is used only for
the contrastive objective.

## Active and legacy paths

The active O16 downstream builder preserves true event lengths and writes
`O16_UNSAMPLED` splits. Files under `src/data/legacy/`,
`src/evaluation/legacy/`, and `scripts/legacy/` retain earlier fixed-512 or
superseded workflows for provenance and should not be mixed into the active
pipeline.

## Refactor checks

```bash
python -m compileall -q src
find scripts -type f -name '*.sh' -exec bash -n {} \;
```

Use `git status --short` before deleting or moving files: converted arrays,
checkpoints, results, and local TorchSparse build artifacts can be large and
are not interchangeable with source files.
