# Source code guide

`src/` is the importable Python source tree. Shell entrypoints under
[`../scripts/`](../scripts/README.md) invoke these packages from the repository
root.

## Directory map

| Directory | Purpose |
|---|---|
| [`data/`](data/README.md) | Raw conversion, sparse dataset loading, labeled split creation, and legacy data workflows. |
| [`models/`](models/README.md) | SparseSimCLR encoder, pooling, projection head, and contrastive loss. |
| [`training/`](training/README.md) | Contrastive pretraining and the separate supervised ShapeNet reference trainer. |
| [`utils/`](utils/README.md) | Shared point-cloud augmentations. |
| [`evaluation/`](evaluation/README.md) | Latent extraction, shallow probes, and latent-space exploration. |
| [`downstream/`](downstream/README.md) | Dynamic pretrained checkpoint evaluation, random baselines, and plots. |

`__init__.py` marks the root package so modules can be invoked with commands
such as `python -m src.training.train_contrastive`.

## Active dependency flow

```text
data/data_conversion_files/
            ↓
data/dataset_loaders/ → training/train_contrastive.py → checkpoints
            ↓                                      ↓
data/downstream_data_loader/                       ↓
            ├─→ evaluation/extract_latents_no_resample.py
            │          └─→ evaluation/probing_files/
            └─→ downstream/training/ + downstream/benchmark/
```

Generated caches (`__pycache__/`, `.ipynb_checkpoints/`) and the accidental
empty `evaluation/home/...` output tree are not source packages.
