# Models

This directory defines the sparse contrastive model used throughout training,
embedding extraction, and downstream checkpoint evaluation.

## Files

### `model.py`

The canonical SparseSimCLR implementation:

- `_sparse_global_avg_pool` converts the final sparse feature map into one
  dense vector per event while preserving the collated batch size.
- `ProjectionHead` is a two-linear-layer MLP with an intermediate batch
  normalization and ReLU.
- `NTXentLoss` computes the temperature-scaled SimCLR objective between two
  batches of normalized projections.
- `SparseSimCLR` wraps a `SparseResNet21D` backbone, global average pooling,
  the projection head, and the contrastive loss.
- `sparse_simclr_21d` is the standard constructor used by the rest of the repo.

`model.encode(x)` returns the 128-dimensional globally pooled backbone
representation. `model.project(x)` sends that representation through the MLP
and L2-normalizes it. Contrastive loss is therefore applied after sparse
feature extraction, global pooling, and the projection head. Downstream tasks
normally attach their supervised heads to `encode`, not to the projection.

### `sparse_simclr.py`

A compatibility re-export of `SparseSimCLR` and `sparse_simclr_21d` from
`model.py`. It provides a stable, descriptive import path without duplicating
the implementation.

### `__init__.py`

Marks `src.models` as a Python package.
