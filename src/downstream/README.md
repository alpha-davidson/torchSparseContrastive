# Dynamic downstream evaluation

This package measures how useful contrastive checkpoints are for supervised
tasks. It discovers one or more checkpoints, recreates the matching sparse
encoder and voxelization from each saved configuration, trains new task heads,
and compares them with the same architecture trained from random
initialization. It consumes canonical train/validation/test arrays and does not
use the old fixed-512 event resampling or training-size trial subsets.

## Directories

| Directory | Purpose |
|---|---|
| [`training/`](training/README.md) | Multi-checkpoint pretrained classification and regression evaluations. |
| [`benchmark/`](benchmark/README.md) | Shared datasets/models/train loops plus random-init baselines. |
| [`plotting/`](plotting/README.md) | CPU-only aggregation, comparison, and integrity-check plots. |

## File

### `common.py`

Shared infrastructure for all downstream programs. It defines repository data
and result roots, reproducible seeding, safe checkpoint identifiers, checkpoint
discovery, adjacent `run_config.json` loading, TorchSparse hash configuration,
and encoder-constructor arguments. It also loads complete canonical event or
pointwise train/validation/test splits and validates their data, label, and
event-length alignment.

For classification it supports labels embedded in the event file or a separate
label file. Pointwise tasks require separate padded target arrays and use the
same length arrays as their point clouds.

## Evaluation contract

- Pretrained and random-init conditions use the same SparseResNet architecture,
  voxel size, canonical splits, and supervised task heads.
- Multiple `trial_<n>` runs provide seed-to-seed variation; reported standard
  deviations are across those independent trials.
- Model selection uses validation results. The test split is evaluated after
  the selected weights are restored.
- Event tensors enter TorchSparse collation as `(x, y, z)` and emerge in public
  `(batch, x, y, z)` coordinate order.
