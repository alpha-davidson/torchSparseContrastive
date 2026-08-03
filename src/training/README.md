# Training entrypoints

This directory contains model-training programs. The O16/multi-isotope
contrastive trainer is the active pretraining path; the supervised trainer is a
separate ShapeNet-oriented reference workflow.

## Files

### `train_contrastive.py`

The main SparseSimCLR pretraining command. It can build either an O16-only
loader from `src.data.dataset_loaders.o16_dataset` or a virtual multi-isotope
loader from `src.data.dataset_loaders.combined_dataset`. It configures
TorchSparse's hash reservation,
constructs `SparseSimCLR`, optimizes NT-Xent loss with Adam, applies gradient
clipping and cosine learning-rate decay, and supports resuming a run.

The save directory receives `run_config.json`, `loss_history.json`, `best.pt`,
periodic `epoch_*.pt` checkpoints, and `final.pt`. These checkpoint/config pairs
are later consumed by latent extraction and dynamic downstream evaluation.

### `train_supervised.py`

A standalone sparse ShapeNet classifier trainer for `.pt` datasets. It can
initialize its SparseResNet backbone from a SimCLR checkpoint, optionally
freeze that backbone, use mixed precision, resume training, and save supervised
checkpoints. It does not drive the current O16 downstream benchmark, whose
implementation is under `src/downstream/`.

### `__init__.py`

Marks `src.training` as a Python package.
