# Data package

`src.data` separates raw conversion, runtime sparse datasets, labeled
downstream split preparation, and preserved legacy workflows.

## Subdirectories

| Directory | Purpose |
|---|---|
| [`data_conversion_files/`](data_conversion_files/README.md) | Convert raw O16, Ar46, C16, and Mg22 HDF5 data into padded NumPy events and length arrays. |
| [`dataset_loaders/`](dataset_loaders/README.md) | Build O16-only or virtual multi-isotope TorchSparse contrastive batches. |
| [`downstream_data_loader/`](downstream_data_loader/README.md) | Create labeled variable-length O16 train/validation/test splits. |
| [`legacy/`](legacy/README.md) | Retain the old fixed-512 O16 and ShapeNet-style data workflows. |

`__init__.py` marks `src.data` as a Python package.

## Shared data contract

Converted isotope arrays have shape `(events, max_hits, columns)` and are zero
padded. A companion `*_event_lens.npy` identifies the real points in every
event. Dataset and downstream loaders must crop by those lengths before
voxelization; padding is never treated as measured detector hits.
