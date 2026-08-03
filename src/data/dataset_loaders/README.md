# Sparse dataset loaders

This package converts variable-length padded event data into independently
augmented and voxelized TorchSparse inputs for contrastive pretraining.

## Files

### `o16_dataset.py`

Defines `O16Dataset`, `collate_o16_batch`, and `make_o16_dataloader`. It
memory-maps O16 events, crops by saved lengths, filters events below
`min_hits`, applies shared detector-coordinate and log-amplitude scaling,
creates two augmented views, shifts each view to nonnegative coordinates, and
voxelizes it. Sparse collation produces public coordinates in
`(batch, x, y, z)` order.

Running this module directly performs its built-in dataset smoke test. The
trainer imports `make_o16_dataloader` from this location.

### `combined_dataset.py`

Defines `CombinedATTPC` and `make_attpc_dataloader`. It presents multiple O16,
Ar46, C16, and Mg22 converted file pairs as one virtual dataset without writing
a duplicate combined array. The isotope name selects the correct charge or
amplitude column, while preprocessing and sparse-view creation remain aligned
with the O16 loader.

### `__init__.py`

Marks `src.data.dataset_loaders` as a package.
