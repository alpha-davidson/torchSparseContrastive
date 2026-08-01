# Downstream data preparation

This package prepares labeled arrays for supervised evaluation after
contrastive pretraining.

## Files

### `O16_downstream_no_resample.py`

The active O16 track-count split builder. It joins manual labels from
`data/O16_labels.csv` to converted events, maps raw track counts
`0/1/2 → class 0`, `3 → class 1`, and `4/5 → class 2`, creates canonical
train/validation/test partitions, applies the shared global detector and
amplitude scaling, and writes `O16_UNSAMPLED_*` arrays plus companion length
files. Events retain their true measured point counts; no event is forced to
512 points.

The file retains optional training-size trial generation for older studies,
but current dynamic downstream evaluators load the complete canonical splits.
`RUN_CONVERT` is normally false so the large raw conversion is not repeated.

### `__init__.py`

Marks `src.data.downstream_data_loader` as a package.

The corresponding Slurm wrapper is documented in
[`scripts/downstream_dataloader_script/`](../../../scripts/downstream_dataloader_script/README.md).
