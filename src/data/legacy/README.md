# Legacy data workflows

These files are retained to reproduce earlier experiments. They are not the
recommended inputs to the current O16 training and dynamic downstream paths.

## Files

### `O16_downstream_pipeline.py`

The original labeled O16 downstream pipeline. It resamples every event to a
fixed 512 points, including upsampling short events and downsampling long ones,
then writes fixed-size splits and training-size trial subsets. Use
`../downstream_data_loader/O16_downstream_no_resample.py` when preserving the
measured point count is required.

### `contrastive_dataset.py`

A general/ShapeNet-style contrastive dataset for point clouds stored in `.pt`
files. It produces two augmented sparse views and includes its own collate and
dataloader helpers. It is separate from the NumPy-based O16 and multi-isotope
loaders used by the current AT-TPC pretraining runs.

### `__init__.py`

Marks the legacy directory as an importable package.
