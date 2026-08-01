# Raw data conversion

These programs turn isotope-specific HDF5 layouts into the padded NumPy event
arrays and companion event-length files consumed by the sparse loaders.

## Files

### `convert-data.py`

One-shot O16 converter. It reads the hard-coded `data/O16_run0160.h5` and
writes `data/O16_w_event_keys.npy` plus `data/O16_event_lens.npy`. Event columns
are `x, y, z, time_bucket, amplitude, event_index`. The file executes at import
time and retains its historical hyphenated filename, so its wrapper invokes it
by file path rather than with `python -m`.

### `Ar46_convert.py`

One-shot Ar46 converter for events beneath the input file's `clean` group. It
writes columns `x, y, z, charge, NND, event_index` and the corresponding event
lengths. Its raw input path is configured inside the file.

### `c16_convert.py`

CLI converter for nested `cluster/event/cluster_i/cloud` C16 files. It
concatenates clusters belonging to the same event, skips empty events,
validates shapes, and memory-maps the padded output. C16 has no NND feature, so
the auxiliary fifth feature column is zero.

### `mg22_convert.py`

CLI converter supporting structured `x,y,z,t,A` records and dense arrays. It
can discover events at the HDF5 root or inside a selected group, naturally
sorts event names, skips empty events, and memory-maps the output.

### `__init__.py`

Marks `src.data.data_conversion_files` as a package, allowing the converters
without hyphens to be launched with `python -m`.

The matching Slurm wrappers are documented in
[`scripts/data_conversion_scripts/`](../../../scripts/data_conversion_scripts/README.md).
