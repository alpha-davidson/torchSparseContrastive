# Data-conversion jobs

These CPU/large-memory Slurm jobs generate the converted arrays needed for
pretraining. They should only be rerun when the raw input, conversion logic, or
desired output changes.

## Scripts

### `run_convert_data.sh`

Runs the O16 one-shot converter at
`src/data/data_conversion_files/convert-data.py`. It creates
`data/O16_w_event_keys.npy` and `data/O16_event_lens.npy` from the converter's
configured HDF5 input.

### `run_convert_ar46.sh`

Runs `src.data.data_conversion_files.Ar46_convert`. It creates the Ar46 padded
event array and event-length file using the raw path configured in the Python
module.

### `run_convert_c16.sh`

Runs `src.data.data_conversion_files.c16_convert` with its default C16 input
and output prefix. Pass additional CLI options in the wrapper when a different
`--input` or `--output-prefix` is required.

### `run_convert_mg22.sh`

Runs `src.data.data_conversion_files.mg22_convert` with its default Mg22 input
and output prefix. The Python program also supports explicit `--group`,
`--input`, and `--output-prefix` settings.

All four jobs create `logs/` and `data/` if necessary and write their Slurm
stdout to isotope-specific log files.
