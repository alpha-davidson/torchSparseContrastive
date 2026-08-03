# Legacy Slurm jobs

These wrappers are kept to reproduce older experiments. They are not the
recommended entrypoints for the current combined-pretraining and variable-point
O16 pipeline.

## Scripts

### `run_train_contrastive.sh`

Older O16-only SparseSimCLR wrapper. It uses a single O16 data/lens pair and is
currently configured to resume from `checkpoints/best.pt`. The maintained
configurable O16/combined wrapper is
`../training_scripts/run_train_O16_contrastive.sh`.

### `run_O16_downstream_pipeline.sh`

Runs `src.data.legacy.O16_downstream_pipeline`, the old labeled pipeline that
forces every event to 512 points and writes fixed-size split/trial files. Use
`../downstream_dataloader_script/run_O16_downstream_no_resample.sh` for the
active variable-length path.

### `run_extract_latents.sh`

Runs `src.evaluation.legacy.extract_latents_legacy`, an older extractor whose
per-event renormalization no longer matches the maintained preprocessing. Use
`../latent_extraction/run_extract_latents_no_resample.sh` for current runs.
