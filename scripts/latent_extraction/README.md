# Latent-extraction job

## Script

### `run_extract_latents_no_resample.sh`

Runs `src.evaluation.extract_latents_no_resample` on a selected SparseSimCLR
checkpoint. With `USE_SPLITS=1`, it processes the labeled variable-length
`O16_UNSAMPLED` train/validation/test files and saves aligned
`latent_vectors.npy` and `labels.npy` under `OUTPUT_DIR`. With `USE_SPLITS=0`,
it processes the raw O16 pretraining arrays and writes `-1` label placeholders.

The wrapper reads architecture and voxel settings from `CONFIG`, while optional
nonempty batch-size and voxel-size variables override those values. It writes
both Slurm output and a timestamped `tee` log. The selected checkpoint, config,
and output directory should always describe the same pretraining run.
