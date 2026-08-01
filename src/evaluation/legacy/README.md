# Legacy latent extractors

These scripts preserve earlier extraction behavior and are not the recommended
path for current variable-length O16 splits.

## Files

### `extract_latents.py`

The original extractor for labeled fixed-512 O16 split files, with a raw-data
fallback. It loads a SparseSimCLR checkpoint and saves pooled encoder features
and labels. Its expected filenames and preprocessing belong to the resampled
downstream workflow.

### `extract_latents_legacy.py`

An earlier no-resample adaptation. It reads companion event lengths but
renormalizes each event's coordinates and amplitude independently before
voxelization. That differs from current training and split preprocessing. The
maintained `../extract_latents_no_resample.py` instead consumes already-scaled
split values directly, applies only the required nonnegative coordinate shift,
handles combined-run configurations safely, and explicitly switches the model
to evaluation mode.
