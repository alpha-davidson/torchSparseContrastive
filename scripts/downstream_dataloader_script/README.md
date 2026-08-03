# Downstream split-generation job

## Script

### `run_O16_downstream_no_resample.sh`

Runs
`src.data.downstream_data_loader.O16_downstream_no_resample` as a module. It
combines converted O16 events with `data/O16_labels.csv`, creates the canonical
labeled train/validation/test partitions, scales features consistently with
pretraining, and writes `O16_UNSAMPLED_*` data and length arrays without
resampling events to 512 points.

This is a CPU/large-memory preprocessing job. It is required before labeled
latent extraction or dynamic O16 classification, but not before contrastive
pretraining itself.
