# Dynamic downstream evaluation jobs

These GPU jobs use the complete canonical labeled splits. They do not use the
old training-size subset files or fixed-512 event resampling.

## Scripts

### `run_downstream_checkpoint_classification.sh`

Runs `src.downstream.training.training_pipeline` over every checkpoint matching
`CHECKPOINT_DIR/CHECKPOINT_PATTERN`. For each checkpoint and trial it loads the
pretrained backbone, trains a new classification head while frozen, fine-tunes
the full model, selects weights by validation weighted F1, and evaluates the
test split. Per-trial artifacts and `checkpoint_summary.csv` are written below
`results/downstream/classification/<prefix>/`.

Set `DRY_RUN=1` to validate datasets, configs, checkpoint discovery, and model
loading without starting supervised training.

### `run_downstream_random_classification.sh`

Runs `src.downstream.benchmark.classification_no_pretrain`. It reads
`run_config.json` only to reproduce the pretrained model's architecture,
feature channels, voxelization, and TorchSparse settings; weights remain
randomly initialized. It trains repeated full-network trials and writes the
baseline below `results/downstream/classification/<prefix>/random_init/`.

For a controlled comparison, keep `DATA_DIR`, `PREFIX`, architecture config,
batch size, and trial count aligned between the two wrappers. Epoch counts may
differ, but plots only compare epochs that were actually recorded.
