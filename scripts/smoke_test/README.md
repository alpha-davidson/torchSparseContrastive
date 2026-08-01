# Dataset smoke test

## Script

### `run_O16_dataset.sh`

Runs `src.data.dataset_loaders.o16_dataset` directly so its built-in check can
load converted O16 events, create augmented sparse views, and collate a batch.
This exercises the same dataset module imported by contrastive training.

The smoke test is optional and creates no separate training dataset. It is a
quick validation after conversion, dependency changes, or loader refactors.
