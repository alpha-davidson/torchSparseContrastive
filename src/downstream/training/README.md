# Pretrained downstream training

These entrypoints dynamically evaluate collections of contrastive checkpoints.
They discover checkpoints, rebuild the matching encoder from its saved config,
run independent supervised trials, and write per-checkpoint plus aggregate
results under `results/downstream/`.

## Files

### `training_pipeline.py`

Dynamic classification evaluator. By default it performs O16 event-level
track-count classification; `--pointwise` selects the pointwise variant when
matching target arrays exist. Each trial loads a pretrained backbone and adds a
new classifier head, trains a head-only phase with the backbone frozen, restores
that phase's best validation model, then fine-tunes the complete network at a
lower learning rate. The better phase is selected by validation weighted F1
before one final test evaluation.

It saves phase weights and histories, learning curves, final probabilities and
labels, confusion matrices, classification reports, and
`checkpoint_summary.csv`. Checkpoint epoch is recorded so plotting can show
downstream performance as pretraining progresses.

### `finetune.py`

Dynamic pointwise-regression evaluator. It applies the same checkpoint
discovery, repeated-trial, frozen-head, and full-fine-tuning structure using the
regression components from `src.downstream.benchmark.regression`. Selection is
based on validation R². Outputs include best models, histories, test prediction
arrays, scaled R²/RMSE/MAE summaries, and optional inverse-scaled physical
metrics and plots when target scalers are supplied.
