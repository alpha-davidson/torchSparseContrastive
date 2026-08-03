# Downstream benchmark components

This directory contains the reusable sparse datasets, supervised heads,
training/evaluation functions, and architecture-matched random-initialization
baselines used by `src/downstream/training/`.

## Files

### `classification.py`

Shared event-level and pointwise classification implementation. Its datasets
crop each padded event by its saved length, shift coordinates to nonnegative
voxel indices, quantize, and sparse-collate batches. `SparseEventClassifier`
attaches a dense MLP head to the encoder's 128-dimensional global pooled
representation. `SparsePointClassifier` predicts on the final sparse feature
map and propagates predictions back to input voxels by nearest-neighbor lookup.

The module also loads pretrained backbone weights, freezes/unfreezes the
encoder, runs optimization and prediction, selects best weights by validation
weighted F1, and supports proton-label swap invariance for applicable
pointwise targets.

### `classification_no_pretrain.py`

Executable random-initialization classification baseline. It reads an existing
pretraining config only to match architecture, feature channels, voxel size,
and TorchSparse settings; it deliberately does not load checkpoint weights.
For each trial it trains the same event or pointwise classifier on the complete
canonical splits, then saves the best model, history, learning curve, test
probabilities/labels, confusion matrix, classification report, and an aggregate
summary CSV under `results/downstream/classification/<prefix>/random_init/`.

### `regression.py`

Shared pointwise regression implementation. It provides padded-array datasets
and loaders, sparse point and event regressors, nearest-neighbor propagation to
input points, weighted regression loss, training/prediction loops, target
scaler loading and inverse transformation, and scaled/physical metric and plot
generation. Dynamic regression evaluation and the random baseline import these
components.

### `regression_no_pretrain.py`

Executable architecture-matched random-initialization pointwise-regression
baseline. Across multiple trials it trains on the full canonical splits and
saves histories, loss curves, best weights, test arrays, event lengths, scaled
R²/RMSE/MAE summaries, and optional inverse-scaled physical-target reports
under `results/downstream/regression/<prefix>/random_init/`.

### `plotting.py`

Low-level plots produced during benchmark runs: training/validation learning
curves, confusion matrices, and selected pointwise event visualizations. This
file remains part of the benchmark because training programs call it directly;
cross-check and comparison plots live in the sibling `../plotting/` directory.
