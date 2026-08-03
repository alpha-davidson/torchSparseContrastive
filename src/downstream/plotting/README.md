# Downstream result plotting

These CPU-only tools consume artifacts already written under
`results/downstream/`. They do not load TorchSparse models, retrain a task, or
modify checkpoints. Each Python program has a matching Slurm wrapper.

## Files

### `benchmark_analysis.py`

An integrity check independent of the evaluator summary CSVs. It scans saved
per-trial test labels and predictions, recomputes classification or regression
metrics, and writes audit tables. Classification analysis creates aggregate
confusion matrices; regression analysis creates prediction-versus-truth plots.
Outputs go below
`results/downstream/plots/<task>/<prefix>/artifact_analysis/` by default.

### `benchmark_analysis.sh`

Slurm wrapper for `benchmark_analysis.py`. Its editable `TASK`, `PREFIX`, and
`RESULTS_DIR` variables currently target O16 classification.

### `comparison_plotting.py`

Reads classification `checkpoint_summary.csv`, extracts each checkpoint's
pretraining epoch, and plots final test accuracy, macro F1, and weighted F1
across checkpoints. It includes individual checkpoint trials and, when present,
the random-init mean and sample-standard-deviation reference band. It also
writes an aggregated summary CSV and text summary.

### `comparison_plotting.sh`

Slurm wrapper for the O16 classification checkpoint-comparison plots.

### `macro_f1_comparison.py`

Compares downstream learning dynamics for one selected pretrained checkpoint
against random initialization. It reads per-trial histories, aligns the epochs
actually recorded, and plots the cross-trial mean ± one sample standard
deviation for validation macro F1 and validation accuracy. The pretrained line
is red and the random-init line is blue. By default it selects the checkpoint
with the greatest parsed pretraining epoch; `--checkpoint-id` can override it.

Default outputs are `macro_f1_pretrained_vs_random.png` and
`accuracy_pretrained_vs_random.png` in the classification plot directory. A
curve can only extend through epochs present in the saved histories; the
program does not extrapolate missing epochs.

### `macro_f1_comparison.sh`

Slurm wrapper for the O16 pretrained-versus-random learning-dynamics plots.

### `regression_comparison_plotting.py`

Plots scaled R², RMSE, and MAE against contrastive pretraining epoch with
random-init reference bands. If physical checkpoint and random summaries are
available, it also creates per-target physical-unit plots. It writes an
aggregated checkpoint summary and plot summary alongside the figures.

### `regression_comparison_plotting.sh`

Slurm wrapper for regression comparison. Its default `C16` prefix is a
placeholder for a dataset that has pointwise regression labels; the current
`O16_UNSAMPLED` files contain event-classification targets and are not suitable
for this script.

### `saveplt.py`

Small optional helper for interactive use. `save_image(path)` writes every
currently open Matplotlib figure into one multipage PDF.

## Expected result layout

```text
results/downstream/
├── classification/<prefix>/
│   ├── checkpoint_summary.csv
│   ├── <checkpoint_id>/trial_<n>/...
│   └── random_init/summary.csv
└── regression/<prefix>/
    ├── checkpoint_summary.csv
    ├── <checkpoint_id>/trial_<n>/...
    └── random_init/scaled_summary.csv
```

The old exploratory notebooks previously associated with this directory are
not present in the current source tree. The maintained reproducible interfaces
are the Python modules listed above.
