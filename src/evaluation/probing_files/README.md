# Embedding probes

These CPU/scikit-learn programs evaluate already extracted feature vectors.
They do not load TorchSparse or modify the contrastive checkpoint.

## Files

### `linear_probe.py`

The repository's linear probe, implemented with standardized
`sklearn.linear_model.LogisticRegression`. It creates a stratified test set,
fits repeated probes over logarithmically spaced training sizes, and records
mean and standard deviation for training and test accuracy. The final full-size
probe saves learning-curve JSON and plots, a performance table, confusion
matrix, and classification report under `linear_probe_results/`.

### `rbf_probe.py`

Nonlinear comparison using `sklearn.svm.SVC(kernel="rbf")`. It follows the
same split, scaling, learning-curve, confusion-matrix, and classification-report
workflow with configurable `C`, `gamma`, and SVC cache size.

### `__init__.py`

Marks `src.evaluation.probing_files` as a package so both probes can be run
with `python -m`.

Matching Slurm wrappers are documented in
[`scripts/probing_scripts/`](../../../scripts/probing_scripts/README.md).
