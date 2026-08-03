# Frozen-representation evaluation

This package extracts globally pooled encoder representations and evaluates
them without end-to-end encoder fine-tuning. Dynamic supervised fine-tuning is
implemented separately under `src/downstream/`.

## Files and directories

### `extract_latents_no_resample.py`

Reconstructs SparseSimCLR from a checkpoint and its `run_config.json`, then
saves the 128-dimensional output of `model.encode`. Recommended split mode
reads variable-length `O16_UNSAMPLED_{train,val,test}.npy` files and their
lengths, preserving aligned labels. Raw mode uses
`src.data.dataset_loaders.o16_dataset.O16Dataset` and writes `-1` placeholder
labels. Outputs are `latent_vectors.npy` and `labels.npy`.

### `clustering.py`

Reusable labeled t-SNE, UMAP, PCA, PCA-variance, and k-means functions. It
validates feature/label alignment, returns calculated embeddings or statistics,
and writes caller-selected plots.

### `global_feature_exploration.ipynb`

Interactive analysis of extracted latent vectors. It imports
`src.evaluation.clustering`, standardizes embeddings, optionally regroups
labels, and explores k-means, PCA, UMAP, and t-SNE. Input and plot paths remain
experiment settings that should be checked before running the notebook.

### [`probing_files/`](probing_files/README.md)

Linear logistic-regression and nonlinear RBF-SVC probes for saved embeddings.

### [`legacy/`](legacy/README.md)

Older extractors for fixed-512 or superseded normalization workflows.

### `__init__.py`

Marks `src.evaluation` as a Python package.
