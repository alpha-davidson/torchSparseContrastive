# Frozen-embedding probe jobs

These CPU jobs consume `latent_vectors.npy` and `labels.npy`. They do not load
or fine-tune a TorchSparse checkpoint.

## Scripts

### `run_linear_probe_copy.sh`

Runs `src.evaluation.probing_files.linear_probe`, a standardized logistic
regression probe. The wrapper selects the experiment name, random seed, maximum
training size, and embedding files. Results include learning curves,
confusion-matrix and classification-report artifacts under
`linear_probe_results/<name>_learning_curve/`.

### `run_rbf_probe.sh`

Runs `src.evaluation.probing_files.rbf_probe`, the nonlinear RBF-SVC
comparison. In addition to the data and learning-curve settings, the wrapper
configures `C`, `gamma`, and SVC cache memory.

Both wrappers currently point at combined-pretraining O16 embeddings. Update
the two final path arguments and `--name` together when probing another run.
