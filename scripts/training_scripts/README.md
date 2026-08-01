# Contrastive training jobs

## Script

### `run_train_O16_contrastive.sh`

The maintained SparseSimCLR Slurm wrapper. `DATASET_MODE="o16"` selects the
single O16 loader; `DATASET_MODE="combined"` presents O16, Ar46, C16, and Mg22
as one virtual training dataset. In combined mode, the `DATA`, `LENS`, and
`DATASET_NAMES` arrays must remain in matching order because dataset names
select isotope-specific feature columns.

The job runs `src.training.train_contrastive`, configures voxel size and
TorchSparse hash reservation, trains with the selected optimization/model
settings, and writes `run_config.json`, `loss_history.json`, `best.pt`, periodic
epoch checkpoints, and `final.pt` beneath `SAVE_DIR`. Set `RESUME` to continue a
compatible run or `MAX_BATCHES` to perform a short training smoke test.

The wrapper currently defaults to combined-isotope training. Conversion files
for every listed isotope must exist before submission.
