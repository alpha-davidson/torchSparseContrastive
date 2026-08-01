# Slurm job scripts

The shell entrypoints are grouped by pipeline stage. Every wrapper activates
the `contrastive` environment, changes to the repository root, and then invokes
the corresponding program under `src/`. GPU jobs also load the CUDA/GCC module
pair expected by the local TorchSparse build.

## Directory map

| Directory | Purpose |
|---|---|
| [`data_conversion_scripts/`](data_conversion_scripts/README.md) | Convert raw isotope HDF5 files into NumPy events and length arrays. |
| [`smoke_test/`](smoke_test/README.md) | Exercise the active O16 sparse dataloader without training. |
| [`training_scripts/`](training_scripts/README.md) | Run current O16-only or combined-isotope contrastive pretraining. |
| [`downstream_dataloader_script/`](downstream_dataloader_script/README.md) | Build labeled variable-length O16 downstream splits. |
| [`latent_extraction/`](latent_extraction/README.md) | Extract pooled encoder features from labeled splits. |
| [`probing_scripts/`](probing_scripts/README.md) | Run linear logistic-regression or RBF-SVC embedding probes. |
| [`downstream_eval_scripts/`](downstream_eval_scripts/README.md) | Compare pretrained checkpoints with an architecture-matched random baseline. |
| [`legacy/`](legacy/README.md) | Preserve superseded O16-only, fixed-512, and older extraction jobs. |

## Active order

```text
data_conversion_scripts/
        ↓
smoke_test/ (optional)
        ↓
training_scripts/
        ↓
downstream_dataloader_script/
        ├─→ latent_extraction/ → probing_scripts/
        └─→ downstream_eval_scripts/
```

Submit wrappers from the repository root, for example:

```bash
sbatch scripts/training_scripts/run_train_O16_contrastive.sh
```

Before submission, inspect the variables near the middle of the selected
script. Checkpoint directories, dataset prefixes, dataset mode, epoch counts,
and output names are experiment configuration rather than automatic discovery
in most wrappers.
