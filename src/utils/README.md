# Shared utilities

This directory contains reusable point-cloud transformations. Dataset-specific
loaders decide which transformations to compose and when to voxelize.

## Files

### `augmentations.py`

Defines the augmentation interface and transformations used to build
independent contrastive views:

- `RandomRotation` rotates coordinates around a selected axis.
- `RandomJitter` adds clipped Gaussian coordinate noise.
- `RandomScale` applies a random isotropic scale.
- `RandomFlip` reflects a selected coordinate axis with a probability.
- `RandomPointDropout` removes points and samples replacements so array length
  remains fixed for callers that require it.
- `RandomShift` applies a common translation to all points in an event.
- `Compose` applies a sequence of transformations.
- `simclr_augmentation` and `attpc_augmentation` provide ready-made generic and
  AT-TPC transform compositions.

The active O16 and combined datasets reuse these transformation classes but
define their selected AT-TPC augmentation list locally.

### `__init__.py`

Marks `src.utils` as a Python package.
