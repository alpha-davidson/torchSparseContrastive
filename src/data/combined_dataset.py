"""
combined_o16_ar46_dataset.py
--------------------------------
Virtual combined dataset for multiple AT-TPC numpy files.

The original files remain separate on disk. To the training loop they appear
as one continuous dataset; no combined .npy file is created in memory or on
disk. Each event is read on demand using numpy memory mapping.

Column layout in each *_w_event_keys.npy file (axis 2):
    0  x       mm
    1  y       mm
    2  z       mm
    3  t       time bucket
    4  A       amplitude (charge)
    5  event_idx
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.augmentations import (
    RandomJitter,
    RandomPointDropout,
    RandomRotation,
    RandomScale,
    RandomShift,
)


# Replace these four paths with the real file locations.
DATASETS = {
    "O16": {
        "data_path": "data/O16_w_event_keys.npy",
        "lens_path": "data/O16_event_lens.npy",
    },
    "Ar46": {
        "data_path": "data/Ar46_w_event_keys.npy",
        "lens_path": "data/Ar46_event_lens.npy",
    },
}

RANGES = {
    "MIN_X": -270.0,
    "MAX_X": 270.0,
    "MIN_Y": -270.0,
    "MAX_Y": 270.0,
    "MIN_Z": -185.0,
    "MAX_Z": 1155.0,
    "MIN_LOG_A": 0.0,
    "MAX_LOG_A": 10.80,
}


def _augment(transforms: list, xyz: np.ndarray, feats: np.ndarray):
    """Apply spatial transforms while keeping amplitudes with their hits."""
    combined = np.concatenate([xyz, feats], axis=1)
    for transform in transforms:
        if isinstance(transform, RandomPointDropout):
            combined = transform(combined)
        else:
            combined[:, :3] = transform(combined[:, :3])
    return combined[:, :3].copy(), combined[:, 3:].copy()


def attpc_aug_list() -> list:
    """Augmentations tuned for AT-TPC normalized coordinates."""
    return [
        RandomRotation(axes="z", angle_range=(0, 360)),
        RandomScale(lo=0.9, hi=1.1),
        RandomJitter(sigma=0.02, clip=0.08),
        RandomPointDropout(p=0.2),
        RandomShift(max_shift=0.05),
    ]


class CombinedATTPC(Dataset):
    """Expose multiple AT-TPC numpy datasets as one PyTorch dataset.

    Only the small length arrays and valid-event indexes are held in memory.
    Event data remains in its original files and is read through memory maps
    on demand. Additional isotopes can be supplied through ``datasets`` without
    changing this class.

    Parameters
    ----------
    datasets : mapping of dataset names to data_path/lens_path mappings
    voxel_size : voxelization resolution in normalized [0, 1] coordinates
    min_hits : skip events with fewer hits than this
    aug_list : augmentation transforms; defaults to attpc_aug_list()
    """

    def __init__(
        self,
        datasets: dict[str, dict[str, str]] = DATASETS,
        voxel_size: float = 0.05,
        min_hits: int = 10,
        aug_list: Optional[list] = None,
    ):
        from torchsparse import SparseTensor
        from torchsparse.utils.quantize import sparse_quantize

        if not datasets:
            raise ValueError("At least one dataset must be provided")

        self._SparseTensor = SparseTensor
        self._sparse_quantize = sparse_quantize
        self.voxel_size = voxel_size
        self.aug_list = aug_list if aug_list is not None else attpc_aug_list()

        self._names = []
        self._raw = []
        self._valid_idx = []
        valid_lens = []
        valid_counts = []

        for name, paths in datasets.items():
            missing = {"data_path", "lens_path"} - paths.keys()
            if missing:
                raise KeyError(f"{name} is missing path(s): {sorted(missing)}")

            raw = np.load(paths["data_path"], mmap_mode="r")
            lens_full = np.load(paths["lens_path"])

            if raw.ndim != 3 or raw.shape[2] < 5:
                raise ValueError(
                    f"{name} data must have shape (events, hits, >=5); got {raw.shape}"
                )
            if lens_full.ndim != 1 or len(raw) != len(lens_full):
                raise ValueError(
                    f"{name} data and event_lens must contain the same number of events"
                )
            if np.any(lens_full < 0) or np.any(lens_full > raw.shape[1]):
                raise ValueError(
                    f"{name} event_lens contains values outside [0, {raw.shape[1]}]"
                )

            valid_idx = np.flatnonzero(lens_full >= min_hits)
            self._names.append(name)
            self._raw.append(raw)
            self._valid_idx.append(valid_idx)
            valid_lens.append(lens_full[valid_idx].astype(np.int64, copy=False))
            valid_counts.append(len(valid_idx))

        # Cumulative boundaries translate one global training index into its
        # source dataset and source-local event index.
        self._ends = np.cumsum(valid_counts, dtype=np.int64)
        self.lens = np.concatenate(valid_lens)

        self._xyz_min = np.array(
            [RANGES["MIN_X"], RANGES["MIN_Y"], RANGES["MIN_Z"]], dtype=np.float32
        )
        self._xyz_range = np.array(
            [
                RANGES["MAX_X"] - RANGES["MIN_X"],
                RANGES["MAX_Y"] - RANGES["MIN_Y"],
                RANGES["MAX_Z"] - RANGES["MIN_Z"],
            ],
            dtype=np.float32,
        )
        self._log_a_min = np.float32(RANGES["MIN_LOG_A"])
        self._log_a_range = np.float32(
            RANGES["MAX_LOG_A"] - RANGES["MIN_LOG_A"]
        )

    def __len__(self):
        return int(self._ends[-1])

    def _locate_event(self, i: int):
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(i)

        source_i = int(np.searchsorted(self._ends, i, side="right"))
        source_start = 0 if source_i == 0 else int(self._ends[source_i - 1])
        local_valid_i = i - source_start
        raw_i = int(self._valid_idx[source_i][local_valid_i])
        return source_i, raw_i

    def _load_event(self, i: int):
        source_i, raw_i = self._locate_event(i)
        n_hits = int(self.lens[i])
        event = self._raw[source_i][raw_i, :n_hits]

        xyz = event[:, :3].astype(np.float32)
        amplitude = event[:, 4:5].astype(np.float32)
        xyz = (xyz - self._xyz_min) / self._xyz_range
        amplitude = np.log(np.clip(amplitude, a_min=1.0, a_max=None))
        amplitude = (amplitude - self._log_a_min) / self._log_a_range
        return xyz, amplitude

    def _to_sparse(self, xyz: np.ndarray, feats: np.ndarray):
        xyz = xyz - xyz.min(axis=0, keepdims=True)
        coords_q, idx = self._sparse_quantize(
            xyz, voxel_size=self.voxel_size, return_index=True
        )
        return self._SparseTensor(
            feats=torch.tensor(feats[idx], dtype=torch.float32),
            coords=torch.tensor(coords_q, dtype=torch.int32),
        )

    def __getitem__(self, i: int):
        xyz, feats = self._load_event(i)
        xyz_a, feats_a = _augment(self.aug_list, xyz, feats)
        xyz_b, feats_b = _augment(self.aug_list, xyz, feats)
        return {
            "view_a": self._to_sparse(xyz_a, feats_a),
            "view_b": self._to_sparse(xyz_b, feats_b),
            "original": self._to_sparse(xyz, feats),
        }


# Longer alias for code that follows the original O16Dataset naming style.
CombinedATTPCDataset = CombinedATTPC


def collate_attpc_batch(batch):
    from torchsparse.utils.collate import sparse_collate

    return {
        "view_a": sparse_collate([item["view_a"] for item in batch]),
        "view_b": sparse_collate([item["view_b"] for item in batch]),
        "original": sparse_collate([item["original"] for item in batch]),
    }


def make_attpc_dataloader(
    datasets: dict[str, dict[str, str]] = DATASETS,
    batch_size: int = 16,
    voxel_size: float = 0.05,
    min_hits: int = 10,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    aug_list: Optional[list] = None,
) -> DataLoader:
    dataset = CombinedATTPC(
        datasets=datasets,
        voxel_size=voxel_size,
        min_hits=min_hits,
        aug_list=aug_list,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # BatchNorm and the contrastive loss both require more than one event.
        # Discard an incomplete final batch instead of allowing a size-1 batch.
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_attpc_batch,
        pin_memory=num_workers > 0 and torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Smoke test -- same pattern as the original O16 loader
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = make_attpc_dataloader(batch_size=4, shuffle=False)
    print(f"Dataset size: {len(loader.dataset)} events")
    print(
        "Sources: "
        + ", ".join(
            f"{name}={len(valid_idx)}"
            for name, valid_idx in zip(loader.dataset._names, loader.dataset._valid_idx)
        )
    )

    batch = next(iter(loader))
    view_a = batch["view_a"]
    print(f"view_a  feats : {view_a.feats.shape}   dtype={view_a.feats.dtype}")
    print(f"view_a  coords: {view_a.coords.shape}  dtype={view_a.coords.dtype}")
    print("Smoke-test passed.")
