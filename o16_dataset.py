"""
o16_dataset.py
--------------
Dataset for O16 AT-TPC data stored as numpy arrays from convert-data.py.

Column layout in O16_w_event_keys.npy  (axis-2):
    0  x       mm
    1  y       mm
    2  z       mm
    3  t       time bucket
    4  A       amplitude (charge)
    5  event_idx

Produces two augmented SparseTensor views for SimCLR training.
in_channels=1  (amplitude A)
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from augmentations import (
    RandomRotation, RandomScale, RandomJitter,
    RandomPointDropout, RandomShift,
)


# ---------------------------------------------------------------------------
# Feature-aware augmentation helper
# ---------------------------------------------------------------------------

def _augment(transforms: list, xyz: np.ndarray, feats: np.ndarray):
    """
    Apply spatial augmentations to xyz while carrying feats along.

    RandomPointDropout selects rows so it works on any (N, D) array.
    All other transforms are spatial-only and touch only the xyz columns.
    """
    combined = np.concatenate([xyz, feats], axis=1)   # (N, 3+C)
    for t in transforms:
        if isinstance(t, RandomPointDropout):
            combined = t(combined)
        else:
            combined[:, :3] = t(combined[:, :3])
    return combined[:, :3].copy(), combined[:, 3:].copy()


def attpc_aug_list() -> list:
    """Augmentation transforms tuned for AT-TPC on normalised [0,1] coords."""
    return [
        RandomRotation(axes="y", angle_range=(0, 360)),
        RandomScale(lo=0.9, hi=1.1),
        RandomJitter(sigma=0.02, clip=0.08),
        RandomPointDropout(p=0.2),
        RandomShift(max_shift=0.05),
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class O16Dataset(Dataset):
    """
    Loads O16 AT-TPC data from numpy arrays produced by convert-data.py.

    Parameters
    ----------
    data_path  : path to O16_w_event_keys.npy
    lens_path  : path to O16_event_lens.npy
    voxel_size : voxelisation resolution in normalised [0,1] coordinates
    min_hits   : skip events with fewer hits than this
    aug_list   : list of augmentation transforms; defaults to attpc_aug_list()
    """

    def __init__(
        self,
        data_path: str = "data/O16_w_event_keys.npy",
        lens_path: str = "data/O16_event_lens.npy",
        voxel_size: float = 0.05,
        min_hits: int = 10,
        aug_list: Optional[list] = None,
    ):
        from torchsparse import SparseTensor
        from torchsparse.utils.quantize import sparse_quantize
        self._SparseTensor    = SparseTensor
        self._sparse_quantize = sparse_quantize

        self.voxel_size = voxel_size
        self.aug_list   = aug_list if aug_list is not None else attpc_aug_list()

        # mmap_mode='r' reads slices from disk on demand — avoids loading 12 GB into RAM
        # Keep raw as mmap; fancy-index (raw[valid]) would copy everything into RAM
        self._raw       = np.load(data_path, mmap_mode='r')   # (N_events, max_hits, 6)
        lens_full       = np.load(lens_path)                   # (N_events,) — small, load fully
        self._valid_idx = np.where(lens_full >= min_hits)[0]   # indices into _raw
        self.lens       = lens_full[self._valid_idx]           # lengths for valid events

    def __len__(self):
        return len(self.lens)

    def _load_event(self, i: int):
        """Return xyz (N,3) and amplitude feats (N,1), both normalised to [0,1]."""
        n     = self.lens[i]
        i_raw = self._valid_idx[i]
        ev    = self._raw[i_raw, :n]                # (n, 6) — reads one row from mmap

        xyz = ev[:, :3].astype(np.float32)
        A   = ev[:, 4:5].astype(np.float32)

        # normalise xyz per-event to [0, 1]
        lo  = xyz.min(axis=0, keepdims=True)
        hi  = xyz.max(axis=0, keepdims=True)
        rng = np.where((hi - lo) > 0, hi - lo, 1.0)
        xyz = (xyz - lo) / rng

        # normalise amplitude to [0, 1]
        A_lo, A_hi = float(A.min()), float(A.max())
        A = (A - A_lo) / max(A_hi - A_lo, 1e-6)

        return xyz, A

    def _to_sparse(self, xyz: np.ndarray, feats: np.ndarray):
        coords_q, idx = self._sparse_quantize(
            xyz, voxel_size=self.voxel_size, return_index=True
        )
        return self._SparseTensor(
            feats=torch.tensor(feats[idx],  dtype=torch.float32),
            coords=torch.tensor(coords_q,   dtype=torch.int32),
        )

    def __getitem__(self, i: int):
        xyz, feats = self._load_event(i)

        xyz1, f1 = _augment(self.aug_list, xyz, feats)
        xyz2, f2 = _augment(self.aug_list, xyz, feats)

        return {
            "view_a":   self._to_sparse(xyz1, f1),
            "view_b":   self._to_sparse(xyz2, f2),
            "original": self._to_sparse(xyz,  feats),
        }


# ---------------------------------------------------------------------------
# Collate + DataLoader factory
# ---------------------------------------------------------------------------

def collate_o16_batch(batch):
    from torchsparse.utils.collate import sparse_collate
    return {
        "view_a":   sparse_collate([b["view_a"]   for b in batch]),
        "view_b":   sparse_collate([b["view_b"]   for b in batch]),
        "original": sparse_collate([b["original"] for b in batch]),
    }


def make_o16_dataloader(
    data_path: str = "data/O16_w_event_keys.npy",
    lens_path: str = "data/O16_event_lens.npy",
    batch_size: int = 16,
    voxel_size: float = 0.05,
    min_hits: int = 10,
    shuffle: bool = True,
    num_workers: int = 0,
    aug_list: Optional[list] = None,
) -> DataLoader:
    dataset = O16Dataset(
        data_path=data_path,
        lens_path=lens_path,
        voxel_size=voxel_size,
        min_hits=min_hits,
        aug_list=aug_list,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_o16_batch,
        pin_memory=num_workers > 0 and torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = make_o16_dataloader(batch_size=4, shuffle=False)
    print(f"Dataset size: {len(loader.dataset)} events")

    batch = next(iter(loader))
    va = batch["view_a"]
    print(f"view_a  feats : {va.feats.shape}   dtype={va.feats.dtype}")
    print(f"view_a  coords: {va.coords.shape}  dtype={va.coords.dtype}")
    print("Smoke-test passed.")
