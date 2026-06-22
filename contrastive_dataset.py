"""
Contrastive dataset for SimCLR training on ShapeNet or AT-TPC data.

Exports
-------
ContrastiveShapeNetDataset  — Dataset class
collate_contrastive_batch   — collate function for DataLoader
make_contrastive_dataloader — convenience DataLoader factory
load_metadata               — returns a summary dict about the .pt file
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from augmentations import simclr_augmentation, attpc_augmentation, Compose


class ContrastiveShapeNetDataset(Dataset):
    """
    Returns two augmented views of each point cloud for SimCLR training.

    Parameters
    ----------
    path         : path to .pt file (shapenet_simple.pt or _large.pt)
    voxel_size   : voxelisation resolution
    split        : 'train' | 'val' | 'test' | None
    augmentation : Compose pipeline; defaults to simclr_augmentation()
    use_normals  : if True, concatenates xyz+normals (6ch); else xyz (3ch)
    """

    def __init__(
        self,
        path: str,
        voxel_size: float = 0.05,
        split: Optional[str] = None,
        augmentation: Optional[Compose] = None,
        use_normals: bool = False,   # default False to match --in-channels 3
    ):
        from torchsparse import SparseTensor
        from torchsparse.utils.quantize import sparse_quantize

        self._SparseTensor    = SparseTensor
        self._sparse_quantize = sparse_quantize

        raw = torch.load(path, map_location="cpu", weights_only=False)
        self.class_names = raw["class_names"]
        self.voxel_size  = voxel_size
        self.use_normals = use_normals
        self.aug         = augmentation if augmentation is not None else simclr_augmentation()

        # select split
        if "train_idx" in raw and split is not None:
            idx = raw[f"{split}_idx"].long()
        else:
            idx = torch.arange(len(raw["labels"]))

        self.points  = raw["points"][idx]
        self.normals = raw["normals"][idx]
        self.labels  = raw["labels"][idx]

    def __len__(self):
        return len(self.labels)

    def _to_sparse(self, pts: np.ndarray, nrm: np.ndarray):
        coords_q, idx = self._sparse_quantize(
            pts, voxel_size=self.voxel_size, return_index=True
        )
        feats = np.concatenate([pts, nrm], axis=1)[idx] if self.use_normals else pts[idx]
        return self._SparseTensor(
            feats=torch.tensor(feats,     dtype=torch.float32),
            coords=torch.tensor(coords_q, dtype=torch.int32),
        )

    def __getitem__(self, i):
        pts = self.points[i].numpy()
        nrm = self.normals[i].numpy()

        pts1 = self.aug(pts)
        pts2 = self.aug(pts)

        return {
            "view_a":   self._to_sparse(pts1, nrm),
            "view_b":   self._to_sparse(pts2, nrm),
            "original": self._to_sparse(pts,  nrm),
            "label":    torch.tensor(int(self.labels[i]), dtype=torch.long),
        }


def collate_contrastive_batch(batch):
    from torchsparse.utils.collate import sparse_collate
    return {
        "view_a":   sparse_collate([b["view_a"]   for b in batch]),
        "view_b":   sparse_collate([b["view_b"]   for b in batch]),
        "original": sparse_collate([b["original"] for b in batch]),
        "label":    torch.stack(  [b["label"]    for b in batch]),
    }


def make_contrastive_dataloader(
    path: str,
    batch_size: int = 16,
    voxel_size: float = 0.05,
    split: Optional[str] = "train",
    shuffle: bool = True,
    num_workers: int = 0,
    use_normals: bool = False,
    augmentation: Optional[Compose] = None,
) -> DataLoader:
    """
    Convenience factory used by train_contrastive.py.

    Returns a DataLoader over ContrastiveShapeNetDataset.
    """
    dataset = ContrastiveShapeNetDataset(
        path=path,
        voxel_size=voxel_size,
        split=split,
        augmentation=augmentation,
        use_normals=use_normals,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=num_workers > 0 and torch.cuda.is_available(),
    )


def load_metadata(path: str) -> dict:
    """
    Return a summary dict about the dataset file.
    Used by train_contrastive.py to print dataset info at startup.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    meta = {
        "path":        str(path),
        "num_samples": len(raw["labels"]),
        "num_classes": len(raw["class_names"]),
        "class_names": raw["class_names"],
        "num_points":  raw.get("num_points", raw["points"].shape[1]),
    }
    if "train_idx" in raw:
        meta["train_samples"] = len(raw["train_idx"])
        meta["val_samples"]   = len(raw["val_idx"])
        meta["test_samples"]  = len(raw["test_idx"])
    return meta