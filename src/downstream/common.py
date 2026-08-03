"""Shared path, split, checkpoint, and reproducibility helpers."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "downstream"
DEFAULT_VOXEL_SIZE = 1 / 256


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def checkpoint_name(path: Path) -> str:
    return safe_name(f"{path.parent.name}__{path.stem}")


def discover_checkpoints(checkpoints, checkpoint_dirs, patterns):
    """Return deterministic, de-duplicated checkpoint paths."""
    found = []
    for value in checkpoints or []:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        found.append(path)
    for directory_value in checkpoint_dirs or []:
        directory = Path(directory_value).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(
                f"checkpoint directory does not exist: {directory}"
            )
        for pattern in patterns or ["epoch_*.pt"]:
            found.extend(sorted(directory.glob(pattern)))
    unique = list(dict.fromkeys(path.resolve() for path in found))
    if not unique:
        raise ValueError(
            "No checkpoints found. Supply --checkpoint or --checkpoint-dir."
        )
    return unique


def load_checkpoint_config(checkpoint: Path, config_override=None):
    config_path = (
        Path(config_override).expanduser().resolve()
        if config_override is not None
        else checkpoint.parent / "run_config.json"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No run_config.json found for {checkpoint}. "
            "Supply --config explicitly so preprocessing matches pretraining."
        )
    config = json.loads(config_path.read_text())
    required = ("voxel_size", "in_channels")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"{config_path} is missing {missing}")
    config["_config_path"] = str(config_path)
    return config


def configure_torchsparse(config):
    import torchsparse.backends

    ratio = float(config.get("hash_rsv_ratio", 8.0))
    torchsparse.backends.hash_rsv_ratio = ratio
    return ratio


def _split_paths(data_dir: Path, prefix: str, split: str):
    stem = data_dir / f"{prefix}_{split}"
    return {
        "combined": stem.with_suffix(".npy"),
        "features": data_dir / f"{prefix}_{split}_features.npy",
        "labels": data_dir / f"{prefix}_{split}_labels.npy",
        "lengths": data_dir / f"{prefix}_{split}_lens.npy",
    }


def load_classification_split(data_dir, prefix, split):
    """Load one canonical event-classification split without sampling."""
    data_dir = Path(data_dir).expanduser().resolve()
    paths = _split_paths(data_dir, prefix, split)
    if not paths["lengths"].is_file():
        raise FileNotFoundError(
            f"Missing required event lengths: {paths['lengths']}"
        )
    lengths = np.load(paths["lengths"], mmap_mode="r")
    if paths["combined"].is_file():
        combined = np.load(paths["combined"], mmap_mode="r")
        if combined.ndim != 3 or combined.shape[2] < 5:
            raise ValueError(
                f"{paths['combined']} must have columns x,y,z,q,label"
            )
        features = combined[:, :, :4]
        labels = combined[:, 0, 4].astype(np.int64)
    elif paths["features"].is_file() and paths["labels"].is_file():
        features = np.load(paths["features"], mmap_mode="r")
        labels = np.load(paths["labels"], mmap_mode="r").astype(np.int64)
    else:
        raise FileNotFoundError(
            f"Missing classification split {split}; expected either "
            f"{paths['combined']} or both {paths['features']} and "
            f"{paths['labels']}"
        )
    return features, labels, lengths


def load_pointwise_split(
    data_dir,
    prefix,
    split,
    label_suffix,
):
    """Load pointwise features, labels, and lengths by explicit convention."""
    data_dir = Path(data_dir).expanduser().resolve()
    feature_path = data_dir / f"{prefix}_{split}_features.npy"
    label_path = data_dir / f"{prefix}_{split}_{label_suffix}.npy"
    length_path = data_dir / f"{prefix}_{split}_lens.npy"
    missing = [
        path
        for path in (feature_path, label_path, length_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing pointwise split files: "
            + ", ".join(str(path) for path in missing)
        )
    return (
        np.load(feature_path, mmap_mode="r"),
        np.load(label_path, mmap_mode="r"),
        np.load(length_path, mmap_mode="r"),
    )


def load_classification_splits(data_dir, prefix):
    return {
        split: load_classification_split(data_dir, prefix, split)
        for split in ("train", "val", "test")
    }


def load_pointwise_splits(data_dir, prefix, label_suffix):
    return {
        split: load_pointwise_split(
            data_dir, prefix, split, label_suffix
        )
        for split in ("train", "val", "test")
    }


def encoder_kwargs(config):
    return {
        "in_channels": int(config["in_channels"]),
        "proj_hidden_dim": int(config.get("proj_hidden_dim", 512)),
        "proj_out_dim": int(config.get("proj_out_dim", 128)),
        "temperature": float(config.get("temperature", 0.1)),
        "use_final_bn": bool(config.get("final_bn", False)),
    }
