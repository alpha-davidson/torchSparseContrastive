#!/usr/bin/env python3
"""Convert experimental Mg22 HDF5 point clouds for combined pretraining.

Expected event layout
---------------------
Each event may be either:

1. A one-dimensional structured dataset with fields:
       x, y, z, t, A
2. A two-dimensional dataset with at least five columns:
       0 x, 1 y, 2 z, 3 time, 4 amplitude

The event datasets may be located directly at the HDF5 root or inside one
top-level group. Use --group when automatic group detection is ambiguous.

Outputs
-------
<output_prefix>_w_event_keys.npy
    Shape (events, max_hits, 6), with columns:
    0 x, 1 y, 2 z, 3 time, 4 amplitude, 5 event index.

<output_prefix>_event_lens.npy
    Number of real hits in every retained event.

Empty events are skipped. The padded event array is written through a numpy
memory map so it is not assembled fully in RAM.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import h5py
import numpy as np
from tqdm import tqdm


DEFAULT_INPUT = Path("/data/22Mg/point_clouds/experimental/22Mg_alpha_exp.h5")
DEFAULT_OUTPUT_PREFIX = Path("data/Mg22")
REQUIRED_FIELDS = ("x", "y", "z", "t", "A")


def _natural_key(name: str):
    """Sort names such as Event_[2] before Event_[10]."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def _is_point_cloud_dataset(node) -> bool:
    if not isinstance(node, h5py.Dataset):
        return False
    if node.ndim == 2 and node.shape[1] >= 5:
        return True
    field_names = node.dtype.names
    return (
        node.ndim == 1
        and field_names is not None
        and all(field in field_names for field in REQUIRED_FIELDS)
    )


def _point_cloud_dataset(node, event_name: str) -> h5py.Dataset | None:
    """Find a supported point-cloud dataset at or below an event node."""
    if isinstance(node, h5py.Dataset):
        return node if _is_point_cloud_dataset(node) else None
    if not isinstance(node, h5py.Group):
        return None

    # Prefer conventional point-cloud dataset names when present.
    preferred_names = ("cloud", "point_cloud", "points", "hits", "data")
    for preferred in preferred_names:
        if preferred in node:
            candidate = node[preferred]
            if _is_point_cloud_dataset(candidate):
                return candidate

    candidates: list[h5py.Dataset] = []

    def collect(_name, value):
        if _is_point_cloud_dataset(value):
            candidates.append(value)

    node.visititems(collect)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        paths = [dataset.name for dataset in candidates[:10]]
        raise ValueError(
            f"Event {event_name!r} contains multiple point-cloud datasets "
            f"{paths}. "
            "Add the correct dataset name to preferred_names in "
            "_point_cloud_dataset()."
        )
    return None


def _event_datasets(group: h5py.Group) -> list[tuple[str, h5py.Dataset]]:
    """Return children that resolve to one point-cloud dataset each."""
    events = []
    for key in sorted(group.keys(), key=_natural_key):
        dataset = _point_cloud_dataset(group[key], key)
        if dataset is not None:
            events.append((key, dataset))
    return events


def _select_event_root(
    source: h5py.File,
    group_name: str | None,
) -> tuple[h5py.Group, list[tuple[str, h5py.Dataset]]]:
    if group_name is not None:
        if group_name not in source:
            raise KeyError(
                f"Group {group_name!r} not found. Top-level keys: "
                f"{list(source.keys())}"
            )
        root = source[group_name]
        if not isinstance(root, h5py.Group):
            raise TypeError(f"{root.name!r} is not an HDF5 group")
        events = _event_datasets(root)
        if not events:
            raise ValueError(
                f"Group {root.name!r} contains no recognizable event point clouds"
            )
        return root, events

    # Support either direct event datasets or Event_[N] groups containing a
    # point-cloud dataset.
    root_events = _event_datasets(source)
    if root_events:
        return source, root_events

    # Otherwise accept exactly one top-level group containing event datasets.
    candidates = []
    for key, value in source.items():
        if isinstance(value, h5py.Group):
            events = _event_datasets(value)
            if events:
                candidates.append((value, events))

    if len(candidates) == 1:
        root, events = candidates[0]
        print(f"Automatically selected event group: {root.name}")
        return root, events

    candidate_names = [root.name for root, _ in candidates]
    top_level_preview = list(source.keys())[:10]
    raise ValueError(
        "Could not determine the Mg22 event group automatically. "
        f"First top-level keys: {top_level_preview}; candidate groups: "
        f"{candidate_names}. Pass --group with the correct group name."
    )


def _validate_event(key: str, event: h5py.Dataset) -> int:
    if not _is_point_cloud_dataset(event):
        raise ValueError(
            f"Event {key!r} has shape {event.shape} and dtype {event.dtype}; "
            "expected a structured array with x,y,z,t,A fields or a "
            "(number_of_hits, at least 5) array"
        )
    return int(event.shape[0])


def _read_event_values(event: h5py.Dataset) -> np.ndarray:
    """Return x,y,z,time,amplitude as a regular float32 (N, 5) array."""
    if event.dtype.names is not None:
        structured = event[...]
        values = np.empty((len(structured), 5), dtype=np.float32)
        for column, field in enumerate(REQUIRED_FIELDS):
            values[:, column] = structured[field]
        return values
    return np.asarray(event[:, :5], dtype=np.float32)


def convert_mg22(
    input_path: Path,
    output_prefix: Path,
    group_name: str | None = None,
) -> tuple[Path, Path]:
    data_path = output_prefix.parent / f"{output_prefix.name}_w_event_keys.npy"
    lens_path = output_prefix.parent / f"{output_prefix.name}_event_lens.npy"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as source:
        root, events = _select_event_root(source, group_name)
        print(f"Reading events from: {root.name}")

        retained: list[tuple[str, int]] = []
        for key, event in tqdm(events, desc="Scanning Mg22 events"):
            event_length = _validate_event(key, event)
            if event_length > 0:
                retained.append((key, event_length))

        if not retained:
            raise ValueError("No non-empty Mg22 events were found")

        event_lens = np.fromiter(
            (length for _, length in retained),
            dtype=np.int64,
            count=len(retained),
        )
        np.save(lens_path, event_lens)

        max_hits = int(event_lens.max())
        event_data = np.lib.format.open_memmap(
            data_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(retained), max_hits, 6),
        )

        for output_i, (key, expected_length) in enumerate(
            tqdm(retained, desc="Writing Mg22 events")
        ):
            event_data[output_i] = 0.0
            event = _point_cloud_dataset(root[key], key)
            if event is None:
                raise RuntimeError(f"Could not reopen point cloud for event {key!r}")
            values = _read_event_values(event)
            if len(values) != expected_length:
                raise RuntimeError(
                    f"Event {key!r} changed while reading: expected "
                    f"{expected_length} hits, copied {len(values)}"
                )

            event_data[output_i, :expected_length, :5] = values
            event_data[output_i, :expected_length, 5] = output_i

        event_data.flush()
        del event_data

    print(f"Retained events : {len(event_lens)} / {len(events)}")
    print(f"Maximum hits    : {max_hits}")
    print(f"Event data      : {data_path}")
    print(f"Event lengths   : {lens_path}")
    return data_path, lens_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Mg22 HDF5 events to training numpy files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix used for *_w_event_keys.npy and *_event_lens.npy",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="Optional top-level HDF5 group containing the event datasets",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert_mg22(args.input, args.output_prefix, args.group)
