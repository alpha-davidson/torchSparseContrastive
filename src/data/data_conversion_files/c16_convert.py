#!/usr/bin/env python3
"""Convert nested C16 cluster HDF5 data for virtual combined pretraining.

Input structure
---------------
file["cluster"][event_key][cluster_i]["cloud"]

Each cloud must contain at least:
    0 x, 1 y, 2 z, 3 charge

Outputs
-------
<output_prefix>_w_event_keys.npy
    Shape (events, max_hits, 6), with columns:
    0 x, 1 y, 2 z, 3 charge, 4 auxiliary/NND, 5 event index.
    C16 has no NND field, so column 4 is zero.

<output_prefix>_event_lens.npy
    Number of real hits in each retained event.

Empty events are skipped. All clusters belonging to a retained event are
concatenated into one point cloud. The output event array is written through a
numpy memory map, so the full dense array is not held in RAM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


DEFAULT_INPUT = Path("/data/example-C16/example_run_0017_clusters.h5")
DEFAULT_OUTPUT_PREFIX = Path("data/C16")


def _cluster_names(event: h5py.Group) -> list[str]:
    """Return existing cluster groups in numerical order."""
    nclusters = int(event.attrs.get("nclusters", 0))
    names = [f"cluster_{i}" for i in range(nclusters)]
    missing = [name for name in names if name not in event]
    if missing:
        raise KeyError(
            f"Event {event.name!r} declares {nclusters} clusters but is "
            f"missing {missing}"
        )
    return names


def _event_length(event: h5py.Group) -> int:
    total = 0
    for cluster_name in _cluster_names(event):
        cluster = event[cluster_name]
        if "cloud" not in cluster:
            raise KeyError(f"Missing {cluster.name}/cloud")
        cloud = cluster["cloud"]
        if cloud.ndim != 2 or cloud.shape[1] < 4:
            raise ValueError(
                f"{cloud.name!r} has shape {cloud.shape}; expected "
                "(number_of_hits, at least 4)"
            )
        total += int(cloud.shape[0])
    return total


def convert_c16(input_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    data_path = output_prefix.parent / f"{output_prefix.name}_w_event_keys.npy"
    lens_path = output_prefix.parent / f"{output_prefix.name}_event_lens.npy"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as source:
        if "cluster" not in source:
            raise KeyError(
                f"{input_path} has keys {list(source.keys())}; expected 'cluster'"
            )
        root = source["cluster"]
        all_keys = list(root.keys())

        retained: list[tuple[str, int]] = []
        for key in tqdm(all_keys, desc="Scanning C16 events"):
            event = root[key]
            event_length = _event_length(event)
            if event_length > 0:
                retained.append((key, event_length))

        if not retained:
            raise ValueError("No non-empty C16 events were found")

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
            tqdm(retained, desc="Writing C16 events")
        ):
            event_data[output_i] = 0.0
            event = root[key]
            offset = 0

            for cluster_name in _cluster_names(event):
                cloud = np.asarray(
                    event[cluster_name]["cloud"][:, :4],
                    dtype=np.float32,
                )
                end = offset + len(cloud)
                event_data[output_i, offset:end, 0:4] = cloud
                offset = end

            if offset != expected_length:
                raise RuntimeError(
                    f"Event {key!r} changed while reading: expected "
                    f"{expected_length} hits, copied {offset}"
                )

            # Column 4 remains zero because C16 has no NND value.
            event_data[output_i, :offset, 5] = output_i

        event_data.flush()
        del event_data

    print(f"Retained events : {len(event_lens)} / {len(all_keys)}")
    print(f"Maximum hits    : {max_hits}")
    print(f"Event data      : {data_path}")
    print(f"Event lengths   : {lens_path}")
    return data_path, lens_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert nested C16 HDF5 clusters to training numpy files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix used for *_w_event_keys.npy and *_event_lens.npy",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert_c16(args.input, args.output_prefix)
