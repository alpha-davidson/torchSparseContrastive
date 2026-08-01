#!/usr/bin/env python3
"""Plot dynamic downstream regression performance by pretraining epoch."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "downstream"
METRICS = {"r2": "R²", "rmse": "RMSE", "mae": "MAE"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dynamically evaluated regression checkpoints and the "
            "random-initialization baseline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <results-dir>/plots/regression/<prefix>",
    )
    parser.add_argument("--title", default="Pointwise regression")
    parser.add_argument(
        "--metric",
        action="append",
        choices=tuple(METRICS),
        default=[],
        help="Metric to plot; repeat as needed. Default: all metrics",
    )
    return parser


def _load_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required results file: {path}")
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no results")
    return frame


def _load_optional(path: Path, required: set[str]) -> pd.DataFrame | None:
    if not path.is_file():
        print(f"Optional comparison file not found: {path}")
        return None
    return _load_csv(path, required)


def _series(frame: pd.DataFrame, metric: str):
    grouped = frame.groupby("pretrain_epoch")[metric]
    epochs = np.asarray(sorted(grouped.groups), dtype=int)
    means = grouped.mean().reindex(epochs).to_numpy(dtype=float)
    stds = grouped.std().reindex(epochs).fillna(0).to_numpy(dtype=float)
    return epochs, means, stds


def _plot_metric(
    checkpoints: pd.DataFrame,
    random_results: pd.DataFrame | None,
    metric: str,
    title: str,
    output: Path,
    suffix: str = "",
) -> None:
    epochs, means, stds = _series(checkpoints, metric)
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.plot(epochs, means, "o-", color="#c8102e", label="Contrastive checkpoint")
    axis.fill_between(
        epochs, means - stds, means + stds, color="#c8102e", alpha=0.18
    )
    if random_results is not None:
        baseline_mean = float(random_results[metric].mean())
        baseline_std = float(random_results[metric].std(ddof=1))
        if not np.isfinite(baseline_std):
            baseline_std = 0.0
        axis.axhline(
            baseline_mean,
            color="black",
            linestyle="--",
            label="Random initialization",
        )
        axis.axhspan(
            baseline_mean - baseline_std,
            baseline_mean + baseline_std,
            color="black",
            alpha=0.10,
        )
    axis.set_xlabel("Contrastive pretraining epoch")
    axis.set_ylabel(METRICS[metric])
    axis.set_title(f"{title}{suffix}\n{METRICS[metric]} by checkpoint")
    axis.set_xticks(epochs)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _epoch_from_checkpoint(value: str) -> int:
    match = re.search(r"epoch_(\d+)", str(value))
    return int(match.group(1)) if match else -1


def _plot_physical_targets(
    physical: pd.DataFrame,
    random_physical: pd.DataFrame | None,
    metrics: list[str],
    title: str,
    output_dir: Path,
) -> None:
    physical = physical.copy()
    physical["pretrain_epoch"] = physical["checkpoint"].map(
        _epoch_from_checkpoint
    )
    if (physical["pretrain_epoch"] < 0).any():
        raise ValueError(
            "Could not infer pretraining epoch from every checkpoint value in "
            "physical_checkpoint_summary.csv"
        )
    for target, target_frame in physical.groupby("label"):
        baseline = None
        if random_physical is not None:
            baseline = random_physical[random_physical["label"] == target]
            if baseline.empty:
                baseline = None
        safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target))
        for metric in metrics:
            _plot_metric(
                target_frame,
                baseline,
                metric,
                title,
                output_dir / f"physical_{metric}_{safe_target}.png",
                suffix=f" — {target}",
            )


def main() -> None:
    args = build_parser().parse_args()
    task_root = args.results_dir / "regression" / args.prefix
    output_dir = args.output_dir or (
        args.results_dir / "plots" / "regression" / args.prefix
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = args.metric or list(METRICS)

    checkpoints = _load_csv(
        task_root / "checkpoint_summary.csv",
        {"checkpoint_id", "pretrain_epoch", "trial", *METRICS},
    ).sort_values(["pretrain_epoch", "trial"])
    random_results = _load_optional(
        task_root / "random_init" / "scaled_summary.csv",
        {"trial", *METRICS},
    )

    aggregate = (
        checkpoints.groupby(["checkpoint_id", "pretrain_epoch"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in aggregate.columns
    ]
    aggregate.to_csv(output_dir / "regression_checkpoint_summary.csv", index=False)

    summary_lines = []
    for metric in metrics:
        _plot_metric(
            checkpoints,
            random_results,
            metric,
            args.title,
            output_dir / f"scaled_{metric}_by_pretrain_epoch.png",
        )
        epochs, means, stds = _series(checkpoints, metric)
        best_index = int(np.argmax(means) if metric == "r2" else np.argmin(means))
        summary_lines.append(
            f"{METRICS[metric]}: best checkpoint epoch={epochs[best_index]}, "
            f"mean={means[best_index]:.6f}, std={stds[best_index]:.6f}"
        )

    physical = _load_optional(
        task_root / "physical_checkpoint_summary.csv",
        {"checkpoint", "trial", "label", *METRICS},
    )
    if physical is not None:
        random_physical = _load_optional(
            task_root / "random_init" / "physical_summary.csv",
            {"trial", "label", *METRICS},
        )
        _plot_physical_targets(
            physical, random_physical, metrics, args.title, output_dir
        )

    (output_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n"
    )
    print(f"Saved regression comparison plots to {output_dir}")


if __name__ == "__main__":
    main()
