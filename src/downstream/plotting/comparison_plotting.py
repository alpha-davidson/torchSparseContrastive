#!/usr/bin/env python3
"""Plot dynamic downstream classification performance by pretraining epoch."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "downstream"
METRICS = {
    "accuracy": "Accuracy",
    "f1_macro": "Macro F1",
    "f1_weighted": "Weighted F1",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dynamically evaluated contrastive checkpoints and the "
            "random-initialization baseline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prefix", default="O16_UNSAMPLED")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <results-dir>/plots/classification/<prefix>",
    )
    parser.add_argument(
        "--title",
        default=r"$^{16}$O track-count classification",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=tuple(METRICS),
        default=[],
        help="Metric to plot; repeat as needed. Default: all metrics",
    )
    return parser


def load_checkpoint_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run the dynamic checkpoint evaluator first."
        )
    frame = pd.read_csv(path)
    required = {
        "checkpoint_id",
        "pretrain_epoch",
        "trial",
        *METRICS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no results")
    return frame.sort_values(["pretrain_epoch", "trial"])


def load_random_results(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        print(f"Random baseline not found at {path}; plotting checkpoints only.")
        return None
    frame = pd.read_csv(path)
    missing = {"trial", *METRICS} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return frame


def _series(frame: pd.DataFrame, metric: str):
    grouped = frame.groupby("pretrain_epoch")[metric]
    epochs = np.asarray(sorted(grouped.groups), dtype=int)
    means = grouped.mean().reindex(epochs).to_numpy(dtype=float)
    stds = grouped.std().reindex(epochs).fillna(0).to_numpy(dtype=float)
    return epochs, means, stds


def plot_metric(
    checkpoints: pd.DataFrame,
    random_results: pd.DataFrame | None,
    metric: str,
    title: str,
    output: Path,
) -> None:
    epochs, means, stds = _series(checkpoints, metric)
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.plot(
        epochs,
        means,
        marker="o",
        color="#1f77b4",
        label="Contrastive checkpoint",
    )
    axis.fill_between(
        epochs,
        means - stds,
        means + stds,
        color="#1f77b4",
        alpha=0.18,
        label="Checkpoint ±1 SD",
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
    axis.set_title(f"{title}\n{METRICS[metric]} by checkpoint")
    axis.set_xticks(epochs)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_trials(
    checkpoints: pd.DataFrame,
    metric: str,
    title: str,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    for trial, trial_frame in checkpoints.groupby("trial"):
        axis.plot(
            trial_frame["pretrain_epoch"],
            trial_frame[metric],
            marker="o",
            alpha=0.75,
            label=f"Trial {int(trial)}",
        )
    axis.set_xlabel("Contrastive pretraining epoch")
    axis.set_ylabel(METRICS[metric])
    axis.set_title(f"{title}\nPer-trial {METRICS[metric]}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    checkpoints: pd.DataFrame,
    random_results: pd.DataFrame | None,
    metrics: list[str],
    output: Path,
) -> None:
    lines = []
    for metric in metrics:
        epochs, means, stds = _series(checkpoints, metric)
        best_index = int(np.argmax(means))
        lines.append(
            f"{METRICS[metric]}: best checkpoint epoch={epochs[best_index]}, "
            f"mean={means[best_index]:.6f}, std={stds[best_index]:.6f}"
        )
        if random_results is not None:
            baseline = float(random_results[metric].mean())
            lines.append(
                f"  random-init mean={baseline:.6f}; "
                f"difference={means[best_index] - baseline:+.6f}"
            )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    task_root = args.results_dir / "classification" / args.prefix
    output_dir = args.output_dir or (
        args.results_dir / "plots" / "classification" / args.prefix
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = load_checkpoint_results(
        task_root / "checkpoint_summary.csv"
    )
    random_results = load_random_results(
        task_root / "random_init" / "summary.csv"
    )
    metrics = args.metric or list(METRICS)

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
    aggregate.to_csv(output_dir / "classification_checkpoint_summary.csv", index=False)

    for metric in metrics:
        plot_metric(
            checkpoints,
            random_results,
            metric,
            args.title,
            output_dir / f"{metric}_by_pretrain_epoch.png",
        )
        plot_trials(
            checkpoints,
            metric,
            args.title,
            output_dir / f"{metric}_per_trial.png",
        )
    write_summary(
        checkpoints,
        random_results,
        metrics,
        output_dir / "summary.txt",
    )
    print(f"Saved classification comparison plots to {output_dir}")


if __name__ == "__main__":
    main()
