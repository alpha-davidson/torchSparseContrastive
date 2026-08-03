#!/usr/bin/env python3
"""Plot downstream macro-F1 and accuracy curves with and without pretraining.

The x-axis is downstream training epoch for both conditions. Each line is the
mean validation metric across trials and each shaded region is ±1 sample
standard deviation. Only epochs present in the saved ``history.json`` files are
plotted; missing epochs are never extrapolated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "downstream"
PRETRAINED_COLOR = "#c8102e"
RANDOM_COLOR = "#1f77b4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare validation macro F1 and accuracy across downstream "
            "training epochs for a contrastively pretrained model and random "
            "initialization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prefix", default="O16_UNSAMPLED")
    parser.add_argument(
        "--checkpoint-id",
        default=None,
        help=(
            "Pretrained checkpoint result directory to compare. "
            "Default: checkpoint with the largest pretraining epoch."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Default: <results-dir>/plots/classification/<prefix>/"
            "macro_f1_pretrained_vs_random.png"
        ),
    )
    parser.add_argument(
        "--accuracy-output",
        type=Path,
        default=None,
        help=(
            "Default: <results-dir>/plots/classification/<prefix>/"
            "accuracy_pretrained_vs_random.png"
        ),
    )
    return parser


def load_checkpoint_table(task_root: Path) -> pd.DataFrame:
    path = task_root / "checkpoint_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    required = {"checkpoint_id", "pretrain_epoch"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if table.empty:
        raise ValueError(f"{path} contains no rows")
    return table


def choose_checkpoint(
    checkpoints: pd.DataFrame,
    requested_id: str | None,
) -> tuple[str, int]:
    available = (
        checkpoints[["checkpoint_id", "pretrain_epoch"]]
        .drop_duplicates()
        .sort_values("pretrain_epoch")
    )
    if requested_id is None:
        row = available.iloc[-1]
    else:
        selected = available[available["checkpoint_id"] == requested_id]
        if selected.empty:
            choices = ", ".join(available["checkpoint_id"].astype(str))
            raise ValueError(
                f"Unknown --checkpoint-id {requested_id!r}. Available: {choices}"
            )
        row = selected.iloc[0]
    return str(row["checkpoint_id"]), int(row["pretrain_epoch"])


def load_history_frame(model_dir: Path) -> pd.DataFrame:
    rows = []
    for history_path in sorted(model_dir.glob("trial_*/history.json")):
        try:
            trial = int(history_path.parent.name.removeprefix("trial_"))
        except ValueError as error:
            raise ValueError(
                f"Invalid trial directory name: {history_path.parent.name}"
            ) from error
        history = json.loads(history_path.read_text())
        required_metrics = {"val_f1_macro", "val_accuracy"}
        missing = required_metrics - set(history)
        if missing:
            raise KeyError(
                f"{history_path} is missing histories {sorted(missing)}"
            )
        lengths = {len(history[metric]) for metric in required_metrics}
        if len(lengths) != 1:
            raise ValueError(
                f"{history_path} has inconsistent validation-history lengths"
            )
        for epoch, (macro_f1, accuracy) in enumerate(
            zip(history["val_f1_macro"], history["val_accuracy"]), start=1
        ):
            rows.append(
                {
                    "trial": trial,
                    "downstream_epoch": epoch,
                    "val_f1_macro": float(macro_f1),
                    "val_accuracy": float(accuracy),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No trial_*/history.json files below {model_dir}")
    return pd.DataFrame(rows)


def mean_and_std(
    frame: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, ...]:
    grouped = frame.groupby("downstream_epoch")[metric]
    epochs = np.asarray(sorted(grouped.groups), dtype=int)
    means = grouped.mean().reindex(epochs).to_numpy(dtype=float)
    standard_deviations = (
        grouped.std().reindex(epochs).fillna(0).to_numpy(dtype=float)
    )
    return epochs, means, standard_deviations


def plot_condition(
    axis,
    history: pd.DataFrame,
    *,
    color: str,
    marker: str,
    label: str,
    metric: str,
) -> None:
    epochs, means, standard_deviations = mean_and_std(history, metric)
    marker_interval = max(1, len(epochs) // 10)
    axis.plot(
        epochs,
        means,
        color=color,
        marker=marker,
        markevery=marker_interval,
        linewidth=2.2,
        markersize=6,
        label=label,
    )
    axis.fill_between(
        epochs,
        means - standard_deviations,
        means + standard_deviations,
        color=color,
        alpha=0.18,
    )


def save_comparison_plot(
    pretrained_history: pd.DataFrame,
    random_history: pd.DataFrame,
    *,
    metric: str,
    y_label: str,
    output: Path,
    checkpoint_epoch: int,
    pretrained_trials: int,
    random_trials: int,
    maximum_epoch: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 6))
    plot_condition(
        axis,
        random_history,
        color=RANDOM_COLOR,
        marker="o",
        label=f"New/random initialization (n={random_trials})",
        metric=metric,
    )
    plot_condition(
        axis,
        pretrained_history,
        color=PRETRAINED_COLOR,
        marker="s",
        label=(
            f"Contrastive pretraining, checkpoint {checkpoint_epoch} "
            f"(n={pretrained_trials})"
        ),
        metric=metric,
    )

    axis.set_xlim(1, maximum_epoch)
    axis.set_xlabel("Downstream training epoch")
    axis.set_ylabel(y_label)
    axis.set_title(
        r"$^{16}$O track-count classification: pretrained vs random initialization"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    task_root = args.results_dir / "classification" / args.prefix
    checkpoints = load_checkpoint_table(task_root)
    checkpoint_id, checkpoint_epoch = choose_checkpoint(
        checkpoints, args.checkpoint_id
    )
    pretrained_history = load_history_frame(task_root / checkpoint_id)
    random_history = load_history_frame(task_root / "random_init")

    pretrained_trials = pretrained_history["trial"].nunique()
    random_trials = random_history["trial"].nunique()
    maximum_epoch = max(
        int(pretrained_history["downstream_epoch"].max()),
        int(random_history["downstream_epoch"].max()),
    )

    output_dir = (
        args.results_dir
        / "plots"
        / "classification"
        / args.prefix
    )
    macro_f1_output = args.output or (
        output_dir / "macro_f1_pretrained_vs_random.png"
    )
    accuracy_output = args.accuracy_output or (
        output_dir / "accuracy_pretrained_vs_random.png"
    )

    save_comparison_plot(
        pretrained_history,
        random_history,
        metric="val_f1_macro",
        y_label="Validation macro F1",
        output=macro_f1_output,
        checkpoint_epoch=checkpoint_epoch,
        pretrained_trials=pretrained_trials,
        random_trials=random_trials,
        maximum_epoch=maximum_epoch,
    )
    save_comparison_plot(
        pretrained_history,
        random_history,
        metric="val_accuracy",
        y_label="Validation accuracy",
        output=accuracy_output,
        checkpoint_epoch=checkpoint_epoch,
        pretrained_trials=pretrained_trials,
        random_trials=random_trials,
        maximum_epoch=maximum_epoch,
    )

    print(
        f"Saved macro-F1 comparison to {macro_f1_output}\n"
        f"Saved accuracy comparison to {accuracy_output}\n"
        f"Pretrained checkpoint: {checkpoint_id}\n"
        f"Downstream epochs plotted: {maximum_epoch}\n"
        f"Trials: pretrained={pretrained_trials}, random={random_trials}"
    )


if __name__ == "__main__":
    main()
