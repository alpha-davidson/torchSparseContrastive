#!/usr/bin/env python3
"""Recompute downstream metrics from saved test predictions.

The downstream evaluators already write per-trial reports. This utility is an
independent integrity check: it scans their saved ``test_*`` arrays, recomputes
metrics, and writes aggregate confusion matrices or regression scatter plots.
It never loads a model and does not require a GPU.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "downstream"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute downstream metrics from saved test labels and "
            "probabilities/predictions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        default="classification",
    )
    parser.add_argument("--prefix", default="O16_UNSAMPLED")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <results-dir>/plots/<task>/<prefix>/artifact_analysis",
    )
    parser.add_argument(
        "--class-name",
        action="append",
        default=[],
        help="Optional display name for each class, supplied in class-id order",
    )
    parser.add_argument(
        "--target-name",
        action="append",
        default=[],
        help="Optional regression target names, supplied in channel order",
    )
    return parser


def _trial_number(path: Path) -> int:
    match = re.fullmatch(r"trial_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _pretrain_epoch(model_name: str) -> int:
    match = re.search(r"epoch_(\d+)", model_name)
    return int(match.group(1)) if match else -1


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _trial_directories(task_root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    if not task_root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {task_root}")
    for model_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
        for trial_dir in sorted(model_dir.glob("trial_*"), key=_trial_number):
            if _trial_number(trial_dir) >= 0:
                found.append((model_dir.name, trial_dir))
    if not found:
        raise FileNotFoundError(f"No trial directories found below {task_root}")
    return found


def _plot_confusion(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    output: Path,
) -> None:
    matrix = confusion_matrix(
        labels, predictions, labels=list(range(len(class_names)))
    )
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:d}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_classification(
    task_root: Path,
    output_dir: Path,
    requested_names: list[str],
) -> pd.DataFrame:
    rows = []
    combined: dict[str, dict[str, list[np.ndarray]]] = {}
    for model_name, trial_dir in _trial_directories(task_root):
        probability_path = trial_dir / "test_probabilities.npy"
        label_path = trial_dir / "test_labels.npy"
        if not probability_path.is_file() or not label_path.is_file():
            print(f"Skipping incomplete trial: {trial_dir}")
            continue

        probabilities = np.load(probability_path)
        labels = np.load(label_path).reshape(-1).astype(np.int64)
        if probabilities.ndim != 2 or len(probabilities) != len(labels):
            raise ValueError(
                f"Shape mismatch in {trial_dir}: probabilities "
                f"{probabilities.shape}, labels {labels.shape}"
            )
        predictions = probabilities.argmax(axis=1)
        rows.append(
            {
                "model": model_name,
                "pretrain_epoch": _pretrain_epoch(model_name),
                "trial": _trial_number(trial_dir),
                "accuracy": accuracy_score(labels, predictions),
                "f1_macro": f1_score(
                    labels, predictions, average="macro", zero_division=0
                ),
                "f1_weighted": f1_score(
                    labels, predictions, average="weighted", zero_division=0
                ),
                "samples": len(labels),
            }
        )
        grouped = combined.setdefault(
            model_name, {"labels": [], "predictions": []}
        )
        grouped["labels"].append(labels)
        grouped["predictions"].append(predictions)

    if not rows:
        raise FileNotFoundError(
            f"No complete classification prediction artifacts found in {task_root}"
        )

    n_classes = max(
        int(max(np.max(part) for part in values["labels"])) + 1
        for values in combined.values()
    )
    class_names = requested_names or [str(index) for index in range(n_classes)]
    if len(class_names) != n_classes:
        raise ValueError(
            f"Expected {n_classes} --class-name values, got {len(class_names)}"
        )

    for model_name, values in combined.items():
        _plot_confusion(
            np.concatenate(values["labels"]),
            np.concatenate(values["predictions"]),
            class_names,
            output_dir
            / f"confusion_matrix_{_safe_filename(model_name)}.png",
        )
    return pd.DataFrame(rows).sort_values(
        ["pretrain_epoch", "model", "trial"]
    )


def _plot_regression_scatter(
    truth: np.ndarray,
    prediction: np.ndarray,
    target_name: str,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(truth, prediction, alpha=0.25, s=8)
    lower = float(min(truth.min(), prediction.min()))
    upper = float(max(truth.max(), prediction.max()))
    axis.plot([lower, upper], [lower, upper], "r--", label="Perfect prediction")
    axis.set_xlabel(f"True {target_name}")
    axis.set_ylabel(f"Predicted {target_name}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_regression(
    task_root: Path,
    output_dir: Path,
    requested_names: list[str],
) -> pd.DataFrame:
    rows = []
    combined: dict[str, dict[str, list[np.ndarray]]] = {}
    for model_name, trial_dir in _trial_directories(task_root):
        prediction_path = trial_dir / "test_predictions.npy"
        label_path = trial_dir / "test_labels.npy"
        if not prediction_path.is_file() or not label_path.is_file():
            print(f"Skipping incomplete trial: {trial_dir}")
            continue

        predictions = np.load(prediction_path)
        labels = np.load(label_path)
        if predictions.shape != labels.shape or predictions.ndim < 2:
            raise ValueError(
                f"Shape mismatch in {trial_dir}: predictions "
                f"{predictions.shape}, labels {labels.shape}"
            )
        rows.append(
            {
                "model": model_name,
                "pretrain_epoch": _pretrain_epoch(model_name),
                "trial": _trial_number(trial_dir),
                "r2": r2_score(labels, predictions),
                "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
                "mae": mean_absolute_error(labels, predictions),
                "points": int(np.prod(labels.shape[:-1])),
            }
        )
        grouped = combined.setdefault(
            model_name, {"labels": [], "predictions": []}
        )
        grouped["labels"].append(labels.reshape(-1, labels.shape[-1]))
        grouped["predictions"].append(
            predictions.reshape(-1, predictions.shape[-1])
        )

    if not rows:
        raise FileNotFoundError(
            f"No complete regression prediction artifacts found in {task_root}"
        )

    n_targets = next(iter(combined.values()))["labels"][0].shape[-1]
    target_names = requested_names or [
        f"target_{index}" for index in range(n_targets)
    ]
    if len(target_names) != n_targets:
        raise ValueError(
            f"Expected {n_targets} --target-name values, got {len(target_names)}"
        )
    for model_name, values in combined.items():
        truth = np.concatenate(values["labels"])
        prediction = np.concatenate(values["predictions"])
        for index, target_name in enumerate(target_names):
            _plot_regression_scatter(
                truth[:, index],
                prediction[:, index],
                target_name,
                output_dir
                / (
                    f"scatter_{_safe_filename(model_name)}_"
                    f"{_safe_filename(target_name)}.png"
                ),
            )
    return pd.DataFrame(rows).sort_values(
        ["pretrain_epoch", "model", "trial"]
    )


def main() -> None:
    args = build_parser().parse_args()
    task_root = args.results_dir / args.task / args.prefix
    output_dir = args.output_dir or (
        args.results_dir
        / "plots"
        / args.task
        / args.prefix
        / "artifact_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "classification":
        frame = analyze_classification(
            task_root, output_dir, args.class_name
        )
    else:
        frame = analyze_regression(
            task_root, output_dir, args.target_name
        )
    output = output_dir / "recomputed_trial_metrics.csv"
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved recomputed metrics and plots to {output_dir}")


if __name__ == "__main__":
    main()
