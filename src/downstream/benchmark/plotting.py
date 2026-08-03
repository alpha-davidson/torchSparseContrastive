"""Plotting helpers for the TorchSparse downstream evaluations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 11


def plot_learning_curve(history, filename):
    """Plot training and validation loss from a history mapping."""
    history = history.history if hasattr(history, "history") else history
    if "loss" not in history or "val_loss" not in history:
        raise KeyError("history must contain loss and val_loss")
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 6), dpi=100)
    axis.plot(history["loss"], "o-", label="Training loss")
    axis.plot(history["val_loss"], "o:", color="r", label="Validation loss")
    axis.set_title("Learning curve")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend(loc="best")
    positions = list(range(0, len(history["loss"]), 10))
    axis.set_xticks(positions, [position + 1 for position in positions])
    if all(
        value > 0
        for key in ("loss", "val_loss")
        for value in history[key]
    ):
        axis.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes,
    filename,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    """Save an integer confusion matrix with explicit class labels."""
    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(classes)))
    )
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 6), dpi=100)
    image = axis.imshow(matrix, interpolation="nearest", cmap=cmap)
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
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
    fig.savefig(output)
    plt.close(fig)
    return matrix


TRACK_COLORS = np.array(
    ["#E69F00", "#009E73", "#CC79A7", "#F0E442"]
)


def plot_pointwise_events(
    events,
    true_labels,
    predicted_labels,
    plots_folder,
    max_events=5,
):
    """Plot deterministic, already-aligned variable-length pointwise events.

    ``events``, ``true_labels``, and ``predicted_labels`` must be equally sized
    sequences. Every event is an ``(N_i, >=3)`` array, and its two label arrays
    must both have length ``N_i``. No event or point sampling is performed; the
    first ``max_events`` aligned events are visualized.
    """
    if not (len(events) == len(true_labels) == len(predicted_labels)):
        raise ValueError("events, true labels, and predictions must align")
    output_dir = Path(plots_folder) / "event_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for event_index in range(min(int(max_events), len(events))):
        event = np.asarray(events[event_index])
        truth = np.asarray(true_labels[event_index], dtype=int)
        prediction = np.asarray(predicted_labels[event_index], dtype=int)
        if event.ndim != 2 or event.shape[1] < 3:
            raise ValueError("each event must have shape (points, >=3)")
        if len(event) != len(truth) or len(event) != len(prediction):
            raise ValueError(
                f"event {event_index} has unaligned points and labels"
            )

        correct = truth == prediction
        fig = plt.figure(figsize=(15, 5))
        panels = (
            ("True labels", TRACK_COLORS[truth]),
            ("Predicted labels", TRACK_COLORS[prediction]),
            (
                f"Correct: {correct.mean():.2%}",
                np.where(correct[:, None], "#0072B2", "#D55E00").ravel(),
            ),
        )
        for panel_index, (title, colors) in enumerate(panels, start=1):
            axis = fig.add_subplot(1, 3, panel_index, projection="3d")
            axis.scatter(
                event[:, 0],
                event[:, 2],
                event[:, 1],
                c=colors,
                s=3,
                edgecolors="none",
            )
            axis.set_xlabel("x")
            axis.set_ylabel("z")
            axis.set_zlabel("y")
            axis.set_title(title)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"event_{event_index:04d}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
