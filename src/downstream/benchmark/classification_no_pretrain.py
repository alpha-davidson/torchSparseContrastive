"""Random-initialization downstream classification baseline.

This uses the same SparseResNet21D architecture, sparse voxelization, global
pooling, and point-order invariance as the pretrained experiment. The only
difference is that the backbone weights start randomly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from src.downstream.benchmark.classification import (
    SparseEventClassifier,
    SparsePointClassifier,
    fit,
    make_loader,
    predict,
    resolve_proton_swap,
)
from src.downstream.benchmark.plotting import (
    plot_confusion_matrix,
    plot_learning_curve,
)
from src.downstream.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    configure_torchsparse,
    encoder_kwargs,
    load_checkpoint_config,
    load_classification_splits,
    load_pointwise_splits,
    set_seed,
)
from src.models.sparse_simclr import sparse_simclr_21d


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the downstream classifier from random initialization."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--prefix", default="O16_UNSAMPLED")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoints/run_config.json"),
        help="Pretraining config used only to match architecture/voxelization",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--pointwise", action="store_true")
    parser.add_argument(
        "--point-label-suffix",
        default="point_labels",
        help="Pointwise label filename suffix when --pointwise is used",
    )
    return parser


def _labels_for_metrics(result, pointwise):
    predictions = result["outputs"].argmax(axis=-1)
    labels = result["labels"]
    if pointwise:
        labels = resolve_proton_swap(
            labels, predictions, result["event_lengths"]
        )
    return labels, predictions


def main():
    args = build_parser().parse_args()
    if args.trials < 1 or args.epochs < 1:
        raise ValueError("--trials and --epochs must be positive")

    # The helper expects a checkpoint only to locate its adjacent config.
    config_anchor = args.config.parent / "unused.pt"
    config = load_checkpoint_config(config_anchor, args.config)
    ratio = configure_torchsparse(config)
    voxel_size = float(config["voxel_size"])
    kwargs = encoder_kwargs(config)
    if args.pointwise:
        splits = load_pointwise_splits(
            args.data_dir, args.prefix, args.point_label_suffix
        )
    else:
        splits = load_classification_splits(args.data_dir, args.prefix)

    all_labels = [
        np.asarray(splits[name][1]).reshape(-1)
        for name in ("train", "val", "test")
    ]
    num_classes = int(max(labels.max() for labels in all_labels)) + 1
    class_names = [str(index) for index in range(num_classes)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = (
        args.output_dir
        / "classification"
        / args.prefix
        / "random_init"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    print(f"Device: {device}")
    print(f"Voxel size: {voxel_size:g}")
    print(f"TorchSparse hash_rsv_ratio: {ratio:g}")
    print("Using complete canonical splits; no event subset sampling.")

    for trial in range(1, args.trials + 1):
        set_seed(args.seed + trial)
        trial_dir = experiment_dir / f"trial_{trial}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        loaders = {
            split: make_loader(
                *splits[split],
                batch_size=args.batch_size,
                voxel_size=voxel_size,
                shuffle=split == "train",
                drop_last=split == "train",
                is_pointwise=args.pointwise,
                num_workers=args.num_workers,
            )
            for split in ("train", "val", "test")
        }
        encoder = sparse_simclr_21d(**kwargs).to(device)
        model_cls = SparsePointClassifier if args.pointwise else SparseEventClassifier
        model = model_cls(encoder, num_classes).to(device)
        best_path = trial_dir / "best_model.pt"
        history = fit(
            model,
            loaders["train"],
            loaders["val"],
            device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            best_model_path=best_path,
            is_pointwise=args.pointwise,
            checkpoint_dir=trial_dir / "weights",
            save_every=args.save_every,
        )
        plot_learning_curve(
            history.history, trial_dir / "learning_curve_loss.png"
        )
        (trial_dir / "history.json").write_text(
            json.dumps(history.history, indent=2)
        )

        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state"])
        result = predict(
            model,
            loaders["test"],
            device,
            is_pointwise=args.pointwise,
        )
        y_true, y_pred = _labels_for_metrics(result, args.pointwise)
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        (trial_dir / "classification_report.json").write_text(
            json.dumps(report, indent=2)
        )
        (trial_dir / "classification_report.txt").write_text(
            classification_report(
                y_true,
                y_pred,
                labels=list(range(num_classes)),
                target_names=class_names,
                zero_division=0,
            )
        )
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            trial_dir / "confusion_matrix.png",
        )
        np.save(trial_dir / "test_probabilities.npy", result["outputs"])
        np.save(trial_dir / "test_labels.npy", y_true)
        summary_rows.append(
            {
                "model": "random_init",
                "trial": trial,
                "accuracy": report["accuracy"],
                "f1_macro": report["macro avg"]["f1-score"],
                "f1_weighted": report["weighted avg"]["f1-score"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        experiment_dir / "summary.csv", index=False
    )
    print(f"Saved results to {experiment_dir}")


if __name__ == "__main__":
    main()
