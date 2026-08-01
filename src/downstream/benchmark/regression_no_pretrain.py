"""Random-initialization pointwise regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.downstream.benchmark.plotting import plot_learning_curve
from src.downstream.benchmark.regression import (
    SparsePointRegressor,
    evaluate_predictions,
    fit_regression,
    make_regression_loader,
    predict_regression,
    unscale_labels,
)
from src.downstream.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    configure_torchsparse,
    encoder_kwargs,
    load_checkpoint_config,
    load_pointwise_splits,
    set_seed,
)
from src.models.sparse_simclr import sparse_simclr_21d


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train pointwise regression from random initialization."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--label-suffix", default="regression_labels")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoints/run_config.json"),
        help="Pretraining config used to match architecture and voxelization",
    )
    parser.add_argument("--scaler-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-every", type=int, default=0)
    return parser


def _scaled_metrics(labels, predictions):
    return {
        "r2": r2_score(labels, predictions),
        "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
        "mae": mean_absolute_error(labels, predictions),
    }


def main():
    args = build_parser().parse_args()
    config_anchor = args.config.parent / "unused.pt"
    config = load_checkpoint_config(config_anchor, args.config)
    ratio = configure_torchsparse(config)
    voxel_size = float(config["voxel_size"])
    kwargs = encoder_kwargs(config)
    splits = load_pointwise_splits(
        args.data_dir, args.prefix, args.label_suffix
    )
    num_labels = int(splits["train"][1].shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = (
        args.output_dir / "regression" / args.prefix / "random_init"
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
            split: make_regression_loader(
                *splits[split],
                batch_size=args.batch_size,
                voxel_size=voxel_size,
                shuffle=split == "train",
                drop_last=split == "train",
                num_workers=args.num_workers,
            )
            for split in ("train", "val", "test")
        }
        encoder = sparse_simclr_21d(**kwargs).to(device)
        model = SparsePointRegressor(
            encoder, num_labels=num_labels
        ).to(device)
        best_path = trial_dir / "best_model.pt"
        history = fit_regression(
            model,
            loaders["train"],
            loaders["val"],
            device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            best_model_path=best_path,
            scheduler_monitor="val_r2",
            early_stopping_patience=args.patience,
            checkpoint_dir=trial_dir / "weights",
            save_every=args.save_every,
        )
        (trial_dir / "history.json").write_text(
            json.dumps(history.history, indent=2)
        )
        plot_learning_curve(
            history.history, trial_dir / "learning_curve_loss.png"
        )

        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state"])
        result = predict_regression(model, loaders["test"], device)
        metrics = _scaled_metrics(result["labels"], result["outputs"])
        (trial_dir / "scaled_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        np.save(trial_dir / "test_predictions.npy", result["outputs"])
        np.save(trial_dir / "test_labels.npy", result["labels"])
        np.save(trial_dir / "event_lengths.npy", result["event_lengths"])
        summary_rows.append(
            {"model": "random_init", "trial": trial, **metrics}
        )

        if args.scaler_dir is not None:
            true_physical = unscale_labels(
                result["labels"], args.scaler_dir
            )
            predicted_physical = unscale_labels(
                result["outputs"], args.scaler_dir
            )
            evaluate_predictions(
                true_physical,
                predicted_physical,
                trial_dir,
                trial,
                experiment_dir / "physical_summary.csv",
                "random_init",
            )

    pd.DataFrame(summary_rows).to_csv(
        experiment_dir / "scaled_summary.csv", index=False
    )
    print(f"Saved results to {experiment_dir}")


if __name__ == "__main__":
    main()
