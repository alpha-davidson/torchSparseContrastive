"""Dynamic pointwise-regression evaluation of contrastive checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.downstream.benchmark.classification import load_sparse_encoder
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
    checkpoint_name,
    configure_torchsparse,
    discover_checkpoints,
    encoder_kwargs,
    load_checkpoint_config,
    load_pointwise_splits,
    set_seed,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Freeze-then-fine-tune pointwise regression for one or more "
            "SparseSimCLR checkpoints."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--label-suffix", default="regression_labels")
    parser.add_argument("--scaler-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-dir", action="append", default=[])
    parser.add_argument("--checkpoint-pattern", action="append", default=[])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--freeze-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate splits/configs/checkpoints and load each backbone only",
    )
    return parser


def _merge_histories(first, second):
    return {
        key: list(first.get(key, [])) + list(second.get(key, []))
        for key in set(first) | set(second)
    }


def _best_validation_r2(state):
    values = state.get("history", {}).get("val_r2", [])
    return max(values) if values else float("-inf")


def _scaled_metrics(labels, predictions):
    return {
        "r2": r2_score(labels, predictions),
        "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
        "mae": mean_absolute_error(labels, predictions),
    }


def _plot_r2(history, output):
    fig, axis = plt.subplots(figsize=(11, 6), dpi=100)
    axis.plot(history["val_r2"], "o:", label="Validation R²")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("R²")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    if not 0 < args.freeze_epochs < args.epochs:
        raise ValueError("--freeze-epochs must lie between 1 and --epochs - 1")
    checkpoints = discover_checkpoints(
        args.checkpoint,
        args.checkpoint_dir,
        args.checkpoint_pattern or ["epoch_*.pt"],
    )
    splits = load_pointwise_splits(
        args.data_dir, args.prefix, args.label_suffix
    )
    num_labels = int(splits["train"][1].shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows = []
    print(f"Device: {device}")
    print(f"Checkpoints selected: {len(checkpoints)}")
    print("Using complete canonical splits; no event subset sampling.")
    if args.dry_run:
        print(
            "Split sizes: "
            + ", ".join(
                f"{name}={len(splits[name][0])}"
                for name in ("train", "val", "test")
            )
        )
        for checkpoint in checkpoints:
            config = load_checkpoint_config(checkpoint, args.config)
            configure_torchsparse(config)
            load_sparse_encoder(
                checkpoint, device, **encoder_kwargs(config)
            )
            print(
                f"Validated {checkpoint} with "
                f"voxel_size={float(config['voxel_size']):g}"
            )
        print("Dry run passed.")
        return

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        config = load_checkpoint_config(checkpoint, args.config)
        ratio = configure_torchsparse(config)
        voxel_size = float(config["voxel_size"])
        checkpoint_id = checkpoint_name(checkpoint)
        print(
            f"\nCheckpoint {checkpoint_index + 1}/{len(checkpoints)}: "
            f"{checkpoint}\n"
            f"  config={config['_config_path']}\n"
            f"  voxel_size={voxel_size:g}, in_channels={config['in_channels']}, "
            f"hash_rsv_ratio={ratio:g}"
        )
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

        for trial in range(1, args.trials + 1):
            set_seed(args.seed + checkpoint_index * 10_000 + trial)
            trial_dir = (
                args.output_dir
                / "regression"
                / args.prefix
                / checkpoint_id
                / f"trial_{trial}"
            )
            trial_dir.mkdir(parents=True, exist_ok=True)
            encoder = load_sparse_encoder(
                checkpoint, device, **encoder_kwargs(config)
            )
            model = SparsePointRegressor(
                encoder, num_labels=num_labels
            ).to(device)
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(False)

            phase1_best = trial_dir / "phase1_best.pt"
            phase1 = fit_regression(
                model,
                loaders["train"],
                loaders["val"],
                device,
                epochs=args.freeze_epochs,
                learning_rate=args.head_lr,
                best_model_path=phase1_best,
                optimizer_parameters=(
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                scheduler_monitor="val_r2",
                early_stopping_patience=args.patience,
                checkpoint_dir=trial_dir / "weights",
                checkpoint_prefix="phase1",
                save_every=args.save_every,
            )
            phase1_state = torch.load(
                phase1_best, map_location=device, weights_only=False
            )
            model.load_state_dict(phase1_state["model_state"])
            for parameter in model.parameters():
                parameter.requires_grad_(True)

            phase2_best = trial_dir / "phase2_best.pt"
            phase2 = fit_regression(
                model,
                loaders["train"],
                loaders["val"],
                device,
                epochs=args.epochs - args.freeze_epochs,
                learning_rate=args.finetune_lr,
                best_model_path=phase2_best,
                scheduler_monitor="val_r2",
                early_stopping_patience=args.patience,
                checkpoint_dir=trial_dir / "weights",
                checkpoint_prefix="phase2",
                save_every=args.save_every,
            )
            phase2_state = torch.load(
                phase2_best, map_location=device, weights_only=False
            )
            best_state = max(
                (phase1_state, phase2_state), key=_best_validation_r2
            )
            torch.save(best_state, trial_dir / "best_model.pt")
            model.load_state_dict(best_state["model_state"])

            history = _merge_histories(
                phase1.history, phase2.history
            )
            (trial_dir / "history.json").write_text(
                json.dumps(history, indent=2)
            )
            plot_learning_curve(
                history, trial_dir / "learning_curve_loss.png"
            )
            _plot_r2(history, trial_dir / "learning_curve_r2.png")

            result = predict_regression(model, loaders["test"], device)
            metrics = _scaled_metrics(
                result["labels"], result["outputs"]
            )
            (trial_dir / "scaled_metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )
            np.save(
                trial_dir / "test_predictions.npy", result["outputs"]
            )
            np.save(trial_dir / "test_labels.npy", result["labels"])
            np.save(
                trial_dir / "event_lengths.npy", result["event_lengths"]
            )
            pretrain_epoch = int(
                torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                ).get("epoch", -1)
            )
            summary_rows.append(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_id": checkpoint_id,
                    "pretrain_epoch": pretrain_epoch,
                    "trial": trial,
                    "voxel_size": voxel_size,
                    **metrics,
                }
            )
            if args.scaler_dir is not None:
                evaluate_predictions(
                    unscale_labels(result["labels"], args.scaler_dir),
                    unscale_labels(result["outputs"], args.scaler_dir),
                    trial_dir,
                    trial,
                    (
                        args.output_dir
                        / "regression"
                        / args.prefix
                        / "physical_checkpoint_summary.csv"
                    ),
                    checkpoint_id,
                )
            print(
                f"  trial {trial}: scaled R²={metrics['r2']:.4f}, "
                f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"
            )

    summary_dir = args.output_dir / "regression" / args.prefix
    frame = pd.DataFrame(summary_rows)
    frame.to_csv(summary_dir / "checkpoint_summary.csv", index=False)
    aggregate = (
        frame.groupby(["checkpoint", "checkpoint_id", "pretrain_epoch"])
        [["r2", "rmse", "mae"]]
        .agg(["mean", "std"])
    )
    aggregate.to_csv(summary_dir / "checkpoint_summary_aggregate.csv")
    print(f"\nSaved dynamic evaluation to {summary_dir}")


if __name__ == "__main__":
    main()
