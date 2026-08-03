"""Dynamic downstream classification evaluation of contrastive checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from src.downstream.benchmark.classification import (
    SparseEventClassifier,
    SparsePointClassifier,
    fit,
    load_sparse_encoder,
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
    checkpoint_name,
    configure_torchsparse,
    discover_checkpoints,
    encoder_kwargs,
    load_checkpoint_config,
    load_classification_splits,
    load_pointwise_splits,
    set_seed,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Freeze-then-fine-tune classification evaluation for one or more "
            "SparseSimCLR checkpoints."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--prefix", default="O16_UNSAMPLED")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-dir", action="append", default=[])
    parser.add_argument(
        "--checkpoint-pattern",
        action="append",
        default=[],
        help="Glob within each checkpoint directory; default: epoch_*.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override run_config.json for every checkpoint",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--pointwise", action="store_true")
    parser.add_argument("--point-label-suffix", default="point_labels")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate splits/configs/checkpoints and load each backbone only",
    )
    return parser


def _merge_histories(first, second):
    keys = set(first) | set(second)
    return {
        key: list(first.get(key, [])) + list(second.get(key, []))
        for key in keys
    }


def _best_validation_f1(state):
    values = state.get("history", {}).get("val_f1_weighted", [])
    return max(values) if values else float("-inf")


def _labels_for_metrics(result, pointwise):
    predictions = result["outputs"].argmax(axis=-1)
    labels = result["labels"]
    if pointwise:
        labels = resolve_proton_swap(
            labels, predictions, result["event_lengths"]
        )
    return labels, predictions


def _plot_accuracy(history, output):
    fig, axis = plt.subplots(figsize=(11, 6), dpi=100)
    axis.plot(history["accuracy"], "o-", label="Training accuracy")
    axis.plot(history["val_accuracy"], "o:", label="Validation accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    if not 0 < args.freeze_epochs < args.epochs:
        raise ValueError("--freeze-epochs must lie between 1 and --epochs - 1")
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    checkpoints = discover_checkpoints(
        args.checkpoint,
        args.checkpoint_dir,
        args.checkpoint_pattern or ["epoch_*.pt"],
    )
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
    num_classes = int(max(values.max() for values in all_labels)) + 1
    class_names = [str(index) for index in range(num_classes)]
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
        for trial in range(1, args.trials + 1):
            set_seed(args.seed + checkpoint_index * 10_000 + trial)
            trial_dir = (
                args.output_dir
                / "classification"
                / args.prefix
                / checkpoint_id
                / f"trial_{trial}"
            )
            trial_dir.mkdir(parents=True, exist_ok=True)
            encoder = load_sparse_encoder(
                checkpoint, device, **encoder_kwargs(config)
            )
            model_cls = (
                SparsePointClassifier if args.pointwise else SparseEventClassifier
            )
            model = model_cls(encoder, num_classes).to(device)
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(False)

            phase1_best = trial_dir / "phase1_best.pt"
            phase1 = fit(
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
                is_pointwise=args.pointwise,
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
            phase2 = fit(
                model,
                loaders["train"],
                loaders["val"],
                device,
                epochs=args.epochs - args.freeze_epochs,
                learning_rate=args.finetune_lr,
                best_model_path=phase2_best,
                is_pointwise=args.pointwise,
                checkpoint_dir=trial_dir / "weights",
                checkpoint_prefix="phase2",
                save_every=args.save_every,
            )
            phase2_state = torch.load(
                phase2_best, map_location=device, weights_only=False
            )
            best_state = max(
                (phase1_state, phase2_state), key=_best_validation_f1
            )
            best_path = trial_dir / "best_model.pt"
            torch.save(best_state, best_path)
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
            _plot_accuracy(
                history, trial_dir / "learning_curve_accuracy.png"
            )

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
            np.save(
                trial_dir / "test_probabilities.npy", result["outputs"]
            )
            np.save(trial_dir / "test_labels.npy", y_true)
            summary_rows.append(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_id": checkpoint_id,
                    "pretrain_epoch": int(
                        torch.load(
                            checkpoint,
                            map_location="cpu",
                            weights_only=False,
                        ).get("epoch", -1)
                    ),
                    "trial": trial,
                    "voxel_size": voxel_size,
                    "accuracy": report["accuracy"],
                    "f1_macro": report["macro avg"]["f1-score"],
                    "f1_weighted": report["weighted avg"]["f1-score"],
                }
            )
            print(
                f"  trial {trial}: accuracy={report['accuracy']:.4f}, "
                f"macro_f1={report['macro avg']['f1-score']:.4f}, "
                f"weighted_f1={report['weighted avg']['f1-score']:.4f}"
            )

    summary_dir = (
        args.output_dir / "classification" / args.prefix
    )
    frame = pd.DataFrame(summary_rows)
    frame.to_csv(summary_dir / "checkpoint_summary.csv", index=False)
    aggregate = (
        frame.groupby(["checkpoint", "checkpoint_id", "pretrain_epoch"])
        [["accuracy", "f1_macro", "f1_weighted"]]
        .agg(["mean", "std"])
    )
    aggregate.to_csv(summary_dir / "checkpoint_summary_aggregate.csv")
    print(f"\nSaved dynamic evaluation to {summary_dir}")


if __name__ == "__main__":
    main()
