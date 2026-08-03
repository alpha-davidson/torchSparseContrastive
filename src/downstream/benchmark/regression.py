"""Shared TorchSparse utilities for point-wise downstream regression."""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from src.downstream.benchmark.classification import _knn_propagate


class NpySparseRegressionDataset(Dataset):
    """Padded events with continuous labels for every real input point."""

    def __init__(self, features, labels, lengths, voxel_size=1 / 256):
        from torchsparse import SparseTensor
        from torchsparse.utils.quantize import sparse_quantize

        self.features = np.asarray(features)
        self.labels = np.asarray(labels)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        if self.features.ndim != 3 or self.features.shape[2] < 3:
            raise ValueError(
                "features must have shape (events, padded_points, >=3)"
            )
        if self.labels.ndim != 3 or self.labels.shape[:2] != self.features.shape[:2]:
            raise ValueError(
                "regression labels must have shape "
                "(events, padded_points, output_channels)"
            )
        if self.lengths.shape != (len(self.features),):
            raise ValueError(
                f"lengths must have shape ({len(self.features)},)"
            )
        if np.any(self.lengths <= 0) or np.any(
            self.lengths > self.features.shape[1]
        ):
            raise ValueError("event lengths are outside the padded array")
        self.voxel_size = float(voxel_size)
        self._SparseTensor = SparseTensor
        self._sparse_quantize = sparse_quantize

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        n = int(self.lengths[index])
        event = self.features[index, :n].astype(np.float32, copy=False)
        xyz = event[:, :3]
        feats = event[:, 3:]
        if feats.shape[1] == 0:
            feats = np.ones((n, 1), dtype=np.float32)

        xyz_shifted = xyz - xyz.min(axis=0, keepdims=True)
        coords, selected = self._sparse_quantize(
            xyz_shifted, voxel_size=self.voxel_size, return_index=True
        )
        sparse = self._SparseTensor(
            feats=torch.as_tensor(feats[selected], dtype=torch.float32),
            coords=torch.as_tensor(coords, dtype=torch.int32),
        )
        return {
            "x": sparse,
            "label": torch.as_tensor(
                self.labels[index, :n][selected], dtype=torch.float32
            ),
            "selected_idx": torch.as_tensor(selected, dtype=torch.long),
            "event_index": int(index),
        }


def sparse_regression_collate(batch):
    from torchsparse.utils.collate import sparse_collate

    return {
        "x": sparse_collate([item["x"] for item in batch]),
        "label": torch.cat([item["label"] for item in batch]),
        "event_lengths": torch.tensor(
            [item["label"].shape[0] for item in batch], dtype=torch.long
        ),
        "selected_idx": [item["selected_idx"] for item in batch],
        "event_index": torch.tensor(
            [item["event_index"] for item in batch], dtype=torch.long
        ),
    }


def make_regression_loader(
    features,
    labels,
    lengths,
    batch_size,
    voxel_size,
    shuffle,
    drop_last=False,
    num_workers=0,
):
    dataset = NpySparseRegressionDataset(
        features, labels, lengths, voxel_size
    )
    return DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=shuffle,
        drop_last=drop_last and len(dataset) >= int(batch_size),
        num_workers=int(num_workers),
        collate_fn=sparse_regression_collate,
        pin_memory=int(num_workers) > 0 and torch.cuda.is_available(),
    )


class SparsePointRegressor(nn.Module):
    """Final sparse feature map propagated to input voxels and regressed."""

    def __init__(self, encoder, num_labels=5, backbone_dim=128, k_propagate=3):
        super().__init__()
        self.backbone_net = encoder.backbone
        self.k = int(k_propagate)
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels),
        )

    @property
    def backbone(self):
        return self.backbone_net

    def forward(self, sparse_input):
        input_coords = sparse_input.coords.float()
        feature_maps = self.backbone_net(sparse_input)
        final_map = feature_maps[-1]
        propagated = _knn_propagate(
            input_coords,
            final_map.coords.float(),
            final_map.feats,
            self.k,
        )
        return self.head(propagated)


class SparseEventRegressor(nn.Module):
    """Pooled encoder representation followed by an event regression head."""

    def __init__(self, encoder, num_labels, backbone_dim=128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels),
        )

    @property
    def backbone(self):
        return self.encoder.backbone

    def forward(self, sparse_input):
        return self.head(self.encoder.encode(sparse_input))


_LABEL_WEIGHTS = torch.tensor([1.0, 1.0, 2.0, 2.0, 1.0])


def weighted_mse(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"prediction/label shape mismatch: {y_pred.shape} vs {y_true.shape}"
        )
    n_labels = y_pred.shape[-1]
    if n_labels > len(_LABEL_WEIGHTS):
        raise ValueError(
            f"weighted_mse supports at most {len(_LABEL_WEIGHTS)} outputs"
        )
    weights = _LABEL_WEIGHTS[:n_labels].to(y_pred.device)
    return (weights * (y_true - y_pred) ** 2).mean()


def load_scalers(directory):
    directory = Path(directory)
    with (directory / "qt_list.pkl").open("rb") as handle:
        quantile_transformers = pickle.load(handle)
    with (directory / "max_vals.pkl").open("rb") as handle:
        max_values = pickle.load(handle)
    return quantile_transformers, max_values


def unscale_labels(y_scaled, scaler_dir):
    """Convert five scaled channels to theta, phi, and brho.

    Both ``(points, channels)`` and ``(events, points, channels)`` inputs are
    accepted. Output retains the leading dimensions and has three channels.
    """
    values = np.asarray(y_scaled)
    if values.ndim not in (2, 3) or values.shape[-1] < 5:
        raise ValueError(
            "unscale_labels expects (..., 5) scaled regression channels"
        )
    quantile_transformers, max_values = load_scalers(scaler_dir)
    original_shape = values.shape
    flat = values.reshape(-1, original_shape[-1])
    unscaled = np.empty_like(flat, dtype=np.float32)
    for index in range(original_shape[-1]):
        column = flat[:, index : index + 1]
        if quantile_transformers[index] is not None:
            unscaled[:, index] = (
                quantile_transformers[index]
                .inverse_transform(column * max_values[index])
                .ravel()
            )
        else:
            unscaled[:, index] = column.ravel()

    cos_theta = np.clip(unscaled[:, 1], -1.0, 1.0)
    sin_phi = unscaled[:, 2]
    cos_phi = unscaled[:, 3]
    theta = np.arccos(cos_theta)
    phi = np.mod(np.arctan2(sin_phi, cos_phi), 2 * np.pi)
    brho = np.expm1(unscaled[:, 4])
    physical = np.stack([theta, phi, brho], axis=-1)
    return physical.reshape(*original_shape[:-1], 3)


def evaluate_predictions(
    y_true,
    y_pred,
    plots_folder,
    trial,
    summary_path,
    checkpoint_name,
):
    """Write per-target metrics and true-versus-predicted scatter plots."""
    plots_folder = Path(plots_folder)
    plots_folder.mkdir(parents=True, exist_ok=True)
    true_flat = np.asarray(y_true).reshape(-1, 3)
    pred_flat = np.asarray(y_pred).reshape(-1, 3)
    if true_flat.shape != pred_flat.shape:
        raise ValueError(
            f"physical prediction shape mismatch: {pred_flat.shape} vs "
            f"{true_flat.shape}"
        )

    report_lines = [
        f"Regression report — {checkpoint_name}, trial {trial}\n",
        "=" * 60 + "\n",
    ]
    rows = []
    for index, label in enumerate(("theta", "phi", "brho")):
        truth = true_flat[:, index]
        prediction = pred_flat[:, index]
        r2 = r2_score(truth, prediction)
        rmse = np.sqrt(mean_squared_error(truth, prediction))
        mae = mean_absolute_error(truth, prediction)
        report_lines.append(
            f"{label}: R²={r2:.6f} RMSE={rmse:.6f} MAE={mae:.6f}\n"
        )
        rows.append(
            {
                "checkpoint": checkpoint_name,
                "trial": trial,
                "label": label,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
            }
        )
        fig, axis = plt.subplots(figsize=(7, 6))
        axis.scatter(truth, prediction, alpha=0.3, s=10)
        limits = [
            min(truth.min(), prediction.min()),
            max(truth.max(), prediction.max()),
        ]
        axis.plot(limits, limits, "r--", label="Perfect fit")
        axis.set_xlabel(f"True {label}")
        axis.set_ylabel(f"Predicted {label}")
        axis.set_title(
            f"{label}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}"
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(plots_folder / f"scatter_{label}.png", dpi=150)
        plt.close(fig)

    (plots_folder / "regression_report.txt").write_text(
        "".join(report_lines)
    )
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        summary_path,
        mode="a",
        header=not summary_path.exists(),
        index=False,
    )
    print("".join(report_lines))
    return rows


def run_regression_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training and not any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    ):
        model.backbone.eval()
    total_loss = 0.0
    total_examples = 0
    true_values, predictions = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            inputs = batch["x"].to(device)
            labels = batch["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predicted = model(inputs)
            loss = weighted_mse(predicted, labels)
            if training:
                loss.backward()
                optimizer.step()
            count = labels.shape[0]
            total_loss += loss.item() * count
            total_examples += count
            true_values.append(labels.detach().cpu().numpy())
            predictions.append(predicted.detach().cpu().numpy())

    if total_examples == 0:
        raise ValueError("data loader produced no batches")
    truth = np.concatenate(true_values)
    predicted = np.concatenate(predictions)
    return {
        "loss": total_loss / total_examples,
        "mse": mean_squared_error(truth, predicted),
        "mae": mean_absolute_error(truth, predicted),
        "r2": r2_score(truth, predicted),
    }


def predict_regression(model, loader, device):
    model.eval()
    outputs, labels, lengths = [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs.append(model(batch["x"].to(device)).cpu().numpy())
            labels.append(batch["label"].numpy())
            lengths.append(batch["event_lengths"].numpy())
    return {
        "outputs": np.concatenate(outputs),
        "labels": np.concatenate(labels),
        "event_lengths": np.concatenate(lengths),
    }


def fit_regression(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    learning_rate,
    best_model_path,
    optimizer_parameters=None,
    scheduler_patience=7,
    scheduler_monitor="val_r2",
    early_stopping_patience=None,
    checkpoint_dir=None,
    checkpoint_prefix="checkpoint",
    save_every=0,
):
    parameters = (
        optimizer_parameters
        if optimizer_parameters is not None
        else model.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    maximize = scheduler_monitor == "val_r2"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        patience=int(scheduler_patience),
        factor=0.5,
    )
    best_metric = float("-inf") if maximize else float("inf")
    history = defaultdict(list)
    stale_epochs = 0
    best_model_path = Path(best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(epochs) + 1):
        train_metrics = run_regression_epoch(
            model, train_loader, device, optimizer
        )
        val_metrics = run_regression_epoch(model, val_loader, device)
        monitored = (
            val_metrics["r2"] if maximize else val_metrics["loss"]
        )
        scheduler.step(monitored)
        for key, value in train_metrics.items():
            history[key].append(value)
        for key, value in val_metrics.items():
            history[f"val_{key}"].append(value)

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": dict(history),
        }
        if (
            checkpoint_dir is not None
            and int(save_every) > 0
            and epoch % int(save_every) == 0
        ):
            torch.save(
                state,
                checkpoint_dir / f"{checkpoint_prefix}_{epoch:03d}.pt",
            )
        improved = (
            monitored > best_metric if maximize else monitored < best_metric
        )
        if improved:
            best_metric = monitored
            stale_epochs = 0
            torch.save(state, best_model_path)
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:03d}/{epochs}: "
            f"loss={train_metrics['loss']:.4f} "
            f"mse={train_metrics['mse']:.4f} "
            f"mae={train_metrics['mae']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_r2={val_metrics['r2']:.4f}"
        )
        if (
            early_stopping_patience is not None
            and stale_epochs >= int(early_stopping_patience)
        ):
            print(f"Early stopping at epoch {epoch}")
            break
    return SimpleNamespace(history=dict(history))
