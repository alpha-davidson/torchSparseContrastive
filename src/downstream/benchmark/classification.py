"""Shared TorchSparse utilities for downstream classification.

The loaders consume the normalized, padded arrays produced by
``src.data.downstream_data_loader.O16_downstream_no_resample`` and always
require the companion event
lengths. Padding is therefore never voxelized. Both event-wise and point-wise
classification are supported.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset


def _validate_arrays(features, labels, lengths, pointwise):
    features = np.asarray(features)
    labels = np.asarray(labels)
    lengths = np.asarray(lengths, dtype=np.int64)
    if features.ndim != 3 or features.shape[2] < 3:
        raise ValueError(
            "features must have shape (events, padded_points, >=3); "
            f"got {features.shape}"
        )
    if lengths.shape != (len(features),):
        raise ValueError(
            f"lengths must have shape ({len(features)},); got {lengths.shape}"
        )
    if np.any(lengths <= 0) or np.any(lengths > features.shape[1]):
        raise ValueError(
            f"event lengths must lie in [1, {features.shape[1]}]"
        )
    expected = features.shape[:2] if pointwise else (len(features),)
    if pointwise:
        if labels.ndim not in (2, 3):
            raise ValueError(
                "point-wise labels must have shape (events, points) or "
                f"(events, points, 1); got {labels.shape}"
            )
        if labels.ndim == 3:
            if labels.shape[2] != 1:
                raise ValueError("point-wise classification labels need one class id")
            labels = labels[..., 0]
        if labels.shape != expected:
            raise ValueError(f"point-wise labels must have shape {expected}")
    else:
        labels = labels.squeeze()
        if labels.shape != expected:
            raise ValueError(f"event labels must have shape {expected}")
    return features, labels, lengths


class NpySparseClassificationDataset(Dataset):
    """Padded NumPy events converted to sparse tensors using true lengths."""

    def __init__(self, features, labels, lengths, voxel_size=1 / 256):
        from torchsparse import SparseTensor
        from torchsparse.utils.quantize import sparse_quantize

        self.features, self.labels, self.lengths = _validate_arrays(
            features, labels, lengths, pointwise=False
        )
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

        # Match O16Dataset._to_sparse and latent extraction.
        xyz = xyz - xyz.min(axis=0, keepdims=True)
        coords, selected = self._sparse_quantize(
            xyz, voxel_size=self.voxel_size, return_index=True
        )
        sparse = self._SparseTensor(
            feats=torch.as_tensor(feats[selected], dtype=torch.float32),
            coords=torch.as_tensor(coords, dtype=torch.int32),
        )
        return {
            "x": sparse,
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


class NpySparsePointwiseDataset(Dataset):
    """Padded events with one integer class label per real input point."""

    def __init__(self, features, labels, lengths, voxel_size=1 / 256):
        from torchsparse import SparseTensor
        from torchsparse.utils.quantize import sparse_quantize

        self.features, self.labels, self.lengths = _validate_arrays(
            features, labels, lengths, pointwise=True
        )
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
                self.labels[index, :n][selected], dtype=torch.long
            ),
            "selected_idx": torch.as_tensor(selected, dtype=torch.long),
            "event_index": int(index),
        }


def sparse_classification_collate(batch):
    from torchsparse.utils.collate import sparse_collate

    return {
        "x": sparse_collate([item["x"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


def sparse_pointwise_collate(batch):
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


def make_loader(
    features,
    labels,
    lengths,
    batch_size,
    voxel_size,
    shuffle,
    drop_last=False,
    is_pointwise=False,
    num_workers=0,
):
    dataset_cls = (
        NpySparsePointwiseDataset
        if is_pointwise
        else NpySparseClassificationDataset
    )
    dataset = dataset_cls(features, labels, lengths, voxel_size)
    collate_fn = (
        sparse_pointwise_collate
        if is_pointwise
        else sparse_classification_collate
    )
    return DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=shuffle,
        drop_last=drop_last and len(dataset) >= int(batch_size),
        num_workers=int(num_workers),
        collate_fn=collate_fn,
        pin_memory=int(num_workers) > 0 and torch.cuda.is_available(),
    )


class SparseEventClassifier(nn.Module):
    """SparseSimCLR encoder representation followed by an event classifier."""

    def __init__(self, encoder, num_classes, backbone_dim=128):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @property
    def backbone(self):
        return self.encoder.backbone

    def forward(self, sparse_input):
        return self.classifier(self.encoder.encode(sparse_input))


def _knn_propagate(fine, coarse, coarse_feats, k=3):
    """Interpolate coarse features onto fine coordinates.

    This repository's sparse collation order is ``(batch, x, y, z)``.
    """
    batches_fine = fine[:, 0].long()
    batches_coarse = coarse[:, 0].long()
    interpolated = torch.zeros(
        fine.shape[0],
        coarse_feats.shape[1],
        device=coarse_feats.device,
        dtype=coarse_feats.dtype,
    )
    for batch_id in batches_fine.unique():
        fine_mask = batches_fine == batch_id
        coarse_mask = batches_coarse == batch_id
        fine_xyz = fine[fine_mask, 1:4]
        coarse_xyz = coarse[coarse_mask, 1:4]
        if coarse_xyz.shape[0] == 0:
            raise RuntimeError(f"backbone produced no voxels for batch {batch_id}")
        distances = torch.cdist(fine_xyz, coarse_xyz)
        k_eff = min(int(k), coarse_xyz.shape[0])
        knn_distances, knn_indices = distances.topk(
            k_eff, dim=1, largest=False
        )
        weights = 1.0 / (knn_distances + 1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True)
        gathered = coarse_feats[coarse_mask][knn_indices]
        interpolated[fine_mask] = (
            gathered * weights.unsqueeze(-1)
        ).sum(dim=1)
    return interpolated


class SparsePointClassifier(nn.Module):
    """Final sparse feature map propagated to input voxels and classified."""

    def __init__(self, encoder, num_classes, backbone_dim=128, k_propagate=3):
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
            nn.Linear(128, num_classes),
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


def proton_swap_invariant_ce(logits, labels, event_lengths):
    """Choose the lower event-level loss under the class-2/class-3 swap."""
    ce = F.cross_entropy(logits, labels, reduction="none")
    swapped = labels.clone()
    swapped[labels == 2] = 3
    swapped[labels == 3] = 2
    ce_swapped = F.cross_entropy(logits, swapped, reduction="none")

    losses = []
    offset = 0
    for length in event_lengths:
        n = int(length)
        losses.append(
            torch.minimum(
                ce[offset : offset + n].mean(),
                ce_swapped[offset : offset + n].mean(),
            )
        )
        offset += n
    return torch.stack(losses).mean()


def resolve_proton_swap(y_true, y_pred, event_lengths):
    """Resolve the class-2/class-3 ambiguity consistently per event."""
    resolved = np.asarray(y_true).copy()
    offset = 0
    for length in event_lengths:
        n = int(length)
        true_event = y_true[offset : offset + n]
        pred_event = y_pred[offset : offset + n]
        swapped = np.where(
            true_event == 2, 3, np.where(true_event == 3, 2, true_event)
        )
        if (swapped == pred_event).mean() > (true_event == pred_event).mean():
            resolved[offset : offset + n] = swapped
        offset += n
    return resolved


def load_sparse_encoder(
    checkpoint_path,
    device,
    in_channels=1,
    proj_hidden_dim=512,
    proj_out_dim=128,
    temperature=0.1,
    use_final_bn=False,
):
    """Build the repo model and load only its contrastively trained backbone."""
    from src.models.sparse_simclr import sparse_simclr_21d

    encoder = sparse_simclr_21d(
        in_channels=int(in_channels),
        proj_hidden_dim=int(proj_hidden_dim),
        proj_out_dim=int(proj_out_dim),
        temperature=float(temperature),
        use_final_bn=bool(use_final_bn),
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model_state", checkpoint)
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise RuntimeError(
            f"{checkpoint_path} has no model_state keys beginning with 'backbone.'"
        )
    incompatible = encoder.backbone.load_state_dict(backbone_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint backbone does not match SparseResNet21D. "
            f"Missing={incompatible.missing_keys}; "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return encoder


def run_epoch(model, loader, device, optimizer=None, is_pointwise=False):
    training = optimizer is not None
    model.train(training)
    if training and not any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    ):
        # Freezing weights must also freeze backbone BatchNorm statistics.
        model.backbone.eval()
    total_loss = 0.0
    total_loss_weight = 0
    total_examples = 0
    all_true, all_pred, all_lengths = [], [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            inputs = batch["x"].to(device)
            labels = batch["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if is_pointwise:
                event_lengths = batch["event_lengths"].to(device)
                loss = proton_swap_invariant_ce(logits, labels, event_lengths)
                all_lengths.append(batch["event_lengths"].numpy())
                loss_weight = len(event_lengths)
            else:
                loss = F.cross_entropy(logits, labels)
                loss_weight = labels.numel()
            if training:
                loss.backward()
                optimizer.step()

            count = labels.numel()
            total_loss += loss.item() * loss_weight
            total_loss_weight += loss_weight
            total_examples += count
            all_true.append(labels.detach().cpu().numpy())
            all_pred.append(logits.argmax(dim=-1).detach().cpu().numpy())

    if total_examples == 0:
        raise ValueError("data loader produced no batches")
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    if is_pointwise:
        lengths = np.concatenate(all_lengths)
        y_true = resolve_proton_swap(y_true, y_pred, lengths)

    return {
        "loss": total_loss / total_loss_weight,
        "accuracy": float((y_true == y_pred).mean()),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def predict(model, loader, device, probabilities=True, is_pointwise=False):
    model.eval()
    outputs, labels, lengths = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device))
            output = torch.softmax(logits, dim=-1) if probabilities else logits
            outputs.append(output.cpu().numpy())
            labels.append(batch["label"].numpy())
            if is_pointwise:
                lengths.append(batch["event_lengths"].numpy())
    result = {
        "outputs": np.concatenate(outputs),
        "labels": np.concatenate(labels),
    }
    if is_pointwise:
        result["event_lengths"] = np.concatenate(lengths)
    return result


def fit(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    learning_rate,
    best_model_path,
    optimizer_parameters=None,
    scheduler_patience=7,
    early_stopping_patience=None,
    is_pointwise=False,
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=int(scheduler_patience)
    )
    history = defaultdict(list)
    best_f1 = float("-inf")
    stale_epochs = 0
    best_model_path = Path(best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(epochs) + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer, is_pointwise
        )
        val_metrics = run_epoch(
            model, val_loader, device, is_pointwise=is_pointwise
        )
        scheduler.step(val_metrics["loss"])
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
        if val_metrics["f1_weighted"] > best_f1:
            best_f1 = val_metrics["f1_weighted"]
            stale_epochs = 0
            torch.save(state, best_model_path)
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:03d}/{epochs}: "
            f"loss={train_metrics['loss']:.4f} "
            f"accuracy={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_f1_weighted={val_metrics['f1_weighted']:.4f}"
        )
        if (
            early_stopping_patience is not None
            and stale_epochs >= int(early_stopping_patience)
        ):
            print(f"Early stopping at epoch {epoch}")
            break

    return SimpleNamespace(history=dict(history))
