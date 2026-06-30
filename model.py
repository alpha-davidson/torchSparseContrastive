# Edited: J. Gelina 06/22/26

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsparse import SparseTensor
from torchsparse.backbones.resnet import SparseResNet, SparseResNet21D



def _sparse_global_avg_pool(x: SparseTensor) -> torch.Tensor:
    """
    Global average pool over all occupied voxels.

    TorchSparse stores features in x.feats  (N_total, C)
    and batch indices in x.coords[:, 0]     (N_total,).

    Returns a dense tensor of shape (B, C).
    """
    feats  = x.feats                   # (N, C)
    coords = x.coords                  # (N, 4) — (batch, x, y, z)
    batch  = coords[:, 3].long()  # torchsparse v2.0: (x,y,z,batch)
    B      = int(batch.max().item()) + 1
    C      = feats.shape[1]

    out   = torch.zeros(B, C, device=feats.device, dtype=feats.dtype)
    count = torch.zeros(B,    device=feats.device, dtype=feats.dtype)

    out.scatter_add_(0, batch.unsqueeze(1).expand_as(feats), feats)
    count.scatter_add_(0, batch, torch.ones(len(batch), device=feats.device))
    count = count.clamp(min=1).unsqueeze(1)

    return out / count                 # (B, C)


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """2-layer MLP: Linear → BN → ReLU → Linear (→ optional BN)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 128,
        use_final_bn: bool = False,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=not use_final_bn),
            nn.BatchNorm1d(out_dim) if use_final_bn else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """Normalised temperature-scaled cross-entropy loss (SimCLR)."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1, z2 : (B, D) — L2-normalised projection vectors."""
        B  = z1.size(0)
        z  = torch.cat([z1, z2], dim=0)             # (2B, D)
        z  = F.normalize(z, dim=1)

        sim  = torch.mm(z, z.T) / self.temperature  # (2B, 2B)
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim  = sim.masked_fill(mask, float("-inf"))

        labels = torch.cat(
            [torch.arange(B, 2 * B), torch.arange(0, B)]
        ).to(z.device)

        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# SparseSimCLR
# ---------------------------------------------------------------------------

# The last stage of SparseResNet21D always outputs 128 channels.
_RESNET21D_OUT_CHANNELS = 128


class SparseSimCLR(nn.Module):
    """
    SimCLR wrapper around a local SparseResNet backbone.

    Parameters
    ----------
    backbone : SparseResNet
        Pre-constructed backbone (e.g. SparseResNet21D(in_channels=1)).
    backbone_out_channels : int
        Channel width of the backbone's last stage. Defaults to 128
        (correct for SparseResNet21D). Override if using a custom backbone.
    proj_hidden_dim : int
        Hidden dimension of the projection MLP.
    proj_out_dim : int
        Output/embedding dimension for NT-Xent.
    temperature : float
        NT-Xent temperature.
    use_final_bn : bool
        Append BN after the final projection linear (SimCLR v2 style).
    """

    def __init__(
        self,
        backbone: Optional[SparseResNet] = None,
        backbone_out_channels: int = _RESNET21D_OUT_CHANNELS,
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 128,
        temperature: float = 0.1,
        use_final_bn: bool = False,
    ) -> None:
        super().__init__()

        self.backbone  = backbone if backbone is not None else SparseResNet21D()
        self.projector = ProjectionHead(
            in_dim=backbone_out_channels,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
            use_final_bn=use_final_bn,
        )
        self.criterion = NTXentLoss(temperature=temperature)

    def encode(self, x: SparseTensor) -> torch.Tensor:
        """Backbone representation before projection. Shape: (B, backbone_out_channels)."""
        feature_maps = self.backbone(x)          # list of SparseTensors
        return _sparse_global_avg_pool(feature_maps[-1])

    def project(self, x: SparseTensor) -> torch.Tensor:
        """L2-normalised projected representation. Shape: (B, proj_out_dim)."""
        return F.normalize(self.projector(self.encode(x)), dim=1)

    def forward(
        self,
        view1: SparseTensor,
        view2: SparseTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        loss : scalar NT-Xent loss
        z1, z2 : (B, proj_out_dim) normalised projections
        """
        z1   = self.project(view1)
        z2   = self.project(view2)
        loss = self.criterion(z1, z2)
        return loss, z1, z2


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def sparse_simclr_21d(
    in_channels: int = 1,
    proj_out_dim: int = 128,
    proj_hidden_dim: int = 2048,
    temperature: float = 0.1,
    use_final_bn: bool = False,
) -> SparseSimCLR:
    """SparseSimCLR with the local SparseResNet21D backbone."""
    backbone = SparseResNet21D(in_channels=in_channels)
    return SparseSimCLR(
        backbone=backbone,
        backbone_out_channels=_RESNET21D_OUT_CHANNELS,
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
        temperature=temperature,
        use_final_bn=use_final_bn,
    )