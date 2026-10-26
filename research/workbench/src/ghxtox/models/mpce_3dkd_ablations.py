"""Explicit module-ablation variants for the frozen MPCE-3DKD architecture.

The frozen candidate itself is MPCE-Net B + outer-safe 3D teacher KD and is
not re-implemented here.  These classes remove exactly one architectural
module while keeping the rest of the design and capacity identical:

- no_cross: the four low-rank cross interaction values are zeroed in the
  global branch, and the prototype latent consumes the concatenated views
  rather than the interactions;
- no_proto: the prototype branch and its auxiliary loss are absent, so the
  model predicts from the global branch only.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ghxtox.train_mpce_net import (
    LATENT_DIM,
    PROTOTYPES_PER_CLASS,
    VIEW_DIM,
)


class MPCE3KDAblation(nn.Module):
    """MPCE-3DKD backbone with one module removed."""

    def __init__(self, embedding_dim: int = 1024, phys_dim: int = 9, variant: str = "no_cross") -> None:
        super().__init__()
        if variant not in {"no_cross", "no_proto"}:
            raise ValueError("variant must be 'no_cross' or 'no_proto'")
        self.variant = variant
        self.proj_mean = nn.Sequential(nn.Linear(embedding_dim, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_max = nn.Sequential(nn.Linear(embedding_dim, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_std = nn.Sequential(nn.Linear(embedding_dim, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_phys = nn.Sequential(nn.Linear(phys_dim, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        global_input = VIEW_DIM * 8
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 1),
        )
        self.latent = nn.Sequential(
            nn.Linear(VIEW_DIM * 4, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
            nn.GELU(),
        )
        if variant == "no_proto":
            self.prototypes = None
            self.alpha = None
        else:
            self.prototypes = nn.Parameter(torch.randn(2 * PROTOTYPES_PER_CLASS, LATENT_DIM) * 0.02)
            self.alpha = nn.Parameter(torch.zeros(1))

    def _views(self, tri: torch.Tensor, phys: torch.Tensor):
        mean_view = tri[:, :1024]
        max_view = tri[:, 1024:2048]
        std_view = tri[:, 2048:]
        h_mean = self.proj_mean(mean_view)
        h_max = self.proj_max(max_view)
        h_std = self.proj_std(std_view)
        h_phys = self.proj_phys(phys)
        return h_mean, h_max, h_std, h_phys

    def forward(self, tri: torch.Tensor, phys: torch.Tensor):
        h_mean, h_max, h_std, h_phys = self._views(tri, phys)
        mm = h_mean * h_max
        ms = h_mean * h_std
        mx = h_max * h_std
        sp = h_mean * h_phys
        interactions = torch.cat([mm, ms, mx, sp], dim=-1)
        if self.variant == "no_cross":
            interaction_input = torch.zeros_like(interactions)
            latent_input = torch.cat([h_mean, h_max, h_std, h_phys], dim=-1)
        else:
            interaction_input = interactions
            latent_input = interactions
        vector = torch.cat([h_mean, h_max, h_std, h_phys, interaction_input], dim=-1)
        z_global = self.global_mlp(vector).squeeze(-1)
        if self.variant == "no_proto":
            return z_global
        latent = self.latent(latent_input)
        latent = F.normalize(latent, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        cosine = latent @ prototypes.t()
        toxic_scores = cosine[:, :PROTOTYPES_PER_CLASS]
        nontoxic_scores = cosine[:, PROTOTYPES_PER_CLASS:]
        affinity_toxic = torch.logsumexp(toxic_scores, dim=-1)
        affinity_nontoxic = torch.logsumexp(nontoxic_scores, dim=-1)
        z_proto = affinity_toxic - affinity_nontoxic
        z_final = z_global + torch.sigmoid(self.alpha) * z_proto
        return z_final, z_global, z_proto, cosine
