"""Sequence-only two-stage student guided by training-only structure-quality proxies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StructureQualityProxyConfig:
    input_dim: int
    shared_dim: int = 192
    proxy_hidden_dim: int = 64
    decision_hidden_dim: int = 96
    dropout: float = 0.10
    aux_dim: int = 3


class StructureQualityProxyStudent(nn.Module):
    """Predict structure-quality proxies first, then make one toxicity decision.

    The predicted proxies are normalized training targets during fitting and are
    sequence-derived at inference.  No coordinates, structures, or teacher
    probabilities are accepted by ``forward``.
    """

    def __init__(self, config: StructureQualityProxyConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.shared_dim),
            nn.LayerNorm(config.shared_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.proxy_head = nn.Sequential(
            nn.Linear(config.shared_dim, config.proxy_hidden_dim),
            nn.GELU(),
            nn.Linear(config.proxy_hidden_dim, config.aux_dim),
        )
        self.decision_head = nn.Sequential(
            nn.Linear(config.shared_dim + config.aux_dim, config.decision_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decision_hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.config.input_dim:
            raise ValueError(f"Features must be [B, {self.config.input_dim}]")
        shared = self.encoder(features)
        proxy = self.proxy_head(shared)
        toxicity_logit = self.decision_head(torch.cat((shared, proxy), dim=-1)).squeeze(-1)
        return toxicity_logit, proxy
