"""Single-output coarse-to-refinement heads for the spectral candidate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ghxtox.models.spectral_physchem import SPECTRAL_FEATURE_NAMES
from ghxtox.models.tristat_head import TriStatMLPHead


@dataclass(frozen=True)
class SpectralRefineConfig:
    base_input_dim: int
    spectral_dim: int = len(SPECTRAL_FEATURE_NAMES)
    variant: str = "gated_residual"
    hidden_dim: int = 192
    spectral_hidden_dim: int = 64
    dropout: float = 0.10


class SpectralRefineModel(nn.Module):
    """A Stage-1 MLP logit plus an optional lightweight spectral correction."""

    def __init__(self, config: SpectralRefineConfig) -> None:
        super().__init__()
        if config.variant not in {"spectral_only", "concat", "gated_residual"}:
            raise ValueError("variant must be spectral_only, concat, or gated_residual")
        self.config = config
        if config.variant in {"spectral_only", "gated_residual"}:
            self.spectral = nn.Sequential(nn.LayerNorm(config.spectral_dim), nn.Linear(config.spectral_dim, config.spectral_hidden_dim), nn.GELU(), nn.Dropout(config.dropout))
            self.spectral_output = nn.Linear(config.spectral_hidden_dim, 1)
        if config.variant == "concat":
            self.concat = nn.Sequential(nn.LayerNorm(config.base_input_dim + config.spectral_dim), nn.Linear(config.base_input_dim + config.spectral_dim, config.hidden_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, 1))
        if config.variant == "gated_residual":
            self.base = TriStatMLPHead(config.base_input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout)
            self.gate = nn.Sequential(nn.LayerNorm(config.base_input_dim + config.spectral_dim), nn.Linear(config.base_input_dim + config.spectral_dim, config.spectral_hidden_dim), nn.GELU(), nn.Linear(config.spectral_hidden_dim, 1), nn.Sigmoid())
            nn.init.zeros_(self.spectral_output.weight)
            nn.init.zeros_(self.spectral_output.bias)

    def forward(self, base_features: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        if base_features.ndim != 2 or base_features.shape[-1] != self.config.base_input_dim:
            raise ValueError(f"Base features must be [B, {self.config.base_input_dim}]")
        if spectral_features.ndim != 2 or spectral_features.shape[-1] != self.config.spectral_dim:
            raise ValueError(f"Spectral features must be [B, {self.config.spectral_dim}]")
        if self.config.variant == "spectral_only":
            return self.spectral_output(self.spectral(spectral_features)).squeeze(-1)
        if self.config.variant == "concat":
            return self.concat(torch.cat((base_features, spectral_features), dim=-1)).squeeze(-1)
        base_logit = self.base(base_features)
        correction = self.spectral_output(self.spectral(spectral_features)).squeeze(-1)
        gate = self.gate(torch.cat((base_features, spectral_features), dim=-1)).squeeze(-1)
        return base_logit + gate * correction

    def base_logit(self, base_features: torch.Tensor) -> torch.Tensor:
        if self.config.variant != "gated_residual":
            raise ValueError("Only gated_residual has an embedded Stage-1 base logit")
        return self.base(base_features)
