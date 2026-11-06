"""Learned Cys-regime-conditioned residual heads for ProtTox-TSK."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ghxtox.models.physchem_features import PHYS_CHEM_FEATURE_NAMES
from ghxtox.models.tristat_head import TriStatMLPHead


CYSCONDITIONAL_REGIME_NAMES = ("cys_0_1", "cys_2_3", "cys_4_plus")


def cys_regime(cysteine_count: torch.Tensor) -> torch.Tensor:
    """Map count to 0–1, 2–3, 4+; this is a conditioning value, not a label."""
    return torch.where(cysteine_count <= 1, 0, torch.where(cysteine_count <= 3, 1, 2)).to(dtype=torch.long)


@dataclass(frozen=True)
class CysConditionalConfig:
    base_input_dim: int
    variant: str = "cys_conditional_residual"
    hidden_dim: int = 192
    residual_hidden_dim: int = 32
    dropout: float = 0.10
    physchem_dim: int = len(PHYS_CHEM_FEATURE_NAMES)


def _residual_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    head = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def _gate_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())


class CysConditionalResidualModel(nn.Module):
    """Stage-1 TriStat MLP with either shared or Cys-regime residual heads."""

    def __init__(self, config: CysConditionalConfig) -> None:
        super().__init__()
        if config.variant not in {"shared_residual", "cys_conditional_residual"}:
            raise ValueError("variant must be shared_residual or cys_conditional_residual")
        self.config = config
        self.base = TriStatMLPHead(config.base_input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout)
        condition_dim = 1 + config.physchem_dim
        head_count = 1 if config.variant == "shared_residual" else len(CYSCONDITIONAL_REGIME_NAMES)
        self.residual_heads = nn.ModuleList([_residual_head(condition_dim, config.residual_hidden_dim, config.dropout) for _ in range(head_count)])
        self.gate_heads = nn.ModuleList([_gate_head(condition_dim, config.residual_hidden_dim) for _ in range(head_count)])

    def forward(self, base_features: torch.Tensor, regimes: torch.Tensor) -> torch.Tensor:
        if base_features.ndim != 2 or base_features.shape[-1] != self.config.base_input_dim:
            raise ValueError(f"Base features must be [B, {self.config.base_input_dim}]")
        if regimes.ndim != 1 or len(regimes) != len(base_features) or not bool(((regimes >= 0) & (regimes < len(CYSCONDITIONAL_REGIME_NAMES))).all()):
            raise ValueError("Regimes must be a valid [B] tensor in 0..2")
        base_logit = self.base(base_features)
        condition = torch.cat((base_logit.unsqueeze(-1), base_features[:, -self.config.physchem_dim:]), dim=-1)
        corrections = torch.cat([head(condition) for head in self.residual_heads], dim=-1)
        gates = torch.cat([head(condition) for head in self.gate_heads], dim=-1)
        if self.config.variant == "shared_residual":
            correction, gate = corrections[:, 0], gates[:, 0]
        else:
            correction = corrections.gather(1, regimes.unsqueeze(-1)).squeeze(-1)
            gate = gates.gather(1, regimes.unsqueeze(-1)).squeeze(-1)
        return base_logit + gate * correction

    def base_logit(self, base_features: torch.Tensor) -> torch.Tensor:
        return self.base(base_features)
