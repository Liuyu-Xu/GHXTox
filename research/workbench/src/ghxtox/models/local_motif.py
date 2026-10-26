"""Frozen-residue local-motif branches for the isolated ProtTox route."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ghxtox.models.physchem_features import PHYS_CHEM_FEATURE_NAMES
from ghxtox.models.tristat_head import TriStatMLPHead


LOCAL_MOTIF_VARIANTS = ("nonlocal_projection_control", "multiscale_cnn_attention")


@dataclass(frozen=True)
class LocalMotifConfig:
    base_input_dim: int
    residue_embedding_dim: int
    variant: str = "multiscale_cnn_attention"
    hidden_dim: int = 192
    local_dim: int = 96
    convolution_dim: int = 32
    residual_hidden_dim: int = 64
    dropout: float = 0.10
    physchem_dim: int = len(PHYS_CHEM_FEATURE_NAMES)


def _residual_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    head = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def _gate_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())


class LocalMotifResidualModel(nn.Module):
    """TriStat Stage 1 plus a learned, zero-initialized local logit correction."""

    def __init__(self, config: LocalMotifConfig) -> None:
        super().__init__()
        if config.variant not in LOCAL_MOTIF_VARIANTS:
            raise ValueError(f"variant must be one of {LOCAL_MOTIF_VARIANTS}")
        if config.local_dim != 3 * config.convolution_dim:
            raise ValueError("local_dim must equal three convolution branches")
        self.config = config
        # Construct Stage 1 first so it has the same seeded initialisation as the retained MLP.
        self.base = TriStatMLPHead(config.base_input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout)
        self.residue_projection = nn.Sequential(nn.Linear(config.residue_embedding_dim, config.local_dim), nn.GELU())
        if config.variant == "multiscale_cnn_attention":
            self.convolutions = nn.ModuleList([nn.Conv1d(config.local_dim, config.convolution_dim, kernel_size=width, padding=width // 2) for width in (3, 5, 7)])
            self.attention = nn.Linear(config.local_dim, 1)
        else:
            candidate_branch = config.local_dim * config.convolution_dim * (3 + 5 + 7) + 3 * config.convolution_dim + config.local_dim + 1
            control_hidden = max(1, round((candidate_branch - config.local_dim) / (2 * config.local_dim + 1)))
            # At the default width this is 46,223 versus 46,273 parameters (0.11%).
            self.nonlocal_projection = nn.Sequential(nn.Linear(config.local_dim, control_hidden), nn.GELU(), nn.Linear(control_hidden, config.local_dim), nn.GELU())
        condition_dim = 1 + config.physchem_dim + config.local_dim
        self.residual = _residual_head(condition_dim, config.residual_hidden_dim, config.dropout)
        self.gate = _gate_head(condition_dim, config.residual_hidden_dim)

    def _local_features(self, residue_embeddings: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        projected = self.residue_projection(residue_embeddings.float())
        mask = residue_mask.to(dtype=torch.bool)
        if self.config.variant == "multiscale_cnn_attention":
            tokens = torch.cat([layer(projected.transpose(1, 2)).transpose(1, 2) for layer in self.convolutions], dim=-1)
            tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0.0)
            attention_logits = self.attention(tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
            weights = torch.softmax(attention_logits, dim=1)
            return torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=projected.dtype)
        pooled = (projected * mask.unsqueeze(-1)).sum(dim=1) / denominator
        return self.nonlocal_projection(pooled)

    def forward(self, base_features: torch.Tensor, residue_embeddings: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        if base_features.ndim != 2 or base_features.shape[-1] != self.config.base_input_dim:
            raise ValueError(f"Base features must be [B, {self.config.base_input_dim}]")
        if residue_embeddings.ndim != 3 or residue_embeddings.shape[:2] != residue_mask.shape or residue_embeddings.shape[0] != base_features.shape[0] or residue_embeddings.shape[-1] != self.config.residue_embedding_dim:
            raise ValueError("Residue embeddings and mask must align as [B, L, D] and [B, L]")
        if not bool(residue_mask.any(dim=1).all()):
            raise ValueError("Every sequence requires at least one valid residue embedding")
        base_logit = self.base(base_features)
        local = self._local_features(residue_embeddings, residue_mask)
        condition = torch.cat((base_logit.unsqueeze(-1), base_features[:, -self.config.physchem_dim :], local), dim=-1)
        return base_logit + self.gate(condition).squeeze(-1) * self.residual(condition).squeeze(-1)

    def base_logit(self, base_features: torch.Tensor) -> torch.Tensor:
        return self.base(base_features)

    def local_branch_parameter_count(self) -> int:
        names = ("residue_projection", "convolutions", "attention") if self.config.variant == "multiscale_cnn_attention" else ("residue_projection", "nonlocal_projection")
        return sum(parameter.numel() for name, parameter in self.named_parameters() if any(name.startswith(prefix) for prefix in names))
