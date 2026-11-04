"""Small MLP control and spline-KAN candidate heads for ProtTox-TSK."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TriStatMLPHead(nn.Module):
    """Compact normalized MLP control used to evaluate the KAN candidate."""

    def __init__(self, input_dim: int, hidden_dim: int = 192, dropout: float = 0.10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class SplineKANLinear(nn.Module):
    """A compact, learnable piecewise-linear KAN layer.

    Each input/output pair owns an independent triangular B-spline expansion;
    outputs sum those learned univariate functions.  A linear base branch keeps
    the small head numerically stable during the smoke stage.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8) -> None:
        super().__init__()
        if grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        self.in_features, self.out_features, self.grid_size = in_features, out_features, grid_size
        self.base = nn.Linear(in_features, out_features)
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        self.register_buffer("centers", torch.linspace(-3.0, 3.0, grid_size))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.in_features:
            raise ValueError(f"KAN input must be [B, {self.in_features}], got {tuple(inputs.shape)}")
        scaled = inputs.clamp(-3.0, 3.0)
        width = float(self.centers[1] - self.centers[0])
        basis = F.relu(1.0 - (scaled.unsqueeze(-1) - self.centers).abs() / width)
        spline = torch.einsum("big,oig->bo", basis, self.spline_weight)
        return self.base(inputs) + spline


class TriStatKANHead(nn.Module):
    """Small fusion projection followed by two spline-KAN transformations."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, grid_size: int = 8, dropout: float = 0.05) -> None:
        super().__init__()
        self.normalizer = nn.LayerNorm(input_dim)
        self.fusion = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.kan_hidden = SplineKANLinear(hidden_dim, hidden_dim // 2, grid_size=grid_size)
        self.kan_output = SplineKANLinear(hidden_dim // 2, 1, grid_size=grid_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(F.gelu(self.fusion(self.normalizer(features))))
        hidden = F.gelu(self.kan_hidden(hidden))
        return self.kan_output(hidden).squeeze(-1)


def build_tristat_head(kind: str, input_dim: int, **kwargs: int | float) -> nn.Module:
    """Build either explicitly named head; no implicit KAN promotion occurs."""
    normalized = kind.lower().replace("_", "-")
    if normalized in {"mlp", "tristat-mlp"}:
        return TriStatMLPHead(input_dim, hidden_dim=int(kwargs.get("hidden_dim", 192)), dropout=float(kwargs.get("dropout", 0.10)))
    if normalized in {"kan", "tristat-kan"}:
        return TriStatKANHead(input_dim, hidden_dim=int(kwargs.get("hidden_dim", 128)), grid_size=int(kwargs.get("grid_size", 8)), dropout=float(kwargs.get("dropout", 0.05)))
    raise ValueError("kind must be 'mlp' or 'kan'")
