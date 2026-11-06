"""Masked mean, max and standard-deviation pooling for residue embeddings."""

from __future__ import annotations

import torch
from torch import nn


def tristat_pool(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked mean, max, and population standard deviation.

    Args:
        embeddings: Float tensor shaped ``[batch, residues, embedding_dim]``.
        mask: Boolean or 0/1 tensor shaped ``[batch, residues]``.  Every
            sequence must contain at least one valid residue.
    """
    if embeddings.ndim != 3:
        raise ValueError(f"embeddings must be [B, L, D], got {tuple(embeddings.shape)}")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(embeddings.shape[:2]):
        raise ValueError("mask must be [B, L] and match the first two embedding dimensions")
    valid = mask.to(device=embeddings.device, dtype=torch.bool)
    counts = valid.sum(dim=1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError("TriStat pooling requires at least one valid residue per sequence")
    expanded = valid.unsqueeze(-1)
    values = embeddings * expanded.to(dtype=embeddings.dtype)
    mean = values.sum(dim=1) / counts.to(dtype=embeddings.dtype)
    centered = (embeddings - mean.unsqueeze(1)) * expanded.to(dtype=embeddings.dtype)
    std = torch.sqrt((centered.square().sum(dim=1) / counts.to(dtype=embeddings.dtype)).clamp_min(0.0))
    max_values = embeddings.masked_fill(~expanded, torch.finfo(embeddings.dtype).min).max(dim=1).values
    return torch.cat((mean, max_values, std), dim=-1)


class TriStatPooling(nn.Module):
    """Module wrapper around :func:`tristat_pool` for model composition."""

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return tristat_pool(embeddings, mask)
