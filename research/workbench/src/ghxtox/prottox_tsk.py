"""Shared, isolated model and cache contracts for the ProtTox-TSK route.

This module intentionally does not import the GHXTox LoRA/Mamba or 3D model
paths.  It consumes frozen residue embeddings and fixed sequence descriptors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ghxtox.models import PHYS_CHEM_FEATURE_NAMES, build_tristat_head, tristat_pool


MODEL_NAME = "ProtTox-TSK"
MODEL_VERSION = "prottox-tsk-v1"
CACHE_FORMAT_VERSION = 1


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256("".join(sequence.upper().split()).encode("ascii")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProtToxTSKConfig:
    embedding_dim: int
    head_kind: str = "mlp"
    hidden_dim: int = 192
    dropout: float = 0.10
    grid_size: int = 8
    model_version: str = MODEL_VERSION

    @property
    def input_dim(self) -> int:
        return 3 * self.embedding_dim + len(PHYS_CHEM_FEATURE_NAMES)


class ProtToxTSKModel(nn.Module):
    """Frozen-encoder downstream classifier; the encoder is deliberately absent."""

    def __init__(self, config: ProtToxTSKConfig) -> None:
        super().__init__()
        self.config = config
        self.head = build_tristat_head(
            config.head_kind,
            config.input_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            grid_size=config.grid_size,
        )

    def combine_features(self, tristat: torch.Tensor, physchem: torch.Tensor) -> torch.Tensor:
        if tristat.ndim != 2 or tristat.shape[-1] != 3 * self.config.embedding_dim:
            raise ValueError(f"TriStat features must be [B, {3 * self.config.embedding_dim}]")
        if physchem.ndim != 2 or physchem.shape[-1] != len(PHYS_CHEM_FEATURE_NAMES):
            raise ValueError(f"PhysChem features must be [B, {len(PHYS_CHEM_FEATURE_NAMES)}]")
        return torch.cat((tristat, physchem.to(dtype=tristat.dtype)), dim=-1)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.config.input_dim:
            raise ValueError(f"Combined features must be [B, {self.config.input_dim}]")
        return self.head(features)

    def forward(self, residue_embeddings: torch.Tensor, residue_mask: torch.Tensor, physchem: torch.Tensor) -> torch.Tensor:
        return self.forward_features(self.combine_features(tristat_pool(residue_embeddings, residue_mask), physchem))


def checkpoint_payload(model: ProtToxTSKModel, *, feature_mean: torch.Tensor, feature_std: torch.Tensor, metadata: dict[str, Any]) -> dict[str, Any]:
    """Create a self-describing checkpoint with explicit cache/model provenance."""
    return {
        "model_name": MODEL_NAME,
        "model_version": model.config.model_version,
        "config": asdict(model.config),
        "model_state": model.state_dict(),
        "feature_mean": feature_mean.detach().cpu().float(),
        "feature_std": feature_std.detach().cpu().float(),
        "metadata": metadata,
    }


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[ProtToxTSKModel, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_name") != MODEL_NAME or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("Checkpoint is not a compatible ProtTox-TSK v1 checkpoint")
    config = ProtToxTSKConfig(**payload["config"])
    model = ProtToxTSKModel(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    mean = torch.as_tensor(payload["feature_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(payload["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    if mean.numel() != config.input_dim or std.numel() != config.input_dim:
        raise ValueError("Checkpoint feature normalization dimension does not match its model config")
    return model, mean, std, dict(payload.get("metadata", {}))


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
