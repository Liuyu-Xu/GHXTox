"""Shared checkpoint contract for the independent spectral-refinement model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ghxtox.models import SpectralRefineConfig, SpectralRefineModel
from ghxtox.models.spectral_physchem import SPECTRAL_FEATURE_VERSION
from ghxtox.prottox_tsk import MODEL_VERSION as BASE_MODEL_VERSION, file_sha256, write_manifest


MODEL_NAME = "ProtTox-SpectralRefine"
MODEL_VERSION = "prottox-spectral-refine-v1"


def checkpoint_payload(model: SpectralRefineModel, *, base_mean: torch.Tensor, base_std: torch.Tensor, spectral_mean: torch.Tensor, spectral_std: torch.Tensor, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "base_model_version": BASE_MODEL_VERSION, "spectral_feature_version": SPECTRAL_FEATURE_VERSION, "config": asdict(model.config), "model_state": model.state_dict(), "base_mean": base_mean.detach().cpu().float(), "base_std": base_std.detach().cpu().float(), "spectral_mean": spectral_mean.detach().cpu().float(), "spectral_std": spectral_std.detach().cpu().float(), "metadata": metadata}


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[SpectralRefineModel, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_name") != MODEL_NAME or payload.get("model_version") != MODEL_VERSION or payload.get("spectral_feature_version") != SPECTRAL_FEATURE_VERSION:
        raise ValueError("Checkpoint is not a compatible ProtTox-SpectralRefine v1 checkpoint")
    model = SpectralRefineModel(SpectralRefineConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    base_mean = torch.as_tensor(payload["base_mean"], dtype=torch.float32, device=device)
    base_std = torch.as_tensor(payload["base_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    spectral_mean = torch.as_tensor(payload["spectral_mean"], dtype=torch.float32, device=device)
    spectral_std = torch.as_tensor(payload["spectral_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    if base_mean.numel() != model.config.base_input_dim or spectral_mean.numel() != model.config.spectral_dim:
        raise ValueError("Checkpoint normalization dimensions do not match its spectral-refinement config")
    return model, base_mean, base_std, spectral_mean, spectral_std, dict(payload.get("metadata", {}))


__all__ = ["MODEL_NAME", "MODEL_VERSION", "checkpoint_payload", "file_sha256", "load_checkpoint", "write_manifest"]
