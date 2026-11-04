"""Checkpoint contract for the Cys-conditioned residual candidate."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ghxtox.models.cys_conditional import CysConditionalConfig, CysConditionalResidualModel
from ghxtox.prottox_tsk import MODEL_VERSION as BASE_MODEL_VERSION, write_manifest


MODEL_NAME = "ProtTox-CysConditional"
MODEL_VERSION = "prottox-cys-conditional-v1"


def checkpoint_payload(model: CysConditionalResidualModel, *, base_mean: torch.Tensor, base_std: torch.Tensor, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "base_model_version": BASE_MODEL_VERSION, "config": asdict(model.config), "model_state": model.state_dict(), "base_mean": base_mean.detach().cpu().float(), "base_std": base_std.detach().cpu().float(), "metadata": metadata}


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[CysConditionalResidualModel, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_name") != MODEL_NAME or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("Checkpoint is not a compatible ProtTox-CysConditional v1 checkpoint")
    model = CysConditionalResidualModel(CysConditionalConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    mean = torch.as_tensor(payload["base_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(payload["base_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    if mean.numel() != model.config.base_input_dim:
        raise ValueError("Checkpoint normalization dimension does not match its model config")
    return model, mean, std, dict(payload.get("metadata", {}))
