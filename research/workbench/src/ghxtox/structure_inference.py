"""Pure 3D structure-expert inference for the current public default.

This module deliberately has no dependency on the previous-generation
ProtT5-TextCNN, cross-modal adapter, or BACC residual pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device, resolve_inference_checkpoint


def predict_structure(
    processed_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Run only the pLDDT-aware 3D expert on a processed structure cache."""

    dataset = PeptideTensorDataset(processed_path, require_labels=False)
    checkpoint, selected, _, fallback_used = resolve_inference_checkpoint(
        checkpoint_path, dataset.records, device, requested_threshold=None
    )
    config = checkpoint["config"]
    model = GHXToxModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    required_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
    validate_plm_feature_dim(dataset.records, required_dim, processed_path)
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, collate_fn=collate_peptides
    )
    probabilities: list[torch.Tensor] = []
    base_probabilities: list[torch.Tensor] = []
    conformer_deltas: list[torch.Tensor] = []
    conformer_available: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    mean_plddt: list[torch.Tensor] = []
    sample_ids: list[str] = []
    sequences: list[str] = []
    labels: list[int | None] = []
    with torch.inference_mode():
        for batch in loader:
            sample_ids.extend(batch["sample_id"])
            sequences.extend(batch["sequence"])
            batch_labels = batch["labels"]
            labels.extend(
                [None] * len(batch["sample_id"])
                if batch_labels is None
                else [int(value) for value in batch_labels.tolist()]
            )
            residue_mask = batch["mask"].bool()
            confidence = batch["plddt"].float()
            mean_plddt.append(
                (
                    (confidence * residue_mask.float()).sum(dim=1)
                    / residue_mask.sum(dim=1).clamp_min(1).float()
                ).cpu()
            )
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            probabilities.append(torch.sigmoid(output["logits"]).float().cpu())
            base_logits = output.get("base_logits", output["logits"])
            base_probabilities.append(torch.sigmoid(base_logits).float().cpu())
            conformer_deltas.append(
                output.get("conformer_delta", torch.zeros_like(output["logits"]))
                .float()
                .cpu()
            )
            conformer_available.append(
                output.get(
                    "conformer_available",
                    torch.zeros_like(output["logits"], dtype=torch.bool),
                )
                .bool()
                .cpu()
            )
            gates.append(output["global_gate"].float().cpu())
    return {
        "sample_ids": sample_ids,
        "sequences": sequences,
        "labels": labels,
        "probabilities": torch.cat(probabilities).numpy(),
        "base_probabilities": torch.cat(base_probabilities).numpy(),
        "conformer_deltas": torch.cat(conformer_deltas).numpy(),
        "conformer_available": torch.cat(conformer_available).numpy(),
        "gates": torch.cat(gates).numpy(),
        "mean_plddt": torch.cat(mean_plddt).numpy(),
        "checkpoint": selected,
        "fallback_used": fallback_used,
    }


_predict_structure = predict_structure
