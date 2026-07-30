"""Leakage-controlled ProtT5-to-3D residue interaction adapter.

The adapter is deliberately asymmetric: frozen ProtT5 residue tokens query
out-of-fold spatial/chemical residue states.  Its bounded, confidence-gated
logit residual is added to an immutable baseline probability, so a missing or
dropped structure is an exact baseline fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.folds import load_fold_indices
from ghxtox.metrics import binary_metrics
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device, resolve_device, set_seed


FORMAT_VERSION = 1


def _safe_logit(probability: torch.Tensor) -> torch.Tensor:
    clipped = probability.float().clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(clipped)


def _load_token_cache(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    shard_paths = sorted(root.glob("shard_*.pt"))
    if not shard_paths:
        raise ValueError(f"No ProtT5 shard_*.pt files found under {root}.")
    embeddings = []
    offsets = [0]
    sample_ids: list[str] = []
    sequences: list[str] = []
    labels: list[int] = []
    feature_dim: int | None = None
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_embeddings = payload["embeddings"]
        shard_offsets = payload["offsets"].long()
        if shard_embeddings.ndim != 2 or shard_offsets.ndim != 1:
            raise ValueError(f"Malformed token shard {shard_path}.")
        if feature_dim is None:
            feature_dim = int(shard_embeddings.shape[1])
        elif int(shard_embeddings.shape[1]) != feature_dim:
            raise ValueError("ProtT5 token dimensions differ between shards.")
        embeddings.append(shard_embeddings.cpu())
        base = offsets[-1]
        offsets.extend(base + int(value) for value in shard_offsets[1:].tolist())
        sample_ids.extend(str(value) for value in payload["sample_ids"])
        sequences.extend(str(value) for value in payload["sequences"])
        labels.extend(int(value) for value in payload["labels"].long().tolist())
    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "sample_ids": sample_ids,
        "sequences": sequences,
        "labels": torch.tensor(labels, dtype=torch.long),
        "feature_dim": int(feature_dim or 0),
    }


def _load_baseline_probabilities(
    path: str | Path,
    sequences: list[str],
    labels: torch.Tensor,
) -> torch.Tensor:
    rows = list(csv.DictReader(Path(path).open("r", encoding="utf-8", newline="")))
    if len(rows) != len(sequences):
        raise ValueError(f"Baseline contains {len(rows)} rows; expected {len(sequences)}.")
    probability_columns = ("toxicity_probability", "fusion_probability")
    probabilities = []
    for index, (row, sequence, label) in enumerate(zip(rows, sequences, labels.tolist())):
        if row.get("sequence") != sequence:
            raise ValueError(f"Baseline sequence mismatch at source index {index}.")
        if int(row["label"]) != int(label):
            raise ValueError(f"Baseline label mismatch at source index {index}.")
        column = next((name for name in probability_columns if row.get(name) not in {None, ""}), None)
        if column is None:
            raise ValueError(f"Baseline row {index} has no supported probability column.")
        probabilities.append(float(row[column]))
    result = torch.tensor(probabilities, dtype=torch.float32)
    if not torch.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Baseline probabilities must be finite values in [0, 1].")
    return result


def _fold_assignments(path: str | Path, count: int) -> torch.Tensor:
    rows = list(csv.DictReader(Path(path).open("r", encoding="utf-8", newline="")))
    assignments = torch.full((count,), -1, dtype=torch.long)
    for row in rows:
        index = int(row["source_index"])
        if index < 0 or index >= count or assignments[index] >= 0:
            raise ValueError(f"Invalid or duplicate fold source index {index}.")
        assignments[index] = int(row["fold"])
    if (assignments < 0).any():
        raise ValueError("Fold manifest does not cover every sample.")
    return assignments


def precompute_oof_features(
    *,
    processed: str | Path,
    token_cache: str | Path,
    baseline_predictions: str | Path,
    fold_manifest: str | Path,
    checkpoint_pattern: str,
    output: str | Path,
    device_name: str = "cuda",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Extract fold-pure spatial nodes and align them with frozen ProtT5 tokens."""

    device = resolve_device(device_name)
    dataset = PeptideTensorDataset(processed, require_labels=True)
    tokens = _load_token_cache(token_cache)
    if len(dataset) != len(tokens["sequences"]):
        raise ValueError("Processed data and ProtT5 token cache have different sample counts.")

    record_sequences = [str(record["sequence"]) for record in dataset.records]
    record_labels = torch.tensor([int(record["label"]) for record in dataset.records])
    if record_sequences != tokens["sequences"]:
        mismatch = next(
            index
            for index, (left, right) in enumerate(zip(record_sequences, tokens["sequences"]))
            if left != right
        )
        raise ValueError(f"Processed/token sequence mismatch at source index {mismatch}.")
    if not torch.equal(record_labels, tokens["labels"]):
        raise ValueError("Processed/token labels do not match.")

    baseline = _load_baseline_probabilities(
        baseline_predictions, record_sequences, record_labels
    )
    folds = _fold_assignments(fold_manifest, len(dataset))
    num_folds = int(folds.max().item()) + 1
    spatial_by_index: list[torch.Tensor | None] = [None] * len(dataset)
    plddt_by_index: list[torch.Tensor | None] = [None] * len(dataset)
    hidden_dim: int | None = None

    for fold in range(num_folds):
        checkpoint_path = Path(checkpoint_pattern.format(fold=fold))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        required_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
        validate_plm_feature_dim(dataset.records, required_dim, processed)
        model = GHXToxModel(config)
        model.load_state_dict(checkpoint["model_state"])
        model.eval().to(device)
        _, validation_indices = load_fold_indices(
            fold_manifest, fold, len(dataset)
        )
        loader = DataLoader(
            Subset(dataset, validation_indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(
                collate_peptides, include_structure=True, include_atom=False
            ),
        )
        position = 0
        with torch.inference_mode():
            for batch in loader:
                batch_on_device = move_batch_to_device(batch, device)
                output_values = model(batch_on_device)
                spatial = output_values.get("spatial_nodes")
                if spatial is None:
                    raise ValueError("Structure checkpoint did not expose spatial residue states.")
                if hidden_dim is None:
                    hidden_dim = int(spatial.shape[-1])
                indices = validation_indices[position : position + spatial.shape[0]]
                position += spatial.shape[0]
                for row, source_index in enumerate(indices):
                    length = int(batch["lengths"][row])
                    spatial_by_index[source_index] = (
                        spatial[row, :length].detach().to(device="cpu", dtype=torch.float16)
                    )
                    plddt_by_index[source_index] = (
                        batch["plddt"][row, :length].detach().cpu().float()
                    )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if any(value is None for value in spatial_by_index):
        raise RuntimeError("OOF spatial feature extraction did not cover every sample.")
    if any(value is None for value in plddt_by_index):
        raise RuntimeError("OOF pLDDT extraction did not cover every sample.")

    spatial_tensors = [value for value in spatial_by_index if value is not None]
    plddt_tensors = [value for value in plddt_by_index if value is not None]
    spatial_offsets = [0]
    for value in spatial_tensors:
        spatial_offsets.append(spatial_offsets[-1] + int(value.shape[0]))
    if spatial_offsets != tokens["offsets"].tolist():
        raise ValueError("ProtT5 and structure residue offsets are not exactly aligned.")

    payload = {
        "format_version": FORMAT_VERSION,
        "token_embeddings": tokens["embeddings"],
        "spatial_embeddings": torch.cat(spatial_tensors, dim=0),
        "plddt": torch.cat(plddt_tensors, dim=0),
        "offsets": tokens["offsets"],
        "base_probabilities": baseline,
        "labels": record_labels,
        "folds": folds,
        "sample_ids": [str(record["sample_id"]) for record in dataset.records],
        "sequences": record_sequences,
        "metadata": {
            "processed": str(processed),
            "token_cache": str(token_cache),
            "baseline_predictions": str(baseline_predictions),
            "fold_manifest": str(fold_manifest),
            "checkpoint_pattern": checkpoint_pattern,
            "num_samples": len(dataset),
            "num_residues": int(tokens["embeddings"].shape[0]),
            "token_dim": int(tokens["feature_dim"]),
            "spatial_dim": int(hidden_dim or 0),
            "protocol": "each spatial state comes from the checkpoint that excluded its fold",
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return payload["metadata"]


def precompute_inference_features(
    *,
    processed: str | Path,
    token_cache: str | Path,
    baseline_predictions: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    device_name: str = "cuda",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Extract full-refit structure states for a frozen evaluation dataset."""

    device = resolve_device(device_name)
    dataset = PeptideTensorDataset(processed, require_labels=True)
    tokens = _load_token_cache(token_cache)
    sequences = [str(record["sequence"]) for record in dataset.records]
    labels = torch.tensor([int(record["label"]) for record in dataset.records])
    if sequences != tokens["sequences"] or not torch.equal(labels, tokens["labels"]):
        raise ValueError("Processed data and token cache are not exactly aligned.")
    baseline = _load_baseline_probabilities(
        baseline_predictions, sequences, labels
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_payload["config"]
    required_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
    validate_plm_feature_dim(dataset.records, required_dim, processed)
    model = GHXToxModel(config)
    model.load_state_dict(checkpoint_payload["model_state"])
    model.eval().to(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_peptides, include_structure=True, include_atom=False
        ),
    )
    spatial_tensors = []
    plddt_tensors = []
    hidden_dim: int | None = None
    with torch.inference_mode():
        for batch in loader:
            batch_on_device = move_batch_to_device(batch, device)
            output_values = model(batch_on_device)
            spatial = output_values.get("spatial_nodes")
            if spatial is None:
                raise ValueError("Structure checkpoint did not expose spatial residue states.")
            hidden_dim = int(spatial.shape[-1])
            for row, length_value in enumerate(batch["lengths"].tolist()):
                length = int(length_value)
                spatial_tensors.append(
                    spatial[row, :length].detach().to(device="cpu", dtype=torch.float16)
                )
                plddt_tensors.append(batch["plddt"][row, :length].detach().cpu().float())
    spatial_offsets = [0]
    for value in spatial_tensors:
        spatial_offsets.append(spatial_offsets[-1] + int(value.shape[0]))
    if spatial_offsets != tokens["offsets"].tolist():
        raise ValueError("ProtT5 and full-refit structure residue offsets are not aligned.")
    payload = {
        "format_version": FORMAT_VERSION,
        "token_embeddings": tokens["embeddings"],
        "spatial_embeddings": torch.cat(spatial_tensors, dim=0),
        "plddt": torch.cat(plddt_tensors, dim=0),
        "offsets": tokens["offsets"],
        "base_probabilities": baseline,
        "labels": labels,
        "folds": torch.zeros(len(dataset), dtype=torch.long),
        "sample_ids": [str(record["sample_id"]) for record in dataset.records],
        "sequences": sequences,
        "metadata": {
            "processed": str(processed),
            "token_cache": str(token_cache),
            "baseline_predictions": str(baseline_predictions),
            "checkpoint": str(checkpoint),
            "num_samples": len(dataset),
            "num_residues": int(tokens["embeddings"].shape[0]),
            "token_dim": int(tokens["feature_dim"]),
            "spatial_dim": int(hidden_dim or 0),
            "protocol": "frozen full-refit structure states for external evaluation",
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return payload["metadata"]


def attach_chemical_features(
    feature_cache: str | Path,
    processed: str | Path,
) -> dict[str, Any]:
    """Attach rotation-invariant per-residue functional-site summaries in place."""

    cache_path = Path(feature_cache)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    processed_payload = torch.load(processed, map_location="cpu", weights_only=False)
    records = processed_payload["records"]
    if len(records) != int(payload["labels"].numel()):
        raise ValueError("Processed data and adapter cache have different sample counts.")
    chemical_features = []
    offsets = [0]
    for index, record in enumerate(records):
        if str(record["sequence"]) != payload["sequences"][index]:
            raise ValueError(f"Chemical feature sequence mismatch at source index {index}.")
        length = len(record["sequence"])
        site_types = record.get("chemical_site_types")
        site_coords = record.get("chemical_site_coords")
        site_mask = record.get("chemical_site_mask")
        residue_coords = record.get("coords")
        if not all(
            torch.is_tensor(value)
            for value in (site_types, site_coords, site_mask, residue_coords)
        ):
            raise ValueError(f"Record {index} has no complete chemical-site tensors.")
        site_types = site_types.float()
        site_coords = site_coords.float()
        site_mask = site_mask.bool()
        residue_coords = residue_coords.float()
        valid = site_mask.float()
        type_counts = (site_types * valid.unsqueeze(-1)).sum(dim=1).clamp_max(2.0) / 2.0
        count = valid.sum(dim=1, keepdim=True)
        distances = torch.linalg.vector_norm(
            site_coords - residue_coords.unsqueeze(1), dim=-1
        )
        mean_distance = (distances * valid).sum(dim=1, keepdim=True) / count.clamp_min(1.0)
        max_distance = distances.masked_fill(~site_mask, 0.0).max(dim=1, keepdim=True).values
        pair_distance = torch.zeros(length, 1)
        if site_coords.shape[1] >= 2:
            pair_valid = site_mask[:, 0] & site_mask[:, 1]
            pair_value = torch.linalg.vector_norm(
                site_coords[:, 0] - site_coords[:, 1], dim=-1
            )
            pair_distance[:, 0] = pair_value * pair_valid.float()
        summary = torch.cat(
            [
                type_counts,
                (count / max(int(site_coords.shape[1]), 1)).clamp(0.0, 1.0),
                mean_distance / 10.0,
                max_distance / 10.0,
                pair_distance / 10.0,
            ],
            dim=-1,
        )
        if summary.shape[0] != length:
            raise ValueError(f"Chemical feature length mismatch at source index {index}.")
        chemical_features.append(summary.to(dtype=torch.float16))
        offsets.append(offsets[-1] + length)
    if offsets != payload["offsets"].tolist():
        raise ValueError("Chemical and ProtT5 residue offsets are not aligned.")
    payload["chemical_features"] = torch.cat(chemical_features, dim=0)
    payload["metadata"]["chemical_dim"] = int(chemical_features[0].shape[1])
    payload["metadata"]["chemical_feature_definition"] = (
        "8 site-type occupancies + site count + mean/max site radius + within-residue pair distance"
    )
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    return {
        "feature_cache": str(cache_path),
        "num_samples": len(records),
        "num_residues": int(payload["chemical_features"].shape[0]),
        "chemical_dim": int(payload["metadata"]["chemical_dim"]),
    }


class CoupledFeatureDataset(Dataset):
    def __init__(self, payload: dict[str, Any]) -> None:
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported cross-modal feature-cache format.")
        self.payload = payload

    def __len__(self) -> int:
        return int(self.payload["labels"].numel())

    def __getitem__(self, index: int) -> dict[str, Any]:
        offsets = self.payload["offsets"]
        start, stop = int(offsets[index]), int(offsets[index + 1])
        result = {
            "source_index": index,
            "sample_id": self.payload["sample_ids"][index],
            "sequence": self.payload["sequences"][index],
            "tokens": self.payload["token_embeddings"][start:stop],
            "spatial": self.payload["spatial_embeddings"][start:stop],
            "plddt": self.payload["plddt"][start:stop],
            "base_probability": self.payload["base_probabilities"][index],
            "label": self.payload["labels"][index],
            "fold": self.payload["folds"][index],
        }
        if "chemical_features" in self.payload:
            result["chemical"] = self.payload["chemical_features"][start:stop]
        if "geometry_ssl_embeddings" in self.payload:
            result["geometry_ssl"] = self.payload["geometry_ssl_embeddings"][start:stop]
        return result


def collate_coupled(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_length = max(int(item["tokens"].shape[0]) for item in batch)
    token_dim = int(batch[0]["tokens"].shape[1])
    spatial_dim = int(batch[0]["spatial"].shape[1])
    tokens = torch.zeros(len(batch), max_length, token_dim, dtype=torch.float32)
    spatial = torch.zeros(len(batch), max_length, spatial_dim, dtype=torch.float32)
    plddt = torch.zeros(len(batch), max_length, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    for index, item in enumerate(batch):
        length = int(item["tokens"].shape[0])
        tokens[index, :length] = item["tokens"].float()
        spatial[index, :length] = item["spatial"].float()
        plddt[index, :length] = item["plddt"].float()
        mask[index, :length] = True
    result = {
        "source_index": torch.tensor([item["source_index"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
        "sequence": [item["sequence"] for item in batch],
        "tokens": tokens,
        "spatial": spatial,
        "plddt": plddt,
        "mask": mask,
        "base_probability": torch.stack(
            [item["base_probability"].float() for item in batch]
        ),
        "labels": torch.stack([item["label"].float() for item in batch]),
        "fold": torch.stack([item["fold"].long() for item in batch]),
    }
    if all("chemical" in item for item in batch):
        chemical_dim = int(batch[0]["chemical"].shape[1])
        chemical = torch.zeros(
            len(batch), max_length, chemical_dim, dtype=torch.float32
        )
        for index, item in enumerate(batch):
            length = int(item["chemical"].shape[0])
            chemical[index, :length] = item["chemical"].float()
        result["chemical"] = chemical
    if all("geometry_ssl" in item for item in batch):
        geometry_dim = int(batch[0]["geometry_ssl"].shape[1])
        geometry_ssl = torch.zeros(
            len(batch), max_length, geometry_dim, dtype=torch.float32
        )
        for index, item in enumerate(batch):
            length = int(item["geometry_ssl"].shape[0])
            geometry_ssl[index, :length] = item["geometry_ssl"].float()
        result["geometry_ssl"] = geometry_ssl
    return result


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    minimum = torch.finfo(values.dtype).min
    pooled = values.masked_fill(~mask.unsqueeze(-1), minimum).max(dim=1).values
    return torch.where(mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))


class ProtT5StructureAdapter(nn.Module):
    """One-way pLDDT-gated cross-attention with a bounded logit residual."""

    def __init__(
        self,
        token_dim: int,
        spatial_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
        initial_interaction_weight: float = 0.10,
        max_logit_delta: float = 1.25,
        chemical_dim: int = 0,
        geometry_ssl_dim: int = 0,
        hierarchical_fusion: bool = False,
        num_motifs: int = 4,
    ) -> None:
        super().__init__()
        self.max_logit_delta = float(max_logit_delta)
        self.chemical_dim = max(int(chemical_dim), 0)
        self.geometry_ssl_dim = max(int(geometry_ssl_dim), 0)
        self.hierarchical_fusion = bool(hierarchical_fusion)
        self.num_motifs = max(int(num_motifs), 1)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
        )
        self.spatial_projection = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, hidden_dim),
            nn.GELU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.chemical_projection = None
        self.chemical_gate = None
        if self.chemical_dim > 0:
            self.chemical_projection = nn.Sequential(
                nn.LayerNorm(self.chemical_dim),
                nn.Linear(self.chemical_dim, hidden_dim),
                nn.GELU(),
            )
            self.chemical_gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        self.geometry_ssl_projection = None
        self.geometry_ssl_gate = None
        if self.geometry_ssl_dim > 0:
            self.geometry_ssl_projection = nn.Sequential(
                nn.LayerNorm(self.geometry_ssl_dim),
                nn.Linear(self.geometry_ssl_dim, hidden_dim),
                nn.GELU(),
            )
            self.geometry_ssl_gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        delta_input_dim = hidden_dim * 3
        self.local_fusion = None
        self.motif_queries = None
        self.motif_attention = None
        self.global_motif_attention = None
        self.sequence_aux_head = None
        self.structure_aux_head = None
        if self.hierarchical_fusion:
            self.local_fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
            )
            self.motif_queries = nn.Parameter(
                torch.randn(1, self.num_motifs, hidden_dim) * 0.02
            )
            self.motif_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.global_motif_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.sequence_aux_head = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.structure_aux_head = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            delta_input_dim = hidden_dim * 6
        self.delta_head = nn.Sequential(
            nn.Linear(delta_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        initial = min(max(float(initial_interaction_weight), 1e-4), 1.0 - 1e-4)
        self.interaction_logit = nn.Parameter(torch.logit(torch.tensor(initial)))
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        *,
        tokens: torch.Tensor,
        spatial: torch.Tensor,
        plddt: torch.Tensor,
        mask: torch.Tensor,
        base_probability: torch.Tensor,
        structure_available: torch.Tensor | None = None,
        chemical: torch.Tensor | None = None,
        geometry_ssl: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = mask.bool()
        token_h = self.token_projection(tokens)
        confidence = plddt.clamp(0.0, 1.0) * mask.to(plddt.dtype)
        spatial_h = self.spatial_projection(spatial)
        if self.chemical_projection is not None:
            if chemical is None:
                raise ValueError("The chemical-site adapter requires chemical features.")
            chemical_h = self.chemical_projection(chemical)
            chemical_gate = self.chemical_gate(
                torch.cat([token_h, chemical_h, token_h * chemical_h], dim=-1)
            )
            spatial_h = spatial_h + chemical_gate * chemical_h
        if self.geometry_ssl_projection is not None:
            if geometry_ssl is None:
                raise ValueError("The geometry-SSL adapter requires geometry embeddings.")
            geometry_h = self.geometry_ssl_projection(geometry_ssl)
            geometry_gate = self.geometry_ssl_gate(
                torch.cat([token_h, geometry_h, token_h * geometry_h], dim=-1)
            )
            spatial_h = spatial_h + geometry_gate * geometry_h
        spatial_h = spatial_h * confidence.unsqueeze(-1)
        context, _ = self.cross_attention(
            query=token_h,
            key=spatial_h,
            value=spatial_h,
            key_padding_mask=~mask,
            need_weights=False,
        )
        context = self.context_norm(context)
        sequence_aux_logits = None
        structure_aux_logits = None
        if self.hierarchical_fusion:
            local_h = self.local_fusion(
                torch.cat(
                    [
                        token_h,
                        spatial_h,
                        token_h * spatial_h,
                        (token_h - spatial_h).abs(),
                    ],
                    dim=-1,
                )
            )
            motif_query = self.motif_queries.expand(token_h.shape[0], -1, -1)
            motifs, _ = self.motif_attention(
                query=motif_query,
                key=local_h,
                value=local_h,
                key_padding_mask=~mask,
                need_weights=False,
            )
            peptide_query = _masked_mean(token_h, mask).unsqueeze(1)
            global_context, _ = self.global_motif_attention(
                query=peptide_query,
                key=motifs,
                value=motifs,
                need_weights=False,
            )
            interaction = torch.cat(
                [
                    _masked_mean(local_h, mask),
                    _masked_max(local_h, mask),
                    motifs.mean(dim=1),
                    motifs.max(dim=1).values,
                    global_context.squeeze(1),
                    _masked_mean(token_h * context, mask),
                ],
                dim=-1,
            )
            sequence_aux_logits = self.sequence_aux_head(
                torch.cat(
                    [_masked_mean(token_h, mask), _masked_max(token_h, mask)],
                    dim=-1,
                )
            ).squeeze(-1)
            structure_aux_logits = self.structure_aux_head(
                torch.cat(
                    [_masked_mean(spatial_h, mask), _masked_max(spatial_h, mask)],
                    dim=-1,
                )
            ).squeeze(-1)
        else:
            interaction = torch.cat(
                [
                    _masked_mean(context, mask),
                    _masked_max(context, mask),
                    _masked_mean(token_h * context, mask),
                ],
                dim=-1,
            )
        raw_delta = torch.tanh(self.delta_head(interaction).squeeze(-1))
        mean_confidence = confidence.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        quality_gate = torch.sigmoid((mean_confidence - 0.55) / 0.12)
        interaction_weight = torch.sigmoid(self.interaction_logit)
        if structure_available is None:
            structure_available = torch.ones_like(base_probability)
        delta = (
            structure_available.to(base_probability.dtype)
            * quality_gate
            * interaction_weight
            * self.max_logit_delta
            * raw_delta
        )
        base_logit = _safe_logit(base_probability)
        result = {
            "logits": base_logit + delta,
            "base_logits": base_logit,
            "interaction_delta": delta,
            "quality_gate": quality_gate,
            "interaction_weight": interaction_weight,
        }
        if sequence_aux_logits is not None and structure_aux_logits is not None:
            result["sequence_aux_logits"] = sequence_aux_logits
            result["structure_aux_logits"] = structure_aux_logits
        return result


@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, float]
    logits: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor
    deltas: torch.Tensor


def _run_adapter_epoch(
    model: ProtT5StructureAdapter,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    threshold: float,
    structure_dropout: float,
    delta_l1: float,
    hard_example_weight: float,
    auxiliary_weight: float = 0.0,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    logits_all = []
    labels_all = []
    indices_all = []
    deltas_all = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            available = (
                torch.rand(batch["labels"].shape[0], device=device) >= structure_dropout
            ).float()
        else:
            available = torch.ones_like(batch["labels"])
        output = model(
            tokens=batch["tokens"],
            spatial=batch["spatial"],
            plddt=batch["plddt"],
            mask=batch["mask"],
            base_probability=batch["base_probability"],
            structure_available=available,
            chemical=batch.get("chemical"),
            geometry_ssl=batch.get("geometry_ssl"),
        )
        per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
            output["logits"], batch["labels"], reduction="none"
        )
        if training and hard_example_weight > 0:
            base_true_probability = torch.where(
                batch["labels"] > 0.5,
                batch["base_probability"],
                1.0 - batch["base_probability"],
            )
            weights = 1.0 + hard_example_weight * (1.0 - base_true_probability).detach()
            per_sample = per_sample * weights
        loss = per_sample.mean()
        if training and delta_l1 > 0:
            loss = loss + delta_l1 * output["interaction_delta"].abs().mean()
        if training and auxiliary_weight > 0 and "sequence_aux_logits" in output:
            sequence_aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["sequence_aux_logits"], batch["labels"]
            )
            structure_aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["structure_aux_logits"], batch["labels"]
            )
            loss = loss + auxiliary_weight * (sequence_aux_loss + structure_aux_loss)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count = int(batch["labels"].numel())
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
        logits_all.append(output["logits"].detach().cpu())
        labels_all.append(batch["labels"].detach().cpu())
        indices_all.append(batch["source_index"].detach().cpu())
        deltas_all.append(output["interaction_delta"].detach().cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return EpochResult(
        loss=total_loss / max(total_count, 1),
        metrics=binary_metrics(logits, labels, threshold=threshold),
        logits=logits,
        labels=labels,
        indices=torch.cat(indices_all),
        deltas=torch.cat(deltas_all),
    )


def train_oof_adapter(
    *,
    feature_cache: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 5e-4,
    patience: int = 7,
    threshold: float = 0.60,
    structure_dropout: float = 0.25,
    delta_l1: float = 0.02,
    hard_example_weight: float = 0.0,
    seed: int = 42,
    use_chemical_sites: bool = False,
    use_geometry_ssl: bool = False,
    hierarchical_fusion: bool = False,
    auxiliary_weight: float = 0.0,
) -> dict[str, Any]:
    payload = torch.load(feature_cache, map_location="cpu", weights_only=False)
    dataset = CoupledFeatureDataset(payload)
    device = resolve_device(device_name)
    folds = payload["folds"].long()
    num_folds = int(folds.max().item()) + 1
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    oof_logits = torch.empty(len(dataset), dtype=torch.float32)
    oof_deltas = torch.empty(len(dataset), dtype=torch.float32)
    fold_reports = []

    for fold in range(num_folds):
        set_seed(seed + fold)
        train_indices = torch.where(folds != fold)[0].tolist()
        validation_indices = torch.where(folds == fold)[0].tolist()
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_coupled,
        )
        validation_loader = DataLoader(
            Subset(dataset, validation_indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_coupled,
        )
        model_config = {
            "token_dim": int(payload["metadata"]["token_dim"]),
            "spatial_dim": int(payload["metadata"]["spatial_dim"]),
            "chemical_dim": (
                int(payload["metadata"].get("chemical_dim", 0))
                if use_chemical_sites
                else 0
            ),
            "geometry_ssl_dim": (
                int(payload["metadata"].get("geometry_ssl_dim", 0))
                if use_geometry_ssl
                else 0
            ),
            "hierarchical_fusion": hierarchical_fusion,
        }
        model = ProtT5StructureAdapter(**model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_score = -float("inf")
        best_auprc = -float("inf")
        best_epoch = 0
        stale = 0
        history = []
        for epoch in range(1, epochs + 1):
            train_result = _run_adapter_epoch(
                model,
                train_loader,
                device,
                optimizer,
                threshold=threshold,
                structure_dropout=structure_dropout,
                delta_l1=delta_l1,
                hard_example_weight=hard_example_weight,
                auxiliary_weight=auxiliary_weight,
            )
            validation_result = _run_adapter_epoch(
                model,
                validation_loader,
                device,
                None,
                threshold=threshold,
                structure_dropout=0.0,
                delta_l1=0.0,
                hard_example_weight=0.0,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_result.loss,
                    "validation_loss": validation_result.loss,
                    "train": train_result.metrics,
                    "validation": validation_result.metrics,
                }
            )
            score = float(validation_result.metrics["mcc"])
            auprc = float(validation_result.metrics["auprc"])
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and auprc > best_auprc
            ):
                best_score = score
                best_auprc = auprc
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError(f"Fold {fold} did not produce a checkpoint.")
        model.load_state_dict(best_state)
        final_result = _run_adapter_epoch(
            model,
            validation_loader,
            device,
            None,
            threshold=threshold,
            structure_dropout=0.0,
            delta_l1=0.0,
            hard_example_weight=0.0,
        )
        for index, logit, delta in zip(
            final_result.indices.tolist(),
            final_result.logits.tolist(),
            final_result.deltas.tolist(),
        ):
            oof_logits[index] = float(logit)
            oof_deltas[index] = float(delta)
        fold_dir = output_root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state,
                "model_config": model_config,
                "epoch": best_epoch,
                "feature_metadata": payload["metadata"],
                "adapter_config": {
                    "threshold": threshold,
                    "structure_dropout": structure_dropout,
                    "delta_l1": delta_l1,
                    "hard_example_weight": hard_example_weight,
                    "seed": seed + fold,
                    "use_chemical_sites": use_chemical_sites,
                    "use_geometry_ssl": use_geometry_ssl,
                    "hierarchical_fusion": hierarchical_fusion,
                    "auxiliary_weight": auxiliary_weight,
                },
                "validation_metrics": final_result.metrics,
            },
            fold_dir / "best_model.pt",
        )
        (fold_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        fold_reports.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "adapter": final_result.metrics,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    labels = payload["labels"].float()
    base_logits = _safe_logit(payload["base_probabilities"])
    base_metrics = binary_metrics(base_logits, labels, threshold=threshold)
    adapter_metrics = binary_metrics(oof_logits, labels, threshold=threshold)
    fold_wins = 0
    for report in fold_reports:
        fold = int(report["fold"])
        fold_mask = folds == fold
        base_fold = binary_metrics(
            base_logits[fold_mask], labels[fold_mask], threshold=threshold
        )
        report["baseline"] = base_fold
        report["mcc_delta"] = float(report["adapter"]["mcc"] - base_fold["mcc"])
        if report["mcc_delta"] > 0:
            fold_wins += 1

    rows = []
    probabilities = torch.sigmoid(oof_logits)
    for index in range(len(dataset)):
        rows.append(
            {
                "source_index": index,
                "fold": int(folds[index]),
                "sample_id": payload["sample_ids"][index],
                "sequence": payload["sequences"][index],
                "label": int(labels[index]),
                "baseline_probability": f"{float(payload['base_probabilities'][index]):.10g}",
                "interaction_delta": f"{float(oof_deltas[index]):.10g}",
                "toxicity_probability": f"{float(probabilities[index]):.10g}",
                "prediction": int(probabilities[index] >= threshold),
            }
        )
    with (output_root / "oof_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    retention = {
        "mcc_above_default": adapter_metrics["mcc"] > base_metrics["mcc"],
        "auprc_not_lower": adapter_metrics["auprc"] >= base_metrics["auprc"],
        "brier_not_higher": adapter_metrics["brier"] <= base_metrics["brier"],
        "mcc_fold_wins_at_least_3": fold_wins >= 3,
    }
    retention["passed"] = all(retention.values())
    summary = {
        "protocol": {
            "feature_cache": str(feature_cache),
            "num_samples": len(dataset),
            "num_folds": num_folds,
            "architecture": (
                "ProtT5-query, local functional-site semantic gate, pLDDT-gated 3D key/value, "
                "bounded residual over immutable default"
                if use_chemical_sites
                else "ProtT5-query, pLDDT-gated 3D key/value, bounded residual over immutable default"
            ),
            "selection": "fold validation MCC at the frozen default threshold",
            "test_labels_used": False,
            "threshold": threshold,
            "structure_dropout": structure_dropout,
            "delta_l1": delta_l1,
            "hard_example_weight": hard_example_weight,
            "use_chemical_sites": use_chemical_sites,
            "use_geometry_ssl": use_geometry_ssl,
            "hierarchical_fusion": hierarchical_fusion,
            "auxiliary_weight": auxiliary_weight,
        },
        "baseline": base_metrics,
        "adapter": adapter_metrics,
        "delta": {
            key: float(adapter_metrics[key] - base_metrics[key])
            for key in ("balanced_accuracy", "mcc", "auroc", "auprc", "brier")
        },
        "fold_reports": fold_reports,
        "fold_mcc_wins": fold_wins,
        "interaction_delta": {
            "mean": float(oof_deltas.mean()),
            "mean_absolute": float(oof_deltas.abs().mean()),
            "max_absolute": float(oof_deltas.abs().max()),
        },
        "retention_gate": retention,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def train_full_adapter(
    *,
    feature_cache: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
    epochs: int = 6,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 5e-4,
    threshold: float = 0.60,
    structure_dropout: float = 0.25,
    delta_l1: float = 0.02,
    hard_example_weight: float = 0.0,
    seed: int = 42,
    use_chemical_sites: bool = True,
    use_geometry_ssl: bool = False,
    hierarchical_fusion: bool = False,
    auxiliary_weight: float = 0.0,
) -> dict[str, Any]:
    """Refit the selected adapter for a fixed OOF-derived epoch count."""

    if epochs <= 0:
        raise ValueError("Full-data epochs must be positive.")
    payload = torch.load(feature_cache, map_location="cpu", weights_only=False)
    dataset = CoupledFeatureDataset(payload)
    device = resolve_device(device_name)
    set_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_coupled,
    )
    model_config = {
        "token_dim": int(payload["metadata"]["token_dim"]),
        "spatial_dim": int(payload["metadata"]["spatial_dim"]),
        "chemical_dim": (
            int(payload["metadata"].get("chemical_dim", 0))
            if use_chemical_sites
            else 0
        ),
        "geometry_ssl_dim": (
            int(payload["metadata"].get("geometry_ssl_dim", 0))
            if use_geometry_ssl
            else 0
        ),
        "hierarchical_fusion": hierarchical_fusion,
        "hidden_dim": 128,
        "num_heads": 4,
        "dropout": 0.2,
        "initial_interaction_weight": 0.10,
        "max_logit_delta": 1.25,
    }
    model = ProtT5StructureAdapter(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    for epoch in range(1, epochs + 1):
        result = _run_adapter_epoch(
            model,
            loader,
            device,
            optimizer,
            threshold=threshold,
            structure_dropout=structure_dropout,
            delta_l1=delta_l1,
            hard_example_weight=hard_example_weight,
            auxiliary_weight=auxiliary_weight,
        )
        history.append({"epoch": epoch, "loss": result.loss, "metrics": result.metrics})
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "epoch": epochs,
        "model_config": model_config,
        "feature_metadata": payload["metadata"],
        "adapter_config": {
            "threshold": threshold,
            "structure_dropout": structure_dropout,
            "delta_l1": delta_l1,
            "hard_example_weight": hard_example_weight,
            "seed": seed,
            "use_chemical_sites": use_chemical_sites,
            "use_geometry_ssl": use_geometry_ssl,
            "hierarchical_fusion": hierarchical_fusion,
            "auxiliary_weight": auxiliary_weight,
        },
        "refit_protocol": {
            "full_training_data": True,
            "training_features": "strict OOF base probabilities and structure states",
            "fixed_epochs": epochs,
            "epoch_source": "median best epoch from five group-aware OOF folds",
        },
    }
    torch.save(checkpoint, output_root / "best_model.pt")
    summary = {
        "checkpoint": str(output_root / "best_model.pt"),
        "epochs": epochs,
        "model_config": model_config,
        "adapter_config": checkpoint["adapter_config"],
        "final_training_metrics": history[-1]["metrics"],
        "refit_protocol": checkpoint["refit_protocol"],
    }
    (output_root / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def predict_adapter(
    *,
    feature_cache: str | Path,
    checkpoint: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
    device_name: str = "cuda",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Apply a frozen full-data adapter without threshold or weight selection."""

    payload = torch.load(feature_cache, map_location="cpu", weights_only=False)
    dataset = CoupledFeatureDataset(payload)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ProtT5StructureAdapter(**checkpoint_payload["model_config"])
    model.load_state_dict(checkpoint_payload["model_state"])
    device = resolve_device(device_name)
    model.eval().to(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_coupled,
    )
    threshold = float(checkpoint_payload["adapter_config"]["threshold"])
    result = _run_adapter_epoch(
        model,
        loader,
        device,
        None,
        threshold=threshold,
        structure_dropout=0.0,
        delta_l1=0.0,
        hard_example_weight=0.0,
    )
    order = torch.argsort(result.indices)
    logits = result.logits[order]
    deltas = result.deltas[order]
    probabilities = torch.sigmoid(logits)
    labels = payload["labels"].float()
    base_metrics = binary_metrics(
        _safe_logit(payload["base_probabilities"]), labels, threshold=threshold
    )
    adapter_metrics = binary_metrics(logits, labels, threshold=threshold)
    rows = []
    for index in range(len(dataset)):
        rows.append(
            {
                "source_index": index,
                "sample_id": payload["sample_ids"][index],
                "sequence": payload["sequences"][index],
                "label": int(labels[index]),
                "baseline_probability": f"{float(payload['base_probabilities'][index]):.10g}",
                "interaction_delta": f"{float(deltas[index]):.10g}",
                "toxicity_probability": f"{float(probabilities[index]):.10g}",
                "prediction": int(probabilities[index] >= threshold),
                "decision_threshold": threshold,
            }
        )
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": {
            "feature_cache": str(feature_cache),
            "checkpoint": str(checkpoint),
            "threshold": threshold,
            "test_labels_used_for_selection": False,
        },
        "baseline": base_metrics,
        "adapter": adapter_metrics,
        "delta": {
            key: float(adapter_metrics[key] - base_metrics[key])
            for key in ("balanced_accuracy", "mcc", "auroc", "auprc", "brier")
        },
        "interaction_delta": {
            "mean": float(deltas.mean()),
            "mean_absolute": float(deltas.abs().mean()),
            "max_absolute": float(deltas.abs().max()),
        },
    }
    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    precompute = subparsers.add_parser("precompute")
    precompute.add_argument("--processed", required=True)
    precompute.add_argument("--token-cache", required=True)
    precompute.add_argument("--baseline-predictions", required=True)
    precompute.add_argument("--fold-manifest", required=True)
    precompute.add_argument("--checkpoint-pattern", required=True)
    precompute.add_argument("--output", required=True)
    precompute.add_argument("--device", default="cuda")
    precompute.add_argument("--batch-size", type=int, default=64)

    inference = subparsers.add_parser("precompute-inference")
    inference.add_argument("--processed", required=True)
    inference.add_argument("--token-cache", required=True)
    inference.add_argument("--baseline-predictions", required=True)
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--device", default="cuda")
    inference.add_argument("--batch-size", type=int, default=64)

    attach = subparsers.add_parser("attach-chemical")
    attach.add_argument("--feature-cache", required=True)
    attach.add_argument("--processed", required=True)

    train = subparsers.add_parser("train-oof")
    train.add_argument("--feature-cache", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=5e-4)
    train.add_argument("--patience", type=int, default=7)
    train.add_argument("--threshold", type=float, default=0.60)
    train.add_argument("--structure-dropout", type=float, default=0.25)
    train.add_argument("--delta-l1", type=float, default=0.02)
    train.add_argument("--hard-example-weight", type=float, default=0.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--use-chemical-sites", action="store_true")
    train.add_argument("--use-geometry-ssl", action="store_true")
    train.add_argument("--hierarchical-fusion", action="store_true")
    train.add_argument("--auxiliary-weight", type=float, default=0.0)

    refit = subparsers.add_parser("train-full")
    refit.add_argument("--feature-cache", required=True)
    refit.add_argument("--output-dir", required=True)
    refit.add_argument("--device", default="cuda")
    refit.add_argument("--epochs", type=int, default=6)
    refit.add_argument("--batch-size", type=int, default=64)
    refit.add_argument("--learning-rate", type=float, default=3e-4)
    refit.add_argument("--weight-decay", type=float, default=5e-4)
    refit.add_argument("--threshold", type=float, default=0.60)
    refit.add_argument("--structure-dropout", type=float, default=0.25)
    refit.add_argument("--delta-l1", type=float, default=0.02)
    refit.add_argument("--hard-example-weight", type=float, default=0.0)
    refit.add_argument("--seed", type=int, default=42)
    refit.add_argument("--without-chemical-sites", action="store_true")
    refit.add_argument("--use-geometry-ssl", action="store_true")
    refit.add_argument("--hierarchical-fusion", action="store_true")
    refit.add_argument("--auxiliary-weight", type=float, default=0.0)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--feature-cache", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output-csv", required=True)
    predict.add_argument("--output-json", required=True)
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "precompute":
        result = precompute_oof_features(
            processed=args.processed,
            token_cache=args.token_cache,
            baseline_predictions=args.baseline_predictions,
            fold_manifest=args.fold_manifest,
            checkpoint_pattern=args.checkpoint_pattern,
            output=args.output,
            device_name=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "precompute-inference":
        result = precompute_inference_features(
            processed=args.processed,
            token_cache=args.token_cache,
            baseline_predictions=args.baseline_predictions,
            checkpoint=args.checkpoint,
            output=args.output,
            device_name=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "attach-chemical":
        result = attach_chemical_features(args.feature_cache, args.processed)
    elif args.command == "train-oof":
        result = train_oof_adapter(
            feature_cache=args.feature_cache,
            output_dir=args.output_dir,
            device_name=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            threshold=args.threshold,
            structure_dropout=args.structure_dropout,
            delta_l1=args.delta_l1,
            hard_example_weight=args.hard_example_weight,
            seed=args.seed,
            use_chemical_sites=args.use_chemical_sites,
            use_geometry_ssl=args.use_geometry_ssl,
            hierarchical_fusion=args.hierarchical_fusion,
            auxiliary_weight=args.auxiliary_weight,
        )
    elif args.command == "train-full":
        result = train_full_adapter(
            feature_cache=args.feature_cache,
            output_dir=args.output_dir,
            device_name=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            threshold=args.threshold,
            structure_dropout=args.structure_dropout,
            delta_l1=args.delta_l1,
            hard_example_weight=args.hard_example_weight,
            seed=args.seed,
            use_chemical_sites=not args.without_chemical_sites,
            use_geometry_ssl=args.use_geometry_ssl,
            hierarchical_fusion=args.hierarchical_fusion,
            auxiliary_weight=args.auxiliary_weight,
        )
    else:
        result = predict_adapter(
            feature_cache=args.feature_cache,
            checkpoint=args.checkpoint,
            output_csv=args.output_csv,
            output_json=args.output_json,
            device_name=args.device,
            batch_size=args.batch_size,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
