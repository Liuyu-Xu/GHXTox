"""Fold-pure self-supervised pretraining for a sequence-free 3D encoder.

The encoder only sees residue physicochemical features, structure descriptors,
C-alpha coordinates and pLDDT.  It is trained without toxicity labels to
reconstruct masked residue features, pair distances and 6/8/10 Angstrom
contacts.  OOF embeddings can then be attached to the existing residue-aligned
ProtT5/3D adapter cache without exposing a held-out fold to its toxicity labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ghxtox.data import PeptideTensorDataset, collate_peptides
from ghxtox.features import RESIDUE_FEATURE_DIM
from ghxtox.geometry_features import structure_feature_matrix
from ghxtox.models.layers import SpatialBranch
from ghxtox.utils import move_batch_to_device, resolve_device, set_seed


class IndexedDataset(Dataset):
    def __init__(self, dataset: PeptideTensorDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[int, dict[str, Any]]:
        source_index = self.indices[index]
        return source_index, self.dataset[source_index]


def collate_indexed(
    batch: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    indices, records = zip(*batch)
    result = collate_peptides(
        list(records),
        include_structure=True,
        include_atom=False,
    )
    result["source_index"] = torch.tensor(indices, dtype=torch.long)
    return result


def collate_cross_view(
    batch: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    """Collate an ESMFold anchor and one randomly sampled ensemble conformer."""

    result = collate_indexed(batch)
    conformer_mask = result.get("conformer_graph_conformer_mask")
    if conformer_mask is None:
        raise ValueError("Cross-view pretraining requires per-conformer graph fields.")
    batch_size, max_length = result["mask"].shape
    view_coords = torch.zeros(batch_size, max_length, 3, dtype=torch.float32)
    view_plddt = torch.zeros(batch_size, max_length, dtype=torch.float32)
    view_structure = torch.zeros(
        batch_size,
        max_length,
        int(result["structure_features"].shape[-1]),
        dtype=torch.float32,
    )
    selected = torch.full((batch_size,), -1, dtype=torch.long)
    for row in range(batch_size):
        valid = torch.where(conformer_mask[row])[0]
        if valid.numel() == 0:
            raise ValueError("Cross-view batches may only contain samples with conformers.")
        choice = valid[torch.randint(valid.numel(), (1,)).item()]
        selected[row] = choice
        length = int(result["lengths"][row])
        coords = result["conformer_graph_residue_coords"][row, choice, :length].float()
        plddt = result["conformer_graph_plddt"][row, choice, :length].float()
        view_coords[row, :length] = coords
        view_plddt[row, :length] = plddt
        view_structure[row, :length] = structure_feature_matrix(coords, plddt)
    result["view_coords"] = view_coords
    result["view_plddt"] = view_plddt
    result["view_structure_features"] = view_structure
    result["selected_conformer"] = selected
    return result


def load_fold_assignments(path: str | Path, count: int) -> torch.Tensor:
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


class GeometrySSLModel(nn.Module):
    """Sequence-free EGNN with masked-node and pair-geometry objectives."""

    def __init__(
        self,
        *,
        structure_feature_dim: int = 16,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.structure_feature_dim = int(structure_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = SpatialBranch(
            residue_feature_dim=RESIDUE_FEATURE_DIM,
            structure_feature_dim=self.structure_feature_dim,
            plm_embedding_dim=0,
            hidden_dim=self.hidden_dim,
            num_layers=int(num_layers),
            rbf_bins=12,
            rbf_max_distance=24.0,
            dropout=float(dropout),
            graph_mode="hybrid",
            spatial_top_k=10,
            plddt_edge_min=0.10,
            plddt_edge_power=1.5,
        )
        self.node_decoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, RESIDUE_FEATURE_DIM),
        )
        self.pair_decoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 4),
        )

    def encode(
        self,
        *,
        residue_features: torch.Tensor,
        structure_features: torch.Tensor,
        coords: torch.Tensor,
        plddt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(
            residue_features,
            structure_features,
            coords,
            plddt,
            mask,
            plm_features=None,
        )

    def forward(
        self,
        *,
        residue_features: torch.Tensor,
        structure_features: torch.Tensor,
        coords: torch.Tensor,
        plddt: torch.Tensor,
        mask: torch.Tensor,
        node_mask_rate: float = 0.15,
    ) -> dict[str, torch.Tensor]:
        mask = mask.bool()
        random_mask = torch.rand(mask.shape, device=mask.device) < float(node_mask_rate)
        node_mask = random_mask & mask
        missing = mask.any(dim=1) & ~node_mask.any(dim=1)
        if missing.any():
            first_valid = mask.float().argmax(dim=1)
            node_mask[missing, first_valid[missing]] = True

        masked_residue = residue_features.clone()
        masked_residue[node_mask] = 0.0
        hidden = self.encode(
            residue_features=masked_residue,
            structure_features=structure_features,
            coords=coords,
            plddt=plddt,
            mask=mask,
        )
        node_prediction = self.node_decoder(hidden)

        left = hidden.unsqueeze(2)
        right = hidden.unsqueeze(1)
        pair_features = torch.cat(
            [
                (left - right).abs(),
                left * right,
            ],
            dim=-1,
        )
        pair_prediction = self.pair_decoder(pair_features)
        return {
            "hidden": hidden,
            "node_prediction": node_prediction,
            "node_mask": node_mask,
            "pair_prediction": pair_prediction,
        }


def geometry_ssl_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"].bool()
    node_mask = output["node_mask"]
    node_error = torch.nn.functional.smooth_l1_loss(
        output["node_prediction"][node_mask],
        batch["residue_features"][node_mask],
    )

    coords = batch["coords"]
    distances = torch.cdist(coords, coords)
    length = mask.shape[1]
    positions = torch.arange(length, device=mask.device)
    pair_mask = (
        mask.unsqueeze(1)
        & mask.unsqueeze(2)
        & (positions.view(1, -1, 1) < positions.view(1, 1, -1))
        & ((positions.view(1, -1, 1) - positions.view(1, 1, -1)).abs() > 1)
    )
    pair_confidence = torch.sqrt(
        (
            batch["plddt"].clamp(0.0, 1.0).unsqueeze(1)
            * batch["plddt"].clamp(0.0, 1.0).unsqueeze(2)
        ).clamp_min(1e-6)
    )
    pair_weight = pair_confidence[pair_mask].clamp_min(0.10)
    pair_prediction = output["pair_prediction"][pair_mask]
    pair_distance = distances[pair_mask]

    distance_target = torch.log1p(pair_distance.clamp_max(24.0)) / torch.log(
        pair_distance.new_tensor(25.0)
    )
    distance_loss_raw = torch.nn.functional.smooth_l1_loss(
        pair_prediction[:, 0],
        distance_target,
        reduction="none",
    )
    distance_loss = (distance_loss_raw * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)

    contact_target = torch.stack(
        [
            (pair_distance <= 6.0).float(),
            (pair_distance <= 8.0).float(),
            (pair_distance <= 10.0).float(),
        ],
        dim=-1,
    )
    contact_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
        pair_prediction[:, 1:],
        contact_target,
        reduction="none",
    ).mean(dim=-1)
    contact_loss = (contact_loss_raw * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)

    loss = 0.25 * node_error + distance_loss + contact_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "node": float(node_error.detach().cpu()),
        "distance": float(distance_loss.detach().cpu()),
        "contact": float(contact_loss.detach().cpu()),
    }


def cross_view_geometry_loss(
    model: GeometrySSLModel,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Learn generator-stable residue states and ensemble contact occupancy."""

    mask = batch["mask"].bool()
    anchor_output = model(
        residue_features=batch["residue_features"],
        structure_features=batch["structure_features"],
        coords=batch["coords"],
        plddt=batch["plddt"],
        mask=mask,
    )
    view_output = model(
        residue_features=batch["residue_features"],
        structure_features=batch["view_structure_features"],
        coords=batch["view_coords"],
        plddt=batch["view_plddt"],
        mask=mask,
    )
    anchor_ssl, anchor_report = geometry_ssl_loss(anchor_output, batch)
    view_batch = {
        **batch,
        "coords": batch["view_coords"],
        "plddt": batch["view_plddt"],
    }
    view_ssl, view_report = geometry_ssl_loss(view_output, view_batch)

    anchor_hidden = torch.nn.functional.normalize(anchor_output["hidden"], dim=-1)
    view_hidden = torch.nn.functional.normalize(view_output["hidden"], dim=-1)
    residue_confidence = torch.sqrt(
        (
            batch["plddt"].clamp(0.0, 1.0)
            * batch["view_plddt"].clamp(0.0, 1.0)
        ).clamp_min(1e-6)
    )
    residue_weight = residue_confidence * mask.to(dtype=residue_confidence.dtype)
    consistency_raw = 1.0 - (anchor_hidden * view_hidden).sum(dim=-1)
    consistency_loss = (
        consistency_raw * residue_weight
    ).sum() / residue_weight.sum().clamp_min(1.0)

    conformer_coords = batch["conformer_graph_residue_coords"].float()
    conformer_mask = batch["conformer_graph_conformer_mask"].bool()
    pair_distances = torch.cdist(conformer_coords, conformer_coords)
    valid_pairs = (
        mask[:, None, :, None]
        & mask[:, None, None, :]
        & conformer_mask[:, :, None, None]
    )
    positions = torch.arange(mask.shape[1], device=mask.device)
    upper_nonlocal = (
        (positions.view(1, 1, -1, 1) < positions.view(1, 1, 1, -1))
        & (
            (
                positions.view(1, 1, -1, 1)
                - positions.view(1, 1, 1, -1)
            ).abs()
            > 1
        )
    )
    valid_pairs = valid_pairs & upper_nonlocal
    conformer_count = conformer_mask.sum(dim=1).clamp_min(1).view(-1, 1, 1)
    occupancy = torch.stack(
        [
            ((pair_distances <= cutoff) & valid_pairs).sum(dim=1)
            / conformer_count
            for cutoff in (6.0, 8.0, 10.0)
        ],
        dim=-1,
    ).float()
    masked_distances = pair_distances.masked_fill(~valid_pairs, float("nan"))
    median_distance = torch.nanmedian(masked_distances, dim=1).values
    pair_mask = valid_pairs.any(dim=1)
    distance_target = (
        torch.log1p(median_distance[pair_mask].clamp_max(24.0))
        / torch.log(median_distance.new_tensor(25.0))
    )
    occupancy_target = occupancy[pair_mask]

    pair_losses = []
    for output in (anchor_output, view_output):
        prediction = output["pair_prediction"][pair_mask]
        distance_loss = torch.nn.functional.smooth_l1_loss(
            prediction[:, 0],
            distance_target,
        )
        contact_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[:, 1:],
            occupancy_target,
        )
        pair_losses.append(distance_loss + contact_loss)
    ensemble_pair_loss = torch.stack(pair_losses).mean()

    pooled_anchor = (
        anchor_output["hidden"] * mask.unsqueeze(-1)
    ).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    pooled_view = (
        view_output["hidden"] * mask.unsqueeze(-1)
    ).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    pooled = torch.cat([pooled_anchor, pooled_view], dim=0)
    variance_loss = torch.relu(
        pooled.new_tensor(0.20) - pooled.std(dim=0, unbiased=False)
    ).mean()

    loss = (
        0.20 * (anchor_ssl + view_ssl)
        + 0.55 * ensemble_pair_loss
        + 0.50 * consistency_loss
        + 0.05 * variance_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "anchor_ssl": float(anchor_ssl.detach().cpu()),
        "view_ssl": float(view_ssl.detach().cpu()),
        "ensemble_pair": float(ensemble_pair_loss.detach().cpu()),
        "consistency": float(consistency_loss.detach().cpu()),
        "variance": float(variance_loss.detach().cpu()),
        "anchor_contact": float(anchor_report["contact"]),
        "view_contact": float(view_report["contact"]),
    }


def train_cross_view_encoder(
    model: GeometrySSLModel,
    loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    history = []
    metric_names = (
        "loss",
        "anchor_ssl",
        "view_ssl",
        "ensemble_pair",
        "consistency",
        "variance",
        "anchor_contact",
        "view_contact",
    )
    for epoch in range(1, int(epochs) + 1):
        model.train()
        totals = {name: 0.0 for name in metric_names}
        count = 0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, report = cross_view_geometry_loss(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            current_batch = int(batch["mask"].shape[0])
            for name in metric_names:
                totals[name] += report[name] * current_batch
            count += current_batch
        epoch_report = {"epoch": epoch}
        epoch_report.update(
            {name: value / max(count, 1) for name, value in totals.items()}
        )
        history.append(epoch_report)
        print(
            f"cross_view_geometry epoch={epoch}/{epochs} "
            f"loss={epoch_report['loss']:.6f} "
            f"consistency={epoch_report['consistency']:.6f} "
            f"ensemble_pair={epoch_report['ensemble_pair']:.6f}",
            flush=True,
        )
    return history


def train_ssl_encoder(
    model: GeometrySSLModel,
    loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        totals = {"loss": 0.0, "node": 0.0, "distance": 0.0, "contact": 0.0}
        count = 0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                residue_features=batch["residue_features"],
                structure_features=batch["structure_features"],
                coords=batch["coords"],
                plddt=batch["plddt"],
                mask=batch["mask"],
            )
            loss, report = geometry_ssl_loss(output, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_size = int(batch["mask"].shape[0])
            for key in totals:
                totals[key] += report[key] * batch_size
            count += batch_size
        epoch_report = {"epoch": epoch}
        epoch_report.update({key: value / max(count, 1) for key, value in totals.items()})
        history.append(epoch_report)
        print(
            f"geometry_ssl epoch={epoch}/{epochs} "
            f"loss={epoch_report['loss']:.6f} "
            f"distance={epoch_report['distance']:.6f} "
            f"contact={epoch_report['contact']:.6f}",
            flush=True,
        )
    return history


@torch.no_grad()
def encode_dataset(
    model: GeometrySSLModel,
    loader: DataLoader,
    device: torch.device,
    destination: list[torch.Tensor | None],
) -> None:
    model.eval().to(device)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        hidden = model.encode(
            residue_features=batch["residue_features"],
            structure_features=batch["structure_features"],
            coords=batch["coords"],
            plddt=batch["plddt"],
            mask=batch["mask"],
        )
        for row, source_index in enumerate(batch["source_index"].tolist()):
            length = int(batch["lengths"][row])
            destination[source_index] = hidden[row, :length].detach().cpu().half()


def _model_config(
    *,
    structure_feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> dict[str, Any]:
    return {
        "structure_feature_dim": int(structure_feature_dim),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
    }


def pretrain_oof(
    *,
    processed: str | Path,
    fold_manifest: str | Path,
    adapter_feature_cache: str | Path,
    output_cache: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
    epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    hidden_dim: int = 96,
    num_layers: int = 2,
    dropout: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    dataset = PeptideTensorDataset(processed, require_labels=True)
    folds = load_fold_assignments(fold_manifest, len(dataset))
    num_folds = int(folds.max().item()) + 1
    device = resolve_device(device_name)
    structure_feature_dim = int(dataset.records[0]["structure_features"].shape[1])
    config = _model_config(
        structure_feature_dim=structure_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    embeddings: list[torch.Tensor | None] = [None] * len(dataset)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    fold_reports = []

    for fold in range(num_folds):
        print(f"geometry_ssl fold={fold + 1}/{num_folds}", flush=True)
        set_seed(int(seed) + fold)
        train_indices = torch.where(folds != fold)[0].tolist()
        validation_indices = torch.where(folds == fold)[0].tolist()
        train_loader = DataLoader(
            IndexedDataset(dataset, train_indices),
            batch_size=int(batch_size),
            shuffle=True,
            collate_fn=collate_indexed,
        )
        validation_loader = DataLoader(
            IndexedDataset(dataset, validation_indices),
            batch_size=int(batch_size),
            shuffle=False,
            collate_fn=collate_indexed,
        )
        model = GeometrySSLModel(**config)
        history = train_ssl_encoder(
            model,
            train_loader,
            device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        encode_dataset(model, validation_loader, device, embeddings)
        checkpoint = {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_config": config,
            "fold": fold,
            "epochs": int(epochs),
            "toxicity_labels_used": False,
        }
        fold_root = output_root / f"fold{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, fold_root / "geometry_ssl.pt")
        (fold_root / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        fold_reports.append(
            {
                "fold": fold,
                "num_train": len(train_indices),
                "num_validation": len(validation_indices),
                "final": history[-1],
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if any(value is None for value in embeddings):
        raise RuntimeError("OOF geometry pretraining did not encode every sample.")
    adapter_payload = torch.load(
        adapter_feature_cache,
        map_location="cpu",
        weights_only=False,
    )
    if adapter_payload["sequences"] != [record["sequence"] for record in dataset.records]:
        raise ValueError("Adapter cache and structure data sequences are not aligned.")
    offsets = adapter_payload["offsets"].tolist()
    expected_offsets = [0]
    for value in embeddings:
        expected_offsets.append(expected_offsets[-1] + int(value.shape[0]))
    if offsets != expected_offsets:
        raise ValueError("Geometry embeddings and adapter residue offsets are not aligned.")
    adapter_payload["geometry_ssl_embeddings"] = torch.cat(
        [value for value in embeddings if value is not None],
        dim=0,
    )
    adapter_payload["metadata"]["geometry_ssl_dim"] = int(hidden_dim)
    adapter_payload["metadata"]["geometry_ssl_protocol"] = (
        "fold-pure sequence-free masked-node, pair-distance and 6/8/10A contact pretraining"
    )
    output_cache_path = Path(output_cache)
    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_cache_path.with_suffix(output_cache_path.suffix + ".tmp")
    torch.save(adapter_payload, temporary)
    temporary.replace(output_cache_path)
    summary = {
        "protocol": {
            "processed": str(processed),
            "fold_manifest": str(fold_manifest),
            "adapter_feature_cache": str(adapter_feature_cache),
            "output_cache": str(output_cache),
            "toxicity_labels_used": False,
            "fold_pure": True,
            "epochs": int(epochs),
            "model_config": config,
        },
        "num_samples": len(dataset),
        "num_residues": int(adapter_payload["geometry_ssl_embeddings"].shape[0]),
        "fold_reports": fold_reports,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def pretrain_full(
    *,
    processed: str | Path,
    output_dir: str | Path,
    device_name: str = "cuda",
    epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    hidden_dim: int = 96,
    num_layers: int = 2,
    dropout: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit the selected label-free encoder on every training structure."""

    dataset = PeptideTensorDataset(processed, require_labels=False)
    device = resolve_device(device_name)
    set_seed(int(seed))
    structure_feature_dim = int(dataset.records[0]["structure_features"].shape[1])
    config = _model_config(
        structure_feature_dim=structure_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    loader = DataLoader(
        IndexedDataset(dataset, list(range(len(dataset)))),
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collate_indexed,
    )
    model = GeometrySSLModel(**config)
    history = train_ssl_encoder(
        model,
        loader,
        device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "geometry_ssl.pt"
    torch.save(
        {
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "model_config": config,
            "epochs": int(epochs),
            "toxicity_labels_used": False,
            "training_scope": "all training structures after OOF architecture selection",
        },
        checkpoint_path,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "processed": str(processed),
        "num_samples": len(dataset),
        "toxicity_labels_used": False,
        "epochs": int(epochs),
        "model_config": config,
        "final": history[-1],
    }
    (output_root / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def pretrain_cross_view_oof(
    *,
    processed: str | Path,
    fold_manifest: str | Path,
    adapter_feature_cache: str | Path,
    output_cache: str | Path,
    output_dir: str | Path,
    initialization_pattern: str,
    device_name: str = "cuda",
    epochs: int = 3,
    batch_size: int = 24,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    """Fold-pure cross-generator refinement of the existing geometry encoder."""

    dataset = PeptideTensorDataset(processed, require_labels=True)
    folds = load_fold_assignments(fold_manifest, len(dataset))
    num_folds = int(folds.max().item()) + 1
    device = resolve_device(device_name)
    available_indices = [
        index
        for index, record in enumerate(dataset.records)
        if bool(record.get("conformer_available", False))
        and torch.is_tensor(record.get("conformer_graph_conformer_mask"))
        and bool(record["conformer_graph_conformer_mask"].any())
    ]
    if not available_indices:
        raise ValueError("No multi-conformer records are available for cross-view training.")
    embeddings: list[torch.Tensor | None] = [None] * len(dataset)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    fold_reports = []

    for fold in range(num_folds):
        print(f"cross_view_geometry fold={fold + 1}/{num_folds}", flush=True)
        set_seed(int(seed) + fold)
        train_indices = [
            index for index in available_indices if int(folds[index]) != fold
        ]
        validation_indices = torch.where(folds == fold)[0].tolist()
        sources = [
            str(dataset.records[index].get("conformer_source", "other"))
            for index in train_indices
        ]
        source_counts: dict[str, int] = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        weights = [
            1.0 / max(source_counts[source], 1)
            for source in sources
        ]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_indices),
            replacement=True,
        )
        train_loader = DataLoader(
            IndexedDataset(dataset, train_indices),
            batch_size=int(batch_size),
            sampler=sampler,
            collate_fn=collate_cross_view,
        )
        validation_loader = DataLoader(
            IndexedDataset(dataset, validation_indices),
            batch_size=int(batch_size),
            shuffle=False,
            collate_fn=collate_indexed,
        )
        initialization = torch.load(
            initialization_pattern.format(fold=fold),
            map_location="cpu",
            weights_only=False,
        )
        model = GeometrySSLModel(**initialization["model_config"])
        model.load_state_dict(initialization["model_state"])
        history = train_cross_view_encoder(
            model,
            train_loader,
            device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        encode_dataset(model, validation_loader, device, embeddings)
        checkpoint = {
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "model_config": initialization["model_config"],
            "fold": fold,
            "epochs": int(epochs),
            "toxicity_labels_used": False,
            "initialization": initialization_pattern.format(fold=fold),
            "protocol": "ESMFold-to-PepFlow/AF2 paired consistency and ensemble occupancy",
        }
        fold_root = output_root / f"fold{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, fold_root / "geometry_cross_view.pt")
        (fold_root / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        fold_reports.append(
            {
                "fold": fold,
                "num_train_multiview": len(train_indices),
                "num_validation": len(validation_indices),
                "source_counts": source_counts,
                "final": history[-1],
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if any(value is None for value in embeddings):
        raise RuntimeError("Cross-view OOF encoding did not cover every sample.")
    adapter_payload = torch.load(
        adapter_feature_cache,
        map_location="cpu",
        weights_only=False,
    )
    sequences = [str(record["sequence"]) for record in dataset.records]
    if adapter_payload["sequences"] != sequences:
        raise ValueError("Adapter cache and cross-view structure data are not aligned.")
    expected_offsets = [0]
    for value in embeddings:
        expected_offsets.append(expected_offsets[-1] + int(value.shape[0]))
    if adapter_payload["offsets"].tolist() != expected_offsets:
        raise ValueError("Cross-view embeddings and adapter offsets are not aligned.")
    adapter_payload["geometry_ssl_embeddings"] = torch.cat(
        [value for value in embeddings if value is not None],
        dim=0,
    )
    hidden_dim = int(adapter_payload["geometry_ssl_embeddings"].shape[-1])
    adapter_payload["metadata"]["geometry_ssl_dim"] = hidden_dim
    adapter_payload["metadata"]["geometry_ssl_protocol"] = (
        "fold-pure ESMFold-to-PepFlow/AF2 paired consistency, "
        "ensemble median-distance/contact-occupancy refinement"
    )
    output_path = Path(output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(adapter_payload, temporary)
    temporary.replace(output_path)
    summary = {
        "protocol": {
            "processed": str(processed),
            "fold_manifest": str(fold_manifest),
            "adapter_feature_cache": str(adapter_feature_cache),
            "output_cache": str(output_cache),
            "initialization_pattern": initialization_pattern,
            "toxicity_labels_used": False,
            "fold_pure": True,
            "source_balanced_sampling": True,
            "epochs": int(epochs),
        },
        "num_samples": len(dataset),
        "num_multiview_samples": len(available_indices),
        "num_residues": int(adapter_payload["geometry_ssl_embeddings"].shape[0]),
        "fold_reports": fold_reports,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def attach_inference_embeddings(
    *,
    processed: str | Path,
    adapter_feature_cache: str | Path,
    checkpoint: str | Path,
    output_cache: str | Path,
    device_name: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Attach embeddings from the frozen full-data SSL encoder to a cache."""

    dataset = PeptideTensorDataset(processed, require_labels=False)
    payload = torch.load(
        adapter_feature_cache,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = GeometrySSLModel(**checkpoint_payload["model_config"])
    model.load_state_dict(checkpoint_payload["model_state"])
    device = resolve_device(device_name)
    loader = DataLoader(
        IndexedDataset(dataset, list(range(len(dataset)))),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_indexed,
    )
    embeddings: list[torch.Tensor | None] = [None] * len(dataset)
    encode_dataset(model, loader, device, embeddings)
    if payload["sequences"] != [record["sequence"] for record in dataset.records]:
        raise ValueError("Adapter cache and structure data sequences are not aligned.")
    offsets = payload["offsets"].tolist()
    expected_offsets = [0]
    for value in embeddings:
        if value is None:
            raise RuntimeError("Frozen geometry encoder did not encode every sample.")
        expected_offsets.append(expected_offsets[-1] + int(value.shape[0]))
    if offsets != expected_offsets:
        raise ValueError("Geometry embeddings and adapter residue offsets are not aligned.")
    payload["geometry_ssl_embeddings"] = torch.cat(
        [value for value in embeddings if value is not None],
        dim=0,
    )
    payload["metadata"]["geometry_ssl_dim"] = int(
        checkpoint_payload["model_config"]["hidden_dim"]
    )
    payload["metadata"]["geometry_ssl_protocol"] = (
        "frozen full-data sequence-free geometry SSL encoder selected by training-only OOF"
    )
    payload["metadata"]["geometry_ssl_checkpoint"] = str(checkpoint)
    output_path = Path(output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return {
        "processed": str(processed),
        "adapter_feature_cache": str(adapter_feature_cache),
        "output_cache": str(output_cache),
        "checkpoint": str(checkpoint),
        "num_samples": len(dataset),
        "num_residues": int(payload["geometry_ssl_embeddings"].shape[0]),
        "geometry_ssl_dim": int(payload["metadata"]["geometry_ssl_dim"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("oof", "full", "encode", "cross-view-oof"),
        default="oof",
    )
    parser.add_argument("--processed", required=True)
    parser.add_argument("--fold-manifest")
    parser.add_argument("--adapter-feature-cache")
    parser.add_argument("--output-cache")
    parser.add_argument("--output-dir", default="runs/geometry_ssl")
    parser.add_argument("--checkpoint")
    parser.add_argument("--initialization-pattern")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "oof":
        if not all(
            (args.fold_manifest, args.adapter_feature_cache, args.output_cache)
        ):
            raise ValueError(
                "OOF mode requires --fold-manifest, --adapter-feature-cache and --output-cache."
            )
        summary = pretrain_oof(
            processed=args.processed,
            fold_manifest=args.fold_manifest,
            adapter_feature_cache=args.adapter_feature_cache,
            output_cache=args.output_cache,
            output_dir=args.output_dir,
            device_name=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seed=args.seed,
        )
    elif args.mode == "cross-view-oof":
        if not all(
            (
                args.fold_manifest,
                args.adapter_feature_cache,
                args.output_cache,
                args.initialization_pattern,
            )
        ):
            raise ValueError(
                "Cross-view OOF mode requires --fold-manifest, "
                "--adapter-feature-cache, --output-cache and --initialization-pattern."
            )
        summary = pretrain_cross_view_oof(
            processed=args.processed,
            fold_manifest=args.fold_manifest,
            adapter_feature_cache=args.adapter_feature_cache,
            output_cache=args.output_cache,
            output_dir=args.output_dir,
            initialization_pattern=args.initialization_pattern,
            device_name=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
    elif args.mode == "full":
        summary = pretrain_full(
            processed=args.processed,
            output_dir=args.output_dir,
            device_name=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seed=args.seed,
        )
    else:
        if not all(
            (args.adapter_feature_cache, args.output_cache, args.checkpoint)
        ):
            raise ValueError(
                "Encode mode requires --adapter-feature-cache, --output-cache and --checkpoint."
            )
        summary = attach_inference_embeddings(
            processed=args.processed,
            adapter_feature_cache=args.adapter_feature_cache,
            checkpoint=args.checkpoint,
            output_cache=args.output_cache,
            device_name=args.device,
            batch_size=args.batch_size,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
