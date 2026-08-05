"""Existing-data multi-conformer SSL and matched-source invariant 3D residual.

The protocol is deliberately fold-pure: every outer-fold encoder sees only
structures from the other four folds and never receives toxicity labels.  A
small bounded residue head is then trained on the same four folds while the
Phase 25 probability remains the immutable base prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ghxtox.bootstrap_ci import _metrics
from ghxtox.features import RESIDUE_FEATURE_DIM, residue_feature_matrix
from ghxtox.geometry_features import STRUCTURE_FEATURE_DIM, structure_feature_matrix
from ghxtox.geometry_pretrain import GeometrySSLModel, cross_view_geometry_loss
from ghxtox.multiconformer_graph_3d import (
    ConformerSample,
    ConformerView,
    build_multiconformer_samples,
)
from ghxtox.oof_source_diagnostics import _read_csv
from ghxtox.utils import save_json


DEFAULT_PROTOCOL = "runs/ensemble_domain_ssl_3d/protocol.json"


@dataclass(frozen=True)
class EnsembleTensor:
    residue_features: torch.Tensor
    coords: torch.Tensor
    structure_features: torch.Tensor


@dataclass(frozen=True)
class EncodedSample:
    residue_embedding: torch.Tensor
    probe_embedding: np.ndarray


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_candidate_oof":
        raise ValueError("Phase 30 protocol must be frozen before execution.")
    if bool(protocol.get("external_test_labels_read", True)):
        raise ValueError("Phase 30 protocol must keep external labels sealed.")
    for input_path, expected in protocol["inputs"]["sha256"].items():
        actual = _sha256(input_path)
        if actual != expected:
            raise ValueError(
                f"Frozen input hash mismatch for {input_path}: {actual} != {expected}."
            )
    return protocol


def _selected_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


@lru_cache(maxsize=4096)
def _load_ensemble(
    summary_path: str,
    sequence: str,
    max_conformers: int,
) -> EnsembleTensor:
    with np.load(summary_path) as payload:
        raw_coords = payload["aligned_coords"].astype(np.float32)
    if raw_coords.ndim != 3 or raw_coords.shape[1:] != (len(sequence), 3):
        raise ValueError(f"Invalid aligned coordinates in {summary_path}.")
    if raw_coords.shape[0] == 0:
        raise ValueError(f"No conformers in {summary_path}.")
    coords = torch.from_numpy(
        raw_coords[_selected_indices(raw_coords.shape[0], max_conformers)]
    )
    residue = residue_feature_matrix(sequence).float()
    confidence = torch.ones(len(sequence), dtype=torch.float32)
    structures = torch.stack(
        [structure_feature_matrix(value, confidence) for value in coords], dim=0
    )
    return EnsembleTensor(
        residue_features=residue,
        coords=coords,
        structure_features=structures,
    )


class _SSLViewDataset(Dataset):
    def __init__(
        self,
        samples: list[ConformerSample],
        max_conformers: int,
    ) -> None:
        self.max_conformers = max_conformers
        self.items: list[tuple[ConformerSample, ConformerView]] = []
        for sample in samples:
            self.items.append((sample, sample.primary))
            if (
                sample.bioemu is not None
                and sample.bioemu.summary_path != sample.primary.summary_path
            ):
                self.items.append((sample, sample.bioemu))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample, view = self.items[index]
        ensemble = _load_ensemble(
            view.summary_path, sample.sequence, self.max_conformers
        )
        return {
            "source_index": sample.source_index,
            "anchor": ensemble,
            "view": ensemble,
            "target_coords": ensemble.coords,
        }


class _SSLPairDataset(Dataset):
    def __init__(
        self,
        samples: list[ConformerSample],
        max_conformers: int,
    ) -> None:
        self.max_conformers = max_conformers
        self.samples = [
            sample
            for sample in samples
            if sample.bioemu is not None
            and sample.bioemu.summary_path != sample.primary.summary_path
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        assert sample.bioemu is not None
        anchor = _load_ensemble(
            sample.primary.summary_path, sample.sequence, self.max_conformers
        )
        view = _load_ensemble(
            sample.bioemu.summary_path, sample.sequence, self.max_conformers
        )
        return {
            "source_index": sample.source_index,
            "anchor": anchor,
            "view": view,
            "target_coords": torch.cat([anchor.coords, view.coords], dim=0),
        }


def _choose_two(count: int) -> tuple[int, int]:
    first = int(torch.randint(count, (1,)).item())
    if count == 1:
        return first, first
    second = int(torch.randint(count - 1, (1,)).item())
    if second >= first:
        second += 1
    return first, second


def _collate_ssl(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_length = max(int(item["anchor"].residue_features.shape[0]) for item in items)
    max_conformers = max(int(item["target_coords"].shape[0]) for item in items)
    batch_size = len(items)
    residue = torch.zeros(batch_size, max_length, RESIDUE_FEATURE_DIM)
    structure = torch.zeros(batch_size, max_length, STRUCTURE_FEATURE_DIM)
    view_structure = torch.zeros_like(structure)
    coords = torch.zeros(batch_size, max_length, 3)
    view_coords = torch.zeros_like(coords)
    confidence = torch.zeros(batch_size, max_length)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    conformers = torch.zeros(batch_size, max_conformers, max_length, 3)
    conformer_mask = torch.zeros(batch_size, max_conformers, dtype=torch.bool)
    for row, item in enumerate(items):
        anchor: EnsembleTensor = item["anchor"]
        view: EnsembleTensor = item["view"]
        length = int(anchor.residue_features.shape[0])
        anchor_index = int(torch.randint(anchor.coords.shape[0], (1,)).item())
        if anchor is view:
            anchor_index, view_index = _choose_two(int(anchor.coords.shape[0]))
        else:
            view_index = int(torch.randint(view.coords.shape[0], (1,)).item())
        target = item["target_coords"]
        count = int(target.shape[0])
        residue[row, :length] = anchor.residue_features
        structure[row, :length] = anchor.structure_features[anchor_index]
        view_structure[row, :length] = view.structure_features[view_index]
        coords[row, :length] = anchor.coords[anchor_index]
        view_coords[row, :length] = view.coords[view_index]
        confidence[row, :length] = 1.0
        mask[row, :length] = True
        conformers[row, :count, :length] = target
        conformer_mask[row, :count] = True
    return {
        "residue_features": residue,
        "structure_features": structure,
        "view_structure_features": view_structure,
        "coords": coords,
        "view_coords": view_coords,
        "plddt": confidence,
        "view_plddt": confidence.clone(),
        "mask": mask,
        "conformer_graph_residue_coords": conformers,
        "conformer_graph_conformer_mask": conformer_mask,
    }


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _train_ssl(
    model: GeometrySSLModel,
    train_samples: list[ConformerSample],
    device: torch.device,
    config: dict[str, Any],
    fold_seed: int,
) -> list[dict[str, float]]:
    maximum = int(config["max_conformers_per_source"])
    view_dataset = _SSLViewDataset(train_samples, maximum)
    pair_dataset = _SSLPairDataset(train_samples, maximum)
    generator = torch.Generator().manual_seed(fold_seed)
    view_loader = DataLoader(
        view_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=_collate_ssl,
    )
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(fold_seed + 1000),
        collate_fn=_collate_ssl,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    model.to(device)
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        same_losses = []
        pair_losses = []
        for batch in view_loader:
            loss, _ = cross_view_geometry_loss(model, _move_batch(batch, device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            same_losses.append(float(loss.detach().cpu()))
        for batch in pair_loader:
            loss, _ = cross_view_geometry_loss(model, _move_batch(batch, device))
            weighted = float(config["matched_source_weight"]) * loss
            optimizer.zero_grad(set_to_none=True)
            weighted.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            pair_losses.append(float(loss.detach().cpu()))
        report = {
            "epoch": epoch,
            "same_source_loss": float(np.mean(same_losses)),
            "matched_source_loss": (
                float(np.mean(pair_losses)) if pair_losses else float("nan")
            ),
        }
        history.append(report)
        print(
            f"ssl epoch={epoch}/{config['epochs']} "
            f"same={report['same_source_loss']:.6f} "
            f"matched={report['matched_source_loss']:.6f}",
            flush=True,
        )
    return history


@torch.no_grad()
def _encode_view(
    model: GeometrySSLModel,
    view: ConformerView,
    sequence: str,
    device: torch.device,
    max_conformers: int,
) -> EncodedSample:
    ensemble = _load_ensemble(view.summary_path, sequence, max_conformers)
    model.eval()
    count = int(ensemble.coords.shape[0])
    length = len(sequence)
    stacked = model.encode(
        residue_features=ensemble.residue_features.unsqueeze(0)
        .expand(count, -1, -1)
        .to(device),
        structure_features=ensemble.structure_features.to(device),
        coords=ensemble.coords.to(device),
        plddt=torch.ones(count, length, device=device),
        mask=torch.ones(count, length, dtype=torch.bool, device=device),
    ).cpu()
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    residue_embedding = torch.cat([mean, std], dim=-1).half()
    probe = mean.mean(dim=0).numpy().astype(np.float32)
    return EncodedSample(residue_embedding=residue_embedding, probe_embedding=probe)


class _ResidualDataset(Dataset):
    def __init__(
        self,
        samples: list[ConformerSample],
        embeddings: dict[int, EncodedSample],
        baseline: dict[int, dict[str, str]],
    ) -> None:
        self.samples = samples
        self.embeddings = embeddings
        self.baseline = baseline

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        probability = float(self.baseline[sample.source_index]["toxicity_probability"])
        probability = min(max(probability, 1e-7), 1.0 - 1e-7)
        return {
            "source_index": sample.source_index,
            "label": sample.label,
            "embedding": self.embeddings[sample.source_index].residue_embedding.float(),
            "base_logit": math.log(probability / (1.0 - probability)),
        }


def _collate_residual(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_length = max(int(item["embedding"].shape[0]) for item in items)
    embedding_dim = int(items[0]["embedding"].shape[1])
    embeddings = torch.zeros(len(items), max_length, embedding_dim)
    mask = torch.zeros(len(items), max_length, dtype=torch.bool)
    for row, item in enumerate(items):
        length = int(item["embedding"].shape[0])
        embeddings[row, :length] = item["embedding"]
        mask[row, :length] = True
    return {
        "source_index": torch.tensor([item["source_index"] for item in items]),
        "embedding": embeddings,
        "mask": mask,
        "base_logit": torch.tensor([item["base_logit"] for item in items]),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.float32),
    }


class ResidueSSLResidualHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        maximum_delta: float,
    ) -> None:
        super().__init__()
        self.maximum_delta = float(maximum_delta)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        embedding: torch.Tensor,
        mask: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.projection(embedding)
        attention_logits = self.attention(hidden).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(attention_logits, dim=1)
        attention_pool = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        max_pool = hidden.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        raw = self.delta_head(
            torch.cat([attention_pool, max_pool, base_logit.unsqueeze(-1)], dim=-1)
        ).squeeze(-1)
        delta = self.maximum_delta * torch.tanh(raw)
        return base_logit + delta, delta


def _train_residual(
    model: ResidueSSLResidualHead,
    samples: list[ConformerSample],
    embeddings: dict[int, EncodedSample],
    baseline: dict[int, dict[str, str]],
    device: torch.device,
    config: dict[str, Any],
    fold_seed: int,
) -> list[dict[str, float]]:
    loader = DataLoader(
        _ResidualDataset(samples, embeddings, baseline),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(fold_seed),
        collate_fn=_collate_residual,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    model.to(device)
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        deltas = []
        for batch in loader:
            logits, delta = model(
                batch["embedding"].to(device),
                batch["mask"].to(device),
                batch["base_logit"].to(device),
            )
            loss = (
                F.binary_cross_entropy_with_logits(logits, batch["label"].to(device))
                + float(config["delta_l1"]) * delta.abs().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            deltas.append(float(delta.abs().mean().detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "mean_absolute_delta": float(np.mean(deltas)),
            }
        )
    return history


@torch.no_grad()
def _predict_residual(
    model: ResidueSSLResidualHead,
    samples: list[ConformerSample],
    embeddings: dict[int, EncodedSample],
    baseline: dict[int, dict[str, str]],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        _ResidualDataset(samples, embeddings, baseline),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_residual,
    )
    model.eval()
    rows = []
    for batch in loader:
        logits, delta = model(
            batch["embedding"].to(device),
            batch["mask"].to(device),
            batch["base_logit"].to(device),
        )
        probabilities = torch.sigmoid(logits).cpu().numpy()
        for offset, source_index in enumerate(batch["source_index"].tolist()):
            rows.append(
                {
                    "source_index": int(source_index),
                    "probability": float(probabilities[offset]),
                    "delta": float(delta[offset].cpu()),
                }
            )
    return rows


def _source_diagnostics(
    samples: list[ConformerSample],
    primary_embeddings: dict[int, EncodedSample],
    bio_embeddings: dict[int, EncodedSample],
    held_fold: int,
    seed: int,
) -> dict[str, Any]:
    pairs = [
        sample
        for sample in samples
        if sample.primary.source != "BioEmu-v1.2"
        and sample.source_index in bio_embeddings
    ]
    train = [sample for sample in pairs if sample.fold != held_fold]
    test = [sample for sample in pairs if sample.fold == held_fold]
    if not train or not test:
        return {"actual": [], "predicted": [], "cosines": []}

    def matrix(rows: list[ConformerSample]) -> tuple[np.ndarray, np.ndarray]:
        legacy = np.stack(
            [primary_embeddings[row.source_index].probe_embedding for row in rows]
        )
        bio = np.stack(
            [bio_embeddings[row.source_index].probe_embedding for row in rows]
        )
        return np.concatenate([legacy, bio]), np.concatenate(
            [np.zeros(len(rows), dtype=np.int64), np.ones(len(rows), dtype=np.int64)]
        )

    train_x, train_y = matrix(train)
    test_x, test_y = matrix(test)
    scaler = StandardScaler()
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    classifier.fit(scaler.fit_transform(train_x), train_y)
    predicted = classifier.predict(scaler.transform(test_x))
    cosines = []
    for sample in test:
        first = primary_embeddings[sample.source_index].probe_embedding
        second = bio_embeddings[sample.source_index].probe_embedding
        cosines.append(
            float(
                np.dot(first, second)
                / (np.linalg.norm(first) * np.linalg.norm(second) + 1e-12)
            )
        )
    return {
        "actual": test_y.tolist(),
        "predicted": predicted.astype(int).tolist(),
        "cosines": cosines,
    }


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _length_mask(lengths: np.ndarray, name: str) -> np.ndarray:
    if name == "11-15":
        return (lengths >= 11) & (lengths <= 15)
    if name == "16-30":
        return (lengths >= 16) & (lengths <= 30)
    if name == "31-50":
        return (lengths >= 31) & (lengths <= 50)
    raise ValueError(name)


def run_phase30(
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    output_root: str | Path | None = None,
    *,
    device_name: str = "auto",
    resume: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _load_protocol(protocol_path)
    inputs = protocol["inputs"]
    samples, cohort = build_multiconformer_samples(
        inputs["fold_manifest"],
        inputs["legacy_manifests"],
        inputs["legacy_cache_roots"],
        inputs["legacy_source_names"],
        inputs["bioemu_manifest"],
        inputs["bioemu_cache_root"],
    )
    expected = protocol["cohort"]
    observed = {
        "canonical_union_samples": cohort["samples"],
        "legacy_samples": cohort["legacy_samples"],
        "bioemu_samples": cohort["bioemu_samples"],
        "matched_legacy_bioemu_pairs": cohort["paired_samples"],
        "bioemu_only_samples": cohort["bioemu_only"],
    }
    for key, value in observed.items():
        if int(expected[key]) != int(value):
            raise ValueError(f"Frozen cohort mismatch for {key}: {value}.")
    output = Path(output_root or Path(protocol_path).parent)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    baseline_rows = _read_csv(inputs["reference_oof"])
    baseline = {int(row["source_index"]): row for row in baseline_rows}
    by_index = {sample.source_index: sample for sample in samples}
    predictions: dict[int, dict[str, Any]] = {}
    source_actual: list[int] = []
    source_predicted: list[int] = []
    paired_cosines: list[float] = []
    fold_histories = []
    pretrain_config = protocol["pretraining"]
    fine_config = protocol["fine_tuning"]
    seed = int(pretrain_config["seed"])

    for fold in sorted({sample.fold for sample in samples}):
        fold_root = output / f"fold_{fold}"
        fold_root.mkdir(exist_ok=True)
        result_path = fold_root / "result.pt"
        if resume and result_path.exists():
            result = torch.load(result_path, map_location="cpu", weights_only=False)
            for row in result["predictions"]:
                predictions[int(row["source_index"])] = row
            source_actual.extend(result["source_diagnostics"]["actual"])
            source_predicted.extend(result["source_diagnostics"]["predicted"])
            paired_cosines.extend(result["source_diagnostics"]["cosines"])
            fold_histories.append(result["history"])
            print(f"phase30 fold={fold} resumed", flush=True)
            continue

        fold_seed = seed + fold
        _set_seed(fold_seed)
        train_samples = [sample for sample in samples if sample.fold != fold]
        held_samples = [sample for sample in samples if sample.fold == fold]
        encoder = GeometrySSLModel(
            structure_feature_dim=int(pretrain_config["structure_feature_dim"]),
            hidden_dim=int(pretrain_config["hidden_dim"]),
            num_layers=int(pretrain_config["num_layers"]),
            dropout=float(pretrain_config["dropout"]),
        )
        ssl_history = _train_ssl(
            encoder, train_samples, device, pretrain_config, fold_seed
        )
        torch.save(
            {
                "state_dict": encoder.state_dict(),
                "fold": fold,
                "seed": fold_seed,
                "config": pretrain_config,
            },
            fold_root / "encoder.pt",
        )
        primary_embeddings: dict[int, EncodedSample] = {}
        bio_embeddings: dict[int, EncodedSample] = {}
        maximum = int(pretrain_config["max_conformers_per_source"])
        for offset, sample in enumerate(samples, start=1):
            primary_embeddings[sample.source_index] = _encode_view(
                encoder, sample.primary, sample.sequence, device, maximum
            )
            if sample.bioemu is not None:
                bio_embeddings[sample.source_index] = _encode_view(
                    encoder, sample.bioemu, sample.sequence, device, maximum
                )
            if offset % 500 == 0:
                print(
                    f"phase30 fold={fold} encoded={offset}/{len(samples)}",
                    flush=True,
                )

        input_dim = int(pretrain_config["hidden_dim"]) * 2
        residual = ResidueSSLResidualHead(
            input_dim=input_dim,
            hidden_dim=int(fine_config["hidden_dim"]),
            dropout=float(fine_config["dropout"]),
            maximum_delta=float(fine_config["maximum_absolute_logit_delta"]),
        )
        residual_history = _train_residual(
            residual,
            train_samples,
            primary_embeddings,
            baseline,
            device,
            fine_config,
            fold_seed,
        )
        held_predictions = _predict_residual(
            residual,
            held_samples,
            primary_embeddings,
            baseline,
            device,
            int(fine_config["batch_size"]),
        )
        for row in held_predictions:
            predictions[int(row["source_index"])] = row
        diagnostics = _source_diagnostics(
            samples,
            primary_embeddings,
            bio_embeddings,
            fold,
            fold_seed,
        )
        source_actual.extend(diagnostics["actual"])
        source_predicted.extend(diagnostics["predicted"])
        paired_cosines.extend(diagnostics["cosines"])
        history = {
            "fold": fold,
            "ssl": ssl_history,
            "residual": residual_history,
        }
        fold_histories.append(history)
        torch.save(
            {
                "predictions": held_predictions,
                "source_diagnostics": diagnostics,
                "history": history,
            },
            result_path,
        )
        torch.save(
            {
                "state_dict": residual.state_dict(),
                "fold": fold,
                "seed": fold_seed,
                "config": fine_config,
            },
            fold_root / "residual_head.pt",
        )

    if set(predictions) != set(by_index):
        raise RuntimeError("Phase 30 did not predict every ensemble-covered row.")
    threshold = float(protocol["evaluation"]["decision_threshold"])
    ordered = sorted(baseline)
    labels = np.asarray([int(baseline[index]["label"]) for index in ordered])
    folds = np.asarray([int(baseline[index]["fold"]) for index in ordered])
    lengths = np.asarray([len(baseline[index]["sequence"]) for index in ordered])
    reference_probabilities = np.asarray(
        [float(baseline[index]["toxicity_probability"]) for index in ordered]
    )
    candidate_probabilities = reference_probabilities.copy()
    index_position = {
        source_index: offset for offset, source_index in enumerate(ordered)
    }
    for source_index, value in predictions.items():
        candidate_probabilities[index_position[source_index]] = float(
            value["probability"]
        )

    reference_metrics = _metrics(labels, reference_probabilities, threshold)
    candidate_metrics = _metrics(labels, candidate_probabilities, threshold)
    by_fold = {}
    fold_non_decreases = 0
    for fold in sorted(np.unique(folds).tolist()):
        selected = folds == fold
        reference_fold = _metrics(
            labels[selected], reference_probabilities[selected], threshold
        )
        candidate_fold = _metrics(
            labels[selected], candidate_probabilities[selected], threshold
        )
        fold_non_decreases += candidate_fold["mcc"] >= reference_fold["mcc"]
        by_fold[str(fold)] = {
            "reference": reference_fold,
            "candidate": candidate_fold,
        }
    by_length = {}
    for name in ("11-15", "16-30", "31-50"):
        selected = _length_mask(lengths, name)
        by_length[name] = {
            "reference": _metrics(
                labels[selected], reference_probabilities[selected], threshold
            ),
            "candidate": _metrics(
                labels[selected], candidate_probabilities[selected], threshold
            ),
        }
    reference_decision = reference_probabilities >= threshold
    candidate_decision = candidate_probabilities >= threshold
    reference_correct = reference_decision == labels
    candidate_correct = candidate_decision == labels
    corrections = int((~reference_correct & candidate_correct).sum())
    regressions = int((reference_correct & ~candidate_correct).sum())
    changed = int((reference_decision != candidate_decision).sum())
    source_bacc = float(balanced_accuracy_score(source_actual, source_predicted))
    paired_cosine = float(np.mean(paired_cosines))
    gates = protocol["promotion_gate"]
    length_mcc_ok = all(
        values["candidate"]["mcc"]
        >= values["reference"]["mcc"] - float(gates["maximum_length_mcc_regression"])
        for values in by_length.values()
    )
    length_auprc_ok = all(
        values["candidate"]["auprc"]
        >= values["reference"]["auprc"]
        - float(gates["maximum_length_auprc_regression"])
        for values in by_length.values()
    )
    gate = {
        "global_mcc_strictly_higher": candidate_metrics["mcc"]
        > float(gates["global_mcc_strictly_above"]),
        "global_auprc_at_least_reference": candidate_metrics["auprc"]
        >= float(gates["global_auprc_at_least"]),
        "brier_within_tolerance": candidate_metrics["brier"]
        <= float(gates["maximum_brier"]),
        "ece_10_within_tolerance": candidate_metrics["ece_10"]
        <= float(gates["maximum_ece_10"]),
        "at_least_three_fold_mcc_non_decreases": fold_non_decreases
        >= int(gates["minimum_fold_mcc_non_decreases"]),
        "all_length_mcc_within_tolerance": length_mcc_ok,
        "all_length_auprc_within_tolerance": length_auprc_ok,
        "positive_net_decision_corrections": corrections - regressions
        >= int(gates["minimum_net_decision_corrections"]),
        "matched_source_probe_bacc_within_limit": source_bacc
        <= float(gates["maximum_matched_source_probe_bacc"]),
        "matched_embedding_cosine_at_least_minimum": paired_cosine
        >= float(gates["minimum_matched_embedding_cosine"]),
    }
    gate["all_conditions_passed"] = all(gate.values())
    rows = []
    fallback_differences = []
    for offset, source_index in enumerate(ordered):
        value = predictions.get(source_index)
        if value is None:
            fallback_differences.append(
                abs(candidate_probabilities[offset] - reference_probabilities[offset])
            )
        rows.append(
            {
                "source_index": source_index,
                "sample_id": baseline[source_index]["sample_id"],
                "fold": int(baseline[source_index]["fold"]),
                "label": int(baseline[source_index]["label"]),
                "sequence": baseline[source_index]["sequence"],
                "length": lengths[offset],
                "ssl_available": int(value is not None),
                "reference_probability": reference_probabilities[offset],
                "candidate_probability": candidate_probabilities[offset],
                "logit_delta": value["delta"] if value else 0.0,
                "decision_threshold": threshold,
                "reference_prediction": int(reference_decision[offset]),
                "candidate_prediction": int(candidate_decision[offset]),
            }
        )
    _write_csv(rows, output / "oof_predictions.csv")
    report = {
        "status": "complete",
        "protocol": protocol,
        "device": str(device),
        "cohort": {**cohort, **observed},
        "overall": {
            "reference": reference_metrics,
            "candidate": candidate_metrics,
        },
        "folds": by_fold,
        "length_strata": by_length,
        "source_robustness": {
            "paired_rows": len(paired_cosines),
            "source_classifier_balanced_accuracy": source_bacc,
            "mean_paired_embedding_cosine": paired_cosine,
        },
        "decision_transitions": {
            "changed": changed,
            "corrected": corrections,
            "regressed": regressions,
            "net": corrections - regressions,
        },
        "fallback": {
            "rows": len(fallback_differences),
            "maximum_absolute_probability_difference": float(
                max(fallback_differences, default=0.0)
            ),
        },
        "fold_mcc_non_decrease_count": int(fold_non_decreases),
        "gate": gate,
        "fold_histories": fold_histories,
    }
    decision_name = (
        "promote_phase30_3d_handoff"
        if gate["all_conditions_passed"]
        else "retain_phase25_3d_handoff"
    )
    decision = {
        "status": "complete",
        "decision": decision_name,
        "promotion_authorized": bool(gate["all_conditions_passed"]),
        "handoff_regeneration_authorized": bool(gate["all_conditions_passed"]),
        "full_bioemu_generation_authorized": False,
        "external_test_labels_read": False,
        "overall": report["overall"],
        "source_robustness": report["source_robustness"],
        "decision_transitions": report["decision_transitions"],
        "gate": gate,
    }
    save_json(report, output / "summary.json")
    save_json(decision, output / "decision.json")
    return report, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    _, decision = run_phase30(
        args.protocol,
        args.output_root,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps({"decision": decision["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
