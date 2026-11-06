"""Standalone source-robust multi-conformer contact-graph OOF encoder."""

from __future__ import annotations

import argparse
import csv
import random
import re
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
from ghxtox.conformer_features import ensemble_feature_tensors
from ghxtox.oof_source_diagnostics import _read_csv, _resolve_summary
from ghxtox.utils import save_json


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {value: index for index, value in enumerate(AA_ALPHABET)}
NODE_FEATURE_DIM = 21 + 8 + 2
EDGE_FEATURE_DIM = 6
GLOBAL_FEATURE_DIM = 12
DEFAULT_SEED = 20260731


@dataclass(frozen=True)
class ConformerView:
    source: str
    summary_path: str


@dataclass(frozen=True)
class ConformerSample:
    source_index: int
    sample_id: str
    sequence: str
    label: int
    fold: int
    primary: ConformerView
    bioemu: ConformerView | None


def _canonical_index(row: dict[str, str]) -> int:
    identifier = str(row.get("original_id") or row["sample_id"])
    match = re.fullmatch(r"peptide_(\d+)", identifier)
    if match is None:
        raise ValueError(f"Cannot derive canonical index from {identifier!r}.")
    return int(match.group(1)) - 1


def _manifest_views(
    manifest: str | Path,
    cache_root: str | Path,
    source_name: str,
) -> dict[int, tuple[ConformerView, dict[str, str]]]:
    views: dict[int, tuple[ConformerView, dict[str, str]]] = {}
    root = Path(cache_root)
    for row in _read_csv(manifest):
        source_index = _canonical_index(row)
        if source_index in views:
            continue
        path = _resolve_summary(row, root)
        views[source_index] = (
            ConformerView(source_name, str(path)),
            row,
        )
    return views


def build_multiconformer_samples(
    fold_manifest: str | Path,
    legacy_manifests: list[str | Path],
    legacy_cache_roots: list[str | Path],
    legacy_source_names: list[str],
    bioemu_manifest: str | Path,
    bioemu_cache_root: str | Path,
) -> tuple[list[ConformerSample], dict[str, Any]]:
    """Build one canonical record per peptide and keep paired views together."""

    if not (
        len(legacy_manifests) == len(legacy_cache_roots) == len(legacy_source_names)
    ):
        raise ValueError("Each legacy manifest needs one cache root and source name.")
    folds = {int(row["source_index"]): row for row in _read_csv(fold_manifest)}
    legacy: dict[int, tuple[ConformerView, dict[str, str]]] = {}
    source_counts = {}
    for manifest, root, name in zip(
        legacy_manifests,
        legacy_cache_roots,
        legacy_source_names,
        strict=True,
    ):
        current = _manifest_views(manifest, root, name)
        overlap = set(legacy) & set(current)
        if overlap:
            raise ValueError(f"Legacy sources overlap at {min(overlap)}.")
        legacy.update(current)
        source_counts[name] = len(current)
    bioemu = _manifest_views(bioemu_manifest, bioemu_cache_root, "BioEmu-v1.2")
    union = sorted(set(legacy) | set(bioemu))
    samples = []
    for source_index in union:
        fold_row = folds.get(source_index)
        if fold_row is None:
            raise ValueError(f"Missing canonical fold row {source_index}.")
        primary_entry = legacy.get(source_index) or bioemu[source_index]
        primary, manifest_row = primary_entry
        sequence = str(fold_row["sequence"])
        label = int(fold_row["label"])
        if str(manifest_row["sequence"]) != sequence:
            raise ValueError(f"Sequence mismatch at source_index {source_index}.")
        if int(manifest_row["label"]) != label:
            raise ValueError(f"Label mismatch at source_index {source_index}.")
        bio_view = bioemu.get(source_index, (None, None))[0]
        if bio_view is not None:
            bio_row = bioemu[source_index][1]
            if str(bio_row["sequence"]) != sequence or int(bio_row["label"]) != label:
                raise ValueError(f"BioEmu alignment mismatch at {source_index}.")
        samples.append(
            ConformerSample(
                source_index=source_index,
                sample_id=f"peptide_{source_index + 1}",
                sequence=sequence,
                label=label,
                fold=int(fold_row["fold"]),
                primary=primary,
                bioemu=bio_view,
            )
        )
    metadata = {
        "samples": len(samples),
        "positive": sum(sample.label for sample in samples),
        "negative": len(samples) - sum(sample.label for sample in samples),
        "legacy_samples": len(legacy),
        "bioemu_samples": len(bioemu),
        "paired_samples": len(set(legacy) & set(bioemu)),
        "bioemu_only": len(set(bioemu) - set(legacy)),
        "source_counts": source_counts,
    }
    return samples, metadata


@lru_cache(maxsize=4096)
def _load_graph(summary_path: str, sequence: str) -> dict[str, torch.Tensor]:
    residue_features, global_features = ensemble_feature_tensors(summary_path)
    length = len(sequence)
    if residue_features.shape != (length, 8):
        raise ValueError(f"Residue feature mismatch in {summary_path}.")
    amino_acids = torch.zeros(length, 21, dtype=torch.float32)
    for index, residue in enumerate(sequence):
        amino_acids[index, AA_TO_INDEX.get(residue, 20)] = 1.0
    position = torch.linspace(0.0, 1.0, length).unsqueeze(1)
    length_feature = torch.full((length, 1), length / 50.0)
    node_features = torch.cat(
        [amino_acids, residue_features.float(), position, length_feature], dim=1
    )
    with np.load(summary_path) as payload:
        pair_mean = payload["pair_distance_mean"].astype(np.float32) / 20.0
        pair_std = payload["pair_distance_std"].astype(np.float32) / 5.0
        occupancy_6 = payload["contact_occupancy_6"].astype(np.float32)
        occupancy_8 = payload["contact_occupancy_8"].astype(np.float32)
        occupancy_10 = payload["contact_occupancy_10"].astype(np.float32)
        conformer_count = int(payload["aligned_coords"].shape[0])
    edge_features = np.stack(
        [
            np.clip(pair_mean, 0.0, 3.0),
            np.clip(pair_std, 0.0, 3.0),
            occupancy_6,
            occupancy_8,
            occupancy_10,
            occupancy_8 * (1.0 - occupancy_8),
        ],
        axis=-1,
    )
    return {
        "node": node_features,
        "edge": torch.from_numpy(
            np.nan_to_num(edge_features, nan=0.0, posinf=3.0, neginf=0.0)
        ),
        "global": global_features.float(),
        "conformer_count": torch.tensor(conformer_count),
    }


def _pack_graphs(graphs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_length = max(int(graph["node"].shape[0]) for graph in graphs)
    batch_size = len(graphs)
    node = torch.zeros(batch_size, max_length, NODE_FEATURE_DIM)
    edge = torch.zeros(batch_size, max_length, max_length, EDGE_FEATURE_DIM)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    global_features = torch.zeros(batch_size, GLOBAL_FEATURE_DIM)
    conformer_count = torch.zeros(batch_size, dtype=torch.long)
    for index, graph in enumerate(graphs):
        length = int(graph["node"].shape[0])
        node[index, :length] = graph["node"]
        edge[index, :length, :length] = graph["edge"]
        mask[index, :length] = True
        global_features[index] = graph["global"]
        conformer_count[index] = graph["conformer_count"]
    return {
        "node": node,
        "edge": edge,
        "mask": mask,
        "global": global_features,
        "conformer_count": conformer_count,
    }


class _TrainingDataset(Dataset):
    def __init__(self, samples: list[ConformerSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "source_index": sample.source_index,
            "label": sample.label,
            "primary": _load_graph(sample.primary.summary_path, sample.sequence),
            "paired": (
                _load_graph(sample.bioemu.summary_path, sample.sequence)
                if sample.bioemu is not None
                and sample.bioemu.summary_path != sample.primary.summary_path
                else None
            ),
        }


def _collate_training(items: list[dict[str, Any]]) -> dict[str, Any]:
    pair_indices = [
        index for index, item in enumerate(items) if item["paired"] is not None
    ]
    return {
        "source_index": torch.tensor([item["source_index"] for item in items]),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.float32),
        "primary": _pack_graphs([item["primary"] for item in items]),
        "pair_indices": torch.tensor(pair_indices, dtype=torch.long),
        "paired": (
            _pack_graphs([items[index]["paired"] for index in pair_indices])
            if pair_indices
            else None
        ),
    }


class _ViewDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[ConformerSample, ConformerView]],
    ) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample, view = self.items[index]
        return {
            "source_index": sample.source_index,
            "source": view.source,
            "graph": _load_graph(view.summary_path, sample.sequence),
        }


def _collate_views(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_index": torch.tensor([item["source_index"] for item in items]),
        "source": [item["source"] for item in items],
        "graph": _pack_graphs([item["graph"] for item in items]),
    }


def _move_graph(
    graph: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in graph.items()}


class EdgeGatedContactEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        embedding_dim: int = 32,
        layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(EDGE_FEATURE_DIM, 16),
                    nn.GELU(),
                    nn.Linear(16, 1),
                )
                for _ in range(layers)
            ]
        )
        self.values = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(layers)]
        )
        self.updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.global_projection = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, hidden_dim),
            nn.GELU(),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        mask: torch.Tensor,
        global_features: torch.Tensor,
        conformer_count: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del conformer_count
        hidden = self.node_projection(node)
        hidden = hidden * mask.unsqueeze(-1)
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        diagonal = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)
        pair_mask = pair_mask & ~diagonal.unsqueeze(0)
        for gate_layer, value_layer, update_layer, norm_layer in zip(
            self.edge_gates,
            self.values,
            self.updates,
            self.norms,
            strict=True,
        ):
            weights = torch.sigmoid(gate_layer(edge).squeeze(-1)) * pair_mask.float()
            denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            aggregated = torch.bmm(weights, value_layer(hidden)) / denominator
            update = update_layer(torch.cat([hidden, aggregated], dim=-1))
            hidden = norm_layer(hidden + update) * mask.unsqueeze(-1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        mean_pool = hidden.sum(dim=1) / denominator
        max_pool = hidden.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        global_hidden = self.global_projection(global_features)
        embedding = self.embedding_head(
            torch.cat([mean_pool, max_pool, global_hidden], dim=-1)
        )
        return self.classifier(embedding).squeeze(-1), embedding


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _predict_views(
    model: nn.Module,
    items: list[tuple[ConformerSample, ConformerView]],
    device: torch.device,
    batch_size: int,
) -> dict[int, dict[str, Any]]:
    loader = DataLoader(
        _ViewDataset(items),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_views,
    )
    model.eval()
    output = {}
    with torch.no_grad():
        for batch in loader:
            graph = _move_graph(batch["graph"], device)
            logits, embeddings = model(
                graph["node"], graph["edge"], graph["mask"], graph["global"]
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()
            embeddings_array = embeddings.cpu().numpy()
            counts = graph["conformer_count"].cpu().numpy()
            for offset, source_index in enumerate(batch["source_index"].tolist()):
                output[int(source_index)] = {
                    "source": batch["source"][offset],
                    "logit": float(logits[offset].cpu()),
                    "probability": float(probabilities[offset]),
                    "embedding": embeddings_array[offset].astype(np.float32),
                    "conformer_count": int(counts[offset]),
                }
    return output


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_multiconformer_graph_oof(
    fold_manifest: str | Path,
    legacy_manifests: list[str | Path],
    legacy_cache_roots: list[str | Path],
    legacy_source_names: list[str],
    bioemu_manifest: str | Path,
    bioemu_cache_root: str | Path,
    baseline_oof: str | Path,
    output_root: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    consistency_weight: float = 0.10,
    hidden_dim: int = 32,
    embedding_dim: int = 32,
    layers: int = 2,
    dropout: float = 0.10,
    threshold: float = 0.50,
    baseline_threshold: float = 0.677819,
    device_name: str = "auto",
) -> tuple[dict[str, Any], dict[str, Any]]:
    samples, cohort = build_multiconformer_samples(
        fold_manifest,
        legacy_manifests,
        legacy_cache_roots,
        legacy_source_names,
        bioemu_manifest,
        bioemu_cache_root,
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    by_index = {sample.source_index: sample for sample in samples}
    primary_predictions: dict[int, dict[str, Any]] = {}
    bioemu_predictions: dict[int, dict[str, Any]] = {}
    source_actual = []
    source_predicted = []
    paired_cosines = []
    paired_probability_deltas = []
    fold_histories = []

    for fold in sorted({sample.fold for sample in samples}):
        fold_seed = seed + fold
        _set_seed(fold_seed)
        train_samples = [sample for sample in samples if sample.fold != fold]
        test_samples = [sample for sample in samples if sample.fold == fold]
        train_labels = np.asarray([sample.label for sample in train_samples])
        positive_weight = float(np.sum(train_labels == 0) / np.sum(train_labels == 1))
        generator = torch.Generator().manual_seed(fold_seed)
        loader = DataLoader(
            _TrainingDataset(train_samples),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=_collate_training,
        )
        model = EdgeGatedContactEncoder(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            layers=layers,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        pos_weight = torch.tensor(positive_weight, device=device)
        history = []
        for epoch in range(epochs):
            model.train()
            losses = []
            supervised_losses = []
            consistency_losses = []
            for batch in loader:
                labels = batch["label"].to(device)
                primary_graph = _move_graph(batch["primary"], device)
                primary_logits, primary_embeddings = model(
                    primary_graph["node"],
                    primary_graph["edge"],
                    primary_graph["mask"],
                    primary_graph["global"],
                )
                per_sample = F.binary_cross_entropy_with_logits(
                    primary_logits,
                    labels,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                consistency = torch.zeros((), device=device)
                pair_indices = batch["pair_indices"].to(device)
                if batch["paired"] is not None:
                    paired_graph = _move_graph(batch["paired"], device)
                    paired_logits, paired_embeddings = model(
                        paired_graph["node"],
                        paired_graph["edge"],
                        paired_graph["mask"],
                        paired_graph["global"],
                    )
                    paired_loss = F.binary_cross_entropy_with_logits(
                        paired_logits,
                        labels[pair_indices],
                        reduction="none",
                        pos_weight=pos_weight,
                    )
                    per_sample = per_sample.clone()
                    per_sample[pair_indices] = 0.5 * (
                        per_sample[pair_indices] + paired_loss
                    )
                    consistency = (
                        1.0
                        - F.cosine_similarity(
                            primary_embeddings[pair_indices], paired_embeddings, dim=1
                        )
                    ).mean()
                supervised = per_sample.mean()
                loss = supervised + consistency_weight * consistency
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                supervised_losses.append(float(supervised.detach().cpu()))
                consistency_losses.append(float(consistency.detach().cpu()))
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(np.mean(losses)),
                    "supervised_loss": float(np.mean(supervised_losses)),
                    "consistency_loss": float(np.mean(consistency_losses)),
                }
            )
            print(
                f"fold={fold} epoch={epoch + 1}/{epochs} "
                f"loss={history[-1]['loss']:.6f} "
                f"consistency={history[-1]['consistency_loss']:.6f}",
                flush=True,
            )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "fold": fold,
                "seed": fold_seed,
                "config": {
                    "hidden_dim": hidden_dim,
                    "embedding_dim": embedding_dim,
                    "layers": layers,
                    "dropout": dropout,
                },
            },
            checkpoints / f"fold_{fold}.pt",
        )
        fold_histories.append({"fold": fold, "epochs": history})

        primary_predictions.update(
            _predict_views(
                model,
                [(sample, sample.primary) for sample in test_samples],
                device,
                batch_size,
            )
        )
        test_bio = [sample for sample in test_samples if sample.bioemu is not None]
        bioemu_predictions.update(
            _predict_views(
                model,
                [(sample, sample.bioemu) for sample in test_bio],  # type: ignore[arg-type]
                device,
                batch_size,
            )
        )

        train_pairs = [
            sample
            for sample in train_samples
            if sample.bioemu is not None
            and sample.bioemu.summary_path != sample.primary.summary_path
        ]
        test_pairs = [
            sample
            for sample in test_samples
            if sample.bioemu is not None
            and sample.bioemu.summary_path != sample.primary.summary_path
        ]
        if train_pairs and test_pairs:
            train_legacy = _predict_views(
                model,
                [(sample, sample.primary) for sample in train_pairs],
                device,
                batch_size,
            )
            train_bio = _predict_views(
                model,
                [(sample, sample.bioemu) for sample in train_pairs],  # type: ignore[arg-type]
                device,
                batch_size,
            )
            test_legacy = _predict_views(
                model,
                [(sample, sample.primary) for sample in test_pairs],
                device,
                batch_size,
            )
            test_bio_values = _predict_views(
                model,
                [(sample, sample.bioemu) for sample in test_pairs],  # type: ignore[arg-type]
                device,
                batch_size,
            )
            train_x = np.concatenate(
                [
                    np.stack(
                        [train_legacy[s.source_index]["embedding"] for s in train_pairs]
                    ),
                    np.stack(
                        [train_bio[s.source_index]["embedding"] for s in train_pairs]
                    ),
                ]
            )
            train_y = np.concatenate(
                [np.zeros(len(train_pairs)), np.ones(len(train_pairs))]
            )
            test_x = np.concatenate(
                [
                    np.stack(
                        [test_legacy[s.source_index]["embedding"] for s in test_pairs]
                    ),
                    np.stack(
                        [
                            test_bio_values[s.source_index]["embedding"]
                            for s in test_pairs
                        ]
                    ),
                ]
            )
            test_y = np.concatenate(
                [np.zeros(len(test_pairs)), np.ones(len(test_pairs))]
            )
            scaler = StandardScaler()
            source_model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                random_state=fold_seed,
            )
            source_model.fit(scaler.fit_transform(train_x), train_y)
            source_predictions = source_model.predict(scaler.transform(test_x))
            source_actual.extend(test_y.astype(int).tolist())
            source_predicted.extend(source_predictions.astype(int).tolist())
            for sample in test_pairs:
                first = test_legacy[sample.source_index]
                second = test_bio_values[sample.source_index]
                paired_cosines.append(
                    float(
                        np.dot(first["embedding"], second["embedding"])
                        / (
                            np.linalg.norm(first["embedding"])
                            * np.linalg.norm(second["embedding"])
                            + 1e-12
                        )
                    )
                )
                paired_probability_deltas.append(
                    abs(first["probability"] - second["probability"])
                )

    if set(primary_predictions) != set(by_index):
        raise RuntimeError("OOF predictions do not cover the complete 3D union.")
    baseline = {int(row["source_index"]): row for row in _read_csv(baseline_oof)}
    fold_rows = {int(row["source_index"]): row for row in _read_csv(fold_manifest)}
    ordered_indices = sorted(by_index)
    labels = np.asarray([by_index[index].label for index in ordered_indices])
    folds = np.asarray([by_index[index].fold for index in ordered_indices])
    probabilities = np.asarray(
        [primary_predictions[index]["probability"] for index in ordered_indices]
    )
    baseline_probabilities = np.asarray(
        [float(baseline[index]["toxicity_probability"]) for index in ordered_indices]
    )
    candidate_metrics = _metrics(labels, probabilities, threshold)
    baseline_metrics = _metrics(labels, baseline_probabilities, baseline_threshold)
    by_fold = {}
    fold_wins = 0
    for fold in sorted(np.unique(folds).tolist()):
        selected = folds == fold
        candidate_fold = _metrics(labels[selected], probabilities[selected], threshold)
        baseline_fold = _metrics(
            labels[selected], baseline_probabilities[selected], baseline_threshold
        )
        fold_wins += candidate_fold["mcc"] > baseline_fold["mcc"]
        by_fold[str(fold)] = {
            "candidate": candidate_fold,
            "baseline_3d_v2": baseline_fold,
        }
    lengths = np.asarray([len(by_index[index].sequence) for index in ordered_indices])
    length_masks = {
        "11-15": lengths <= 15,
        "16-30": (lengths >= 16) & (lengths <= 30),
        "31-50": (lengths >= 31) & (lengths <= 50),
    }
    by_length = {
        name: {
            "candidate": _metrics(labels[selected], probabilities[selected], threshold),
            "baseline_3d_v2": _metrics(
                labels[selected], baseline_probabilities[selected], baseline_threshold
            ),
        }
        for name, selected in length_masks.items()
    }
    source_bacc = float(balanced_accuracy_score(source_actual, source_predicted))
    paired_cosine = float(np.mean(paired_cosines))
    paired_probability_delta = float(np.mean(paired_probability_deltas))
    replacement_gate = {
        "mcc_strictly_higher_than_3d_v2": candidate_metrics["mcc"]
        > baseline_metrics["mcc"],
        "auprc_at_least_baseline_minus_0_002": candidate_metrics["auprc"]
        >= baseline_metrics["auprc"] - 0.002,
        "at_least_three_fold_mcc_wins": fold_wins >= 3,
    }
    replacement_gate["all_conditions_passed"] = all(replacement_gate.values())
    handoff_gate = {
        "mcc_at_least_0_70": candidate_metrics["mcc"] >= 0.70,
        "auprc_at_least_0_85": candidate_metrics["auprc"] >= 0.85,
        "source_classifier_bacc_at_most_0_65": source_bacc <= 0.65,
        "paired_embedding_cosine_at_least_0_90": paired_cosine >= 0.90,
        "paired_probability_delta_at_most_0_10": paired_probability_delta <= 0.10,
    }
    handoff_gate["all_conditions_passed"] = all(handoff_gate.values())

    prediction_rows = []
    bioemu_rows = []
    full_handoff_rows = []
    embeddings = np.zeros((len(fold_rows), embedding_dim), dtype=np.float32)
    available = np.zeros(len(fold_rows), dtype=np.bool_)
    for index in sorted(fold_rows):
        fold_row = fold_rows[index]
        value = primary_predictions.get(index)
        full_handoff_rows.append(
            {
                "source_index": index,
                "sample_id": fold_row["sample_id"],
                "fold": int(fold_row["fold"]),
                "label": int(fold_row["label"]),
                "structure_available": int(value is not None),
                "toxicity_probability": value["probability"] if value else "",
                "quality_score": (
                    min(int(value["conformer_count"]), 8) / 8.0 if value else 0.0
                ),
                "generator_source": value["source"] if value else "not_generated",
            }
        )
        if value is None:
            continue
        sample = by_index[index]
        embeddings[index] = value["embedding"]
        available[index] = True
        prediction_rows.append(
            {
                "source_index": index,
                "sample_id": fold_row["sample_id"],
                "fold": sample.fold,
                "label": sample.label,
                "sequence": sample.sequence,
                "length": len(sample.sequence),
                "generator_source": value["source"],
                "conformer_count": value["conformer_count"],
                "logit": value["logit"],
                "toxicity_probability": value["probability"],
                "decision_threshold": threshold,
                "prediction": int(value["probability"] >= threshold),
            }
        )
    for index in sorted(bioemu_predictions):
        sample = by_index[index]
        value = bioemu_predictions[index]
        bioemu_rows.append(
            {
                "source_index": index,
                "sample_id": sample.sample_id,
                "fold": sample.fold,
                "label": sample.label,
                "paired_legacy": int(sample.primary.source != "BioEmu-v1.2"),
                "logit": value["logit"],
                "toxicity_probability": value["probability"],
                "conformer_count": value["conformer_count"],
            }
        )

    _write_csv(prediction_rows, output / "oof_predictions.csv")
    _write_csv(bioemu_rows, output / "bioemu_oof_views.csv")
    _write_csv(full_handoff_rows, output / "full_handoff_manifest.csv")
    np.savez_compressed(
        output / "full_oof_embeddings.npz",
        embeddings=embeddings,
        structure_available=available,
        source_index=np.arange(len(fold_rows), dtype=np.int64),
    )
    report = {
        "protocol": {
            "scope": "standalone train-only multi-conformer graph OOF",
            "external_test_labels_read": False,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "consistency_weight": consistency_weight,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "layers": layers,
            "dropout": dropout,
            "threshold": threshold,
            "baseline_threshold": baseline_threshold,
            "device": str(device),
            "early_stopping": False,
            "fusion_weights_searched": False,
        },
        "cohort": cohort,
        "primary": {
            "candidate": candidate_metrics,
            "baseline_3d_v2": baseline_metrics,
        },
        "by_fold": by_fold,
        "by_length": by_length,
        "source_robustness": {
            "paired_samples": len(paired_cosines),
            "source_classifier_balanced_accuracy": source_bacc,
            "mean_paired_embedding_cosine": paired_cosine,
            "mean_absolute_paired_probability_delta": paired_probability_delta,
        },
        "fold_histories": fold_histories,
        "replacement_gate": {
            **replacement_gate,
            "fold_mcc_wins": int(fold_wins),
        },
        "fusion_handoff_gate": handoff_gate,
    }
    if replacement_gate["all_conditions_passed"]:
        decision_name = "retain_as_standalone_3d_replacement_candidate"
    elif handoff_gate["all_conditions_passed"]:
        decision_name = "retain_as_experimental_3d_fusion_handoff"
    else:
        decision_name = "stop_multiconformer_graph_candidate"
    decision = {
        "status": "complete",
        "decision": decision_name,
        "standalone_replacement_authorized": bool(
            replacement_gate["all_conditions_passed"]
        ),
        "experimental_fusion_handoff_authorized": bool(
            handoff_gate["all_conditions_passed"]
        ),
        "full_bioemu_generation_authorized": False,
        "default_model_change_authorized": False,
        "external_test_labels_read": False,
        "primary": report["primary"],
        "source_robustness": report["source_robustness"],
        "replacement_gate": report["replacement_gate"],
        "fusion_handoff_gate": report["fusion_handoff_gate"],
    }
    save_json(report, output / "summary.json")
    save_json(decision, output / "decision.json")
    save_json({"folds": fold_histories}, output / "history.json")
    return report, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--legacy-manifest", nargs="+", required=True)
    parser.add_argument("--legacy-cache-root", nargs="+", required=True)
    parser.add_argument("--legacy-source-name", nargs="+", required=True)
    parser.add_argument("--bioemu-manifest", required=True)
    parser.add_argument("--bioemu-cache-root", required=True)
    parser.add_argument("--baseline-oof", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_multiconformer_graph_oof(
        args.fold_manifest,
        args.legacy_manifest,
        args.legacy_cache_root,
        args.legacy_source_name,
        args.bioemu_manifest,
        args.bioemu_cache_root,
        args.baseline_oof,
        args.output_root,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
