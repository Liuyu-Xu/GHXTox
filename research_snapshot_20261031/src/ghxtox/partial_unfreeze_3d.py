"""Fold-pure last-layer fine-tuning of the Phase 30 ensemble 3D encoder.

The supervised model never receives Phase 25 probabilities during fitting.
After a fold-specific standalone 3D evidence head has predicted its held fold,
one fixed bounded evidence term is added to the held Phase 25 logit. Samples
without an ensemble remain exact Phase 25 fallbacks.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ghxtox.bootstrap_ci import _metrics
from ghxtox.ensemble_domain_ssl_3d import (
    EncodedSample,
    EnsembleTensor,
    _encode_view,
    _length_mask,
    _load_ensemble,
    _set_seed,
    _sha256,
    _source_diagnostics,
    _write_csv,
)
from ghxtox.features import RESIDUE_FEATURE_DIM
from ghxtox.geometry_features import STRUCTURE_FEATURE_DIM
from ghxtox.geometry_pretrain import GeometrySSLModel
from ghxtox.multiconformer_graph_3d import (
    ConformerSample,
    build_multiconformer_samples,
)
from ghxtox.oof_source_diagnostics import _read_csv
from ghxtox.utils import save_json


DEFAULT_PROTOCOL = "runs/partial_unfreeze_3d/protocol.json"


def _load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_candidate_oof":
        raise ValueError("Phase 32 protocol must be frozen before execution.")
    if bool(protocol.get("external_test_labels_read", True)):
        raise ValueError("Phase 32 protocol must keep external labels sealed.")
    for input_path, expected in protocol["inputs"]["sha256"].items():
        actual = _sha256(input_path)
        if actual != expected:
            raise ValueError(
                f"Frozen input hash mismatch for {input_path}: {actual} != {expected}."
            )
    return protocol


class _EvidenceDataset(Dataset):
    """Structure-and-label-only dataset; no base probability can enter here."""

    def __init__(self, samples: list[ConformerSample], max_conformers: int) -> None:
        self.samples = samples
        self.max_conformers = int(max_conformers)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        ensemble = _load_ensemble(
            sample.primary.summary_path,
            sample.sequence,
            self.max_conformers,
        )
        return {
            "source_index": sample.source_index,
            "label": sample.label,
            "ensemble": ensemble,
        }


def _view_indices(count: int, maximum: int, stochastic: bool) -> torch.Tensor:
    if count <= 0:
        raise ValueError("An evidence sample must contain at least one conformer.")
    take = min(int(count), int(maximum))
    if stochastic:
        return torch.randperm(count)[:take]
    if count <= take:
        return torch.arange(count)
    return torch.from_numpy(np.linspace(0, count - 1, take, dtype=np.int64))


def _collate_evidence(
    items: list[dict[str, Any]],
    *,
    conformers_per_sample: int,
    stochastic: bool,
) -> dict[str, torch.Tensor]:
    max_length = max(int(item["ensemble"].residue_features.shape[0]) for item in items)
    selected = [
        _view_indices(
            int(item["ensemble"].coords.shape[0]),
            conformers_per_sample,
            stochastic,
        )
        for item in items
    ]
    total_views = sum(int(value.numel()) for value in selected)
    residue = torch.zeros(total_views, max_length, RESIDUE_FEATURE_DIM)
    structure = torch.zeros(total_views, max_length, STRUCTURE_FEATURE_DIM)
    coords = torch.zeros(total_views, max_length, 3)
    confidence = torch.zeros(total_views, max_length)
    view_mask = torch.zeros(total_views, max_length, dtype=torch.bool)
    sample_mask = torch.zeros(len(items), max_length, dtype=torch.bool)
    owners = torch.zeros(total_views, dtype=torch.long)
    cursor = 0
    for row, (item, indices) in enumerate(zip(items, selected, strict=True)):
        ensemble: EnsembleTensor = item["ensemble"]
        length = int(ensemble.residue_features.shape[0])
        sample_mask[row, :length] = True
        for conformer_index in indices.tolist():
            residue[cursor, :length] = ensemble.residue_features
            structure[cursor, :length] = ensemble.structure_features[conformer_index]
            coords[cursor, :length] = ensemble.coords[conformer_index]
            confidence[cursor, :length] = 1.0
            view_mask[cursor, :length] = True
            owners[cursor] = row
            cursor += 1
    return {
        "source_index": torch.tensor(
            [item["source_index"] for item in items], dtype=torch.long
        ),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.float32),
        "residue_features": residue,
        "structure_features": structure,
        "coords": coords,
        "plddt": confidence,
        "view_mask": view_mask,
        "sample_mask": sample_mask,
        "view_owner": owners,
    }


class StandaloneEvidenceHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(embedding)
        attention_logits = self.attention(hidden).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, -1e4)
        attention = torch.softmax(attention_logits, dim=1)
        attention_pool = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        maximum = hidden.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        return self.output(torch.cat([attention_pool, maximum], dim=-1)).squeeze(-1)


class PartialUnfreezeEvidenceModel(nn.Module):
    def __init__(
        self,
        encoder: GeometrySSLModel,
        *,
        head_hidden_dim: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = StandaloneEvidenceHead(
            input_dim=encoder.hidden_dim * 2,
            hidden_dim=int(head_hidden_dim),
            dropout=float(head_dropout),
        )

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder.encode(
            residue_features=batch["residue_features"],
            structure_features=batch["structure_features"],
            coords=batch["coords"],
            plddt=batch["plddt"],
            mask=batch["view_mask"],
        )
        owner = batch["view_owner"]
        batch_size, max_length = batch["sample_mask"].shape
        embedding_sum = hidden.new_zeros(batch_size, max_length, hidden.shape[-1])
        square_sum = torch.zeros_like(embedding_sum)
        embedding_sum.index_add_(0, owner, hidden)
        square_sum.index_add_(0, owner, hidden.square())
        counts = torch.bincount(owner, minlength=batch_size).to(hidden.dtype)
        counts = counts.view(-1, 1, 1).clamp_min(1.0)
        mean = embedding_sum / counts
        variance = (square_sum / counts - mean.square()).clamp_min(0.0)
        # A zero conformer variance is valid, but sqrt has an infinite gradient
        # at exactly zero.  The epsilon prevents 0 * inf from producing NaN
        # gradients when the zero-initialized head starts to backpropagate.
        embedding = torch.cat([mean, (variance + 1e-6).sqrt()], dim=-1)
        logits = self.head(embedding, batch["sample_mask"])
        return logits, embedding


def _configure_partial_unfreeze(model: PartialUnfreezeEvidenceModel) -> list[str]:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.encoder.encoder.layers[-1].parameters():
        parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    return [name for name, value in model.named_parameters() if value.requires_grad]


def _set_fine_tune_mode(model: PartialUnfreezeEvidenceModel) -> None:
    model.eval()
    model.head.train()
    model.encoder.encoder.layers[-1].train()


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _train_fold(
    model: PartialUnfreezeEvidenceModel,
    samples: list[ConformerSample],
    device: torch.device,
    config: dict[str, Any],
    fold_seed: int,
) -> tuple[list[dict[str, float]], list[str]]:
    collate = partial(
        _collate_evidence,
        conformers_per_sample=int(config["training_conformers_per_sample"]),
        stochastic=True,
    )
    loader = DataLoader(
        _EvidenceDataset(samples, int(config["inference_max_conformers_per_sample"])),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(fold_seed),
        collate_fn=collate,
    )
    trainable = _configure_partial_unfreeze(model)
    last_layer_parameters = list(model.encoder.encoder.layers[-1].parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": last_layer_parameters,
                "lr": float(config["encoder_learning_rate"]),
            },
            {
                "params": model.head.parameters(),
                "lr": float(config["head_learning_rate"]),
            },
        ],
        weight_decay=float(config["weight_decay"]),
    )
    positives = sum(int(sample.label) for sample in samples)
    negatives = len(samples) - positives
    positive_weight = torch.tensor(negatives / positives, device=device)
    model.to(device)
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        _set_fine_tune_mode(model)
        total_loss = 0.0
        total_rows = 0
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            logits, _ = model(batch)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch["label"],
                pos_weight=positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                float(config["gradient_clip_norm"]),
            )
            optimizer.step()
            count = int(batch["label"].shape[0])
            total_loss += float(loss.detach().cpu()) * count
            total_rows += count
        report = {
            "epoch": epoch,
            "loss": total_loss / max(total_rows, 1),
            "positive_weight": float(positive_weight.detach().cpu()),
        }
        history.append(report)
        print(
            f"phase32 epoch={epoch}/{config['epochs']} loss={report['loss']:.6f}",
            flush=True,
        )
    return history, trainable


@torch.no_grad()
def _predict_evidence(
    model: PartialUnfreezeEvidenceModel,
    samples: list[ConformerSample],
    device: torch.device,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    loader = DataLoader(
        _EvidenceDataset(samples, int(config["inference_max_conformers_per_sample"])),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=partial(
            _collate_evidence,
            conformers_per_sample=int(config["inference_max_conformers_per_sample"]),
            stochastic=False,
        ),
    )
    model.eval()
    rows = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        logits, _ = model(batch)
        probabilities = torch.sigmoid(logits)
        for offset, source_index in enumerate(batch["source_index"].tolist()):
            rows.append(
                {
                    "source_index": int(source_index),
                    "standalone_logit": float(logits[offset].cpu()),
                    "standalone_probability": float(probabilities[offset].cpu()),
                }
            )
    return rows


def _combine_held_evidence(
    reference_probability: float,
    standalone_logit: float,
    maximum_delta: float,
) -> tuple[float, float]:
    probability = min(max(float(reference_probability), 1e-7), 1.0 - 1e-7)
    reference_logit = math.log(probability / (1.0 - probability))
    delta = float(maximum_delta) * math.tanh(float(standalone_logit))
    candidate = 1.0 / (1.0 + math.exp(-(reference_logit + delta)))
    return candidate, delta


@torch.no_grad()
def _post_fine_tune_source_diagnostics(
    model: PartialUnfreezeEvidenceModel,
    samples: list[ConformerSample],
    device: torch.device,
    config: dict[str, Any],
    held_fold: int,
    fold_seed: int,
) -> dict[str, Any]:
    pairs = [
        sample
        for sample in samples
        if sample.bioemu is not None
        and sample.bioemu.summary_path != sample.primary.summary_path
    ]
    primary: dict[int, EncodedSample] = {}
    bio: dict[int, EncodedSample] = {}
    maximum = int(config["inference_max_conformers_per_sample"])
    model.eval()
    for sample in pairs:
        primary[sample.source_index] = _encode_view(
            model.encoder, sample.primary, sample.sequence, device, maximum
        )
        assert sample.bioemu is not None
        bio[sample.source_index] = _encode_view(
            model.encoder, sample.bioemu, sample.sequence, device, maximum
        )
    return _source_diagnostics(samples, primary, bio, held_fold, fold_seed)


def _load_initialized_model(
    checkpoint_path: str | Path,
    fold: int,
    protocol: dict[str, Any],
) -> PartialUnfreezeEvidenceModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["fold"]) != int(fold):
        raise ValueError(f"Phase 30 checkpoint fold mismatch at {checkpoint_path}.")
    expected = protocol["initialization"]
    for key in ("structure_feature_dim", "hidden_dim", "num_layers", "dropout"):
        if checkpoint["config"][key] != expected[key]:
            raise ValueError(f"Phase 30 checkpoint config mismatch for {key}.")
    encoder = GeometrySSLModel(
        structure_feature_dim=int(expected["structure_feature_dim"]),
        hidden_dim=int(expected["hidden_dim"]),
        num_layers=int(expected["num_layers"]),
        dropout=float(expected["dropout"]),
    )
    encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    fine = protocol["fine_tuning"]
    return PartialUnfreezeEvidenceModel(
        encoder,
        head_hidden_dim=int(fine["head_hidden_dim"]),
        head_dropout=float(fine["head_dropout"]),
    )


def _load_and_validate_inputs(
    protocol_path: str | Path,
) -> tuple[
    dict[str, Any],
    list[ConformerSample],
    dict[str, int],
    dict[int, dict[str, str]],
]:
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
    observed = {
        "canonical_union_samples": int(cohort["samples"]),
        "legacy_samples": int(cohort["legacy_samples"]),
        "bioemu_samples": int(cohort["bioemu_samples"]),
        "matched_legacy_bioemu_pairs": int(cohort["paired_samples"]),
        "bioemu_only_samples": int(cohort["bioemu_only"]),
    }
    for key, value in observed.items():
        if int(protocol["cohort"][key]) != value:
            raise ValueError(f"Frozen cohort mismatch for {key}: {value}.")
    baseline_rows = _read_csv(inputs["reference_oof"])
    baseline = {int(row["source_index"]): row for row in baseline_rows}
    if len(baseline) != len(baseline_rows):
        raise ValueError("Phase 25 reference contains duplicate source indices.")
    for sample in samples:
        row = baseline.get(sample.source_index)
        if row is None:
            raise ValueError(f"Phase 25 is missing source index {sample.source_index}.")
        if (
            row["sample_id"] != sample.sample_id
            or row["sequence"] != sample.sequence
            or int(row["label"]) != sample.label
            or int(row["fold"]) != sample.fold
        ):
            raise ValueError(f"Input alignment mismatch for {sample.sample_id}.")
    for fold in sorted({sample.fold for sample in samples}):
        path = inputs["phase30_encoder_pattern"].format(fold=fold)
        _load_initialized_model(path, fold, protocol)
    return protocol, samples, observed, baseline


def verify_phase32(protocol_path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol, samples, cohort, baseline = _load_and_validate_inputs(protocol_path)
    return {
        "status": "verified",
        "phase": int(protocol["phase"]),
        "cohort": cohort,
        "reference_rows": len(baseline),
        "folds": sorted({sample.fold for sample in samples}),
        "trainable_encoder_modules": protocol["initialization"][
            "trainable_encoder_modules"
        ],
        "external_test_labels_read": False,
    }


def run_phase32(
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    output_root: str | Path | None = None,
    *,
    device_name: str = "auto",
    resume: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, samples, observed, baseline = _load_and_validate_inputs(protocol_path)
    inputs = protocol["inputs"]
    fine = protocol["fine_tuning"]
    output = Path(output_root or Path(protocol_path).parent)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    seed = int(fine["seed"])
    evidence: dict[int, dict[str, Any]] = {}
    source_actual: list[int] = []
    source_predicted: list[int] = []
    paired_cosines: list[float] = []
    fold_histories = []

    for fold in sorted({sample.fold for sample in samples}):
        fold_root = output / f"fold_{fold}"
        fold_root.mkdir(exist_ok=True)
        result_path = fold_root / "result.pt"
        if resume and result_path.exists():
            result = torch.load(result_path, map_location="cpu", weights_only=False)
            for row in result["predictions"]:
                evidence[int(row["source_index"])] = row
            source_actual.extend(result["source_diagnostics"]["actual"])
            source_predicted.extend(result["source_diagnostics"]["predicted"])
            paired_cosines.extend(result["source_diagnostics"]["cosines"])
            fold_histories.append(result["history"])
            print(f"phase32 fold={fold} resumed", flush=True)
            continue

        fold_seed = seed + fold
        _set_seed(fold_seed)
        train_samples = [sample for sample in samples if sample.fold != fold]
        held_samples = [sample for sample in samples if sample.fold == fold]
        checkpoint_path = inputs["phase30_encoder_pattern"].format(fold=fold)
        model = _load_initialized_model(checkpoint_path, fold, protocol)
        history, trainable = _train_fold(model, train_samples, device, fine, fold_seed)
        held_predictions = _predict_evidence(model, held_samples, device, fine)
        for row in held_predictions:
            evidence[int(row["source_index"])] = row
        diagnostics = _post_fine_tune_source_diagnostics(
            model, samples, device, fine, fold, fold_seed
        )
        source_actual.extend(diagnostics["actual"])
        source_predicted.extend(diagnostics["predicted"])
        paired_cosines.extend(diagnostics["cosines"])
        fold_history = {
            "fold": fold,
            "train_rows": len(train_samples),
            "held_rows": len(held_samples),
            "trainable_parameters": int(
                sum(
                    value.numel() for value in model.parameters() if value.requires_grad
                )
            ),
            "trainable_parameter_names": trainable,
            "epochs": history,
        }
        fold_histories.append(fold_history)
        torch.save(
            {
                "predictions": held_predictions,
                "source_diagnostics": diagnostics,
                "history": fold_history,
            },
            result_path,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "fold": fold,
                "seed": fold_seed,
                "config": fine,
                "trainable_parameter_names": trainable,
            },
            fold_root / "fine_tuned_model.pt",
        )
        print(f"phase32 fold={fold} complete", flush=True)

    expected_indices = {sample.source_index for sample in samples}
    if set(evidence) != expected_indices:
        raise RuntimeError("Phase 32 did not predict every ensemble-covered row.")
    ordered = sorted(baseline)
    labels = np.asarray([int(baseline[index]["label"]) for index in ordered])
    folds = np.asarray([int(baseline[index]["fold"]) for index in ordered])
    lengths = np.asarray([len(baseline[index]["sequence"]) for index in ordered])
    reference_probabilities = np.asarray(
        [float(baseline[index]["toxicity_probability"]) for index in ordered]
    )
    candidate_probabilities = reference_probabilities.copy()
    deltas = np.zeros(len(ordered), dtype=np.float64)
    standalone = np.full(len(ordered), np.nan, dtype=np.float64)
    positions = {source_index: offset for offset, source_index in enumerate(ordered)}
    maximum_delta = float(fine["maximum_absolute_logit_delta"])
    for source_index, value in evidence.items():
        offset = positions[source_index]
        candidate, delta = _combine_held_evidence(
            reference_probabilities[offset],
            float(value["standalone_logit"]),
            maximum_delta,
        )
        candidate_probabilities[offset] = candidate
        deltas[offset] = delta
        standalone[offset] = float(value["standalone_probability"])

    threshold = float(protocol["evaluation"]["decision_threshold"])
    reference_metrics = _metrics(labels, reference_probabilities, threshold)
    candidate_metrics = _metrics(labels, candidate_probabilities, threshold)
    covered = ~np.isnan(standalone)
    standalone_metrics = _metrics(labels[covered], standalone[covered], 0.5)
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
        value["candidate"]["mcc"]
        >= value["reference"]["mcc"] - float(gates["maximum_length_mcc_regression"])
        for value in by_length.values()
    )
    length_auprc_ok = all(
        value["candidate"]["auprc"]
        >= value["reference"]["auprc"] - float(gates["maximum_length_auprc_regression"])
        for value in by_length.values()
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
        value = evidence.get(source_index)
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
                "length": int(lengths[offset]),
                "ensemble_available": int(value is not None),
                "reference_probability": float(reference_probabilities[offset]),
                "standalone_3d_probability": (
                    float(standalone[offset]) if value is not None else ""
                ),
                "bounded_logit_delta": float(deltas[offset]),
                "candidate_probability": float(candidate_probabilities[offset]),
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
        "cohort": observed,
        "overall": {
            "reference": reference_metrics,
            "candidate": candidate_metrics,
            "standalone_covered_3d": standalone_metrics,
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
    promotion = bool(gate["all_conditions_passed"])
    decision = {
        "status": "complete",
        "decision": (
            "promote_phase32_3d_handoff" if promotion else "retain_phase25_3d_handoff"
        ),
        "promotion_authorized": promotion,
        "handoff_regeneration_authorized": promotion,
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
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_phase32(args.protocol), ensure_ascii=False))
        return
    _, decision = run_phase32(
        args.protocol,
        args.output_root,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps({"decision": decision["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
