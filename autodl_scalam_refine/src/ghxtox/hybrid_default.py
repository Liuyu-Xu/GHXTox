"""Inference for the public GHXTox ProtT5-plus-3D default.

The default requires a complete pLDDT-aware 3D chemical-site cache and the
residue-aligned cross-modal adapter cache. The pure ProtT5 expert remains
available through ``ghxtox-prott5-predict`` as an explicit ablation/reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ghxtox.cross_modal_adapter import (
    CoupledFeatureDataset,
    ProtT5StructureAdapter,
    collate_coupled,
)
from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.models import GHXToxModel
from ghxtox.plm_textcnn import PooledPLMTextCNN, _predict, _probability_metrics
from ghxtox.utils import (
    DEFAULT_PUBLIC_CONFIG,
    DEFAULT_PUBLIC_MODEL_NAME,
    load_json,
    move_batch_to_device,
    resolve_device,
    resolve_inference_checkpoint,
    save_json,
)


DEFAULT_CONFIG = DEFAULT_PUBLIC_CONFIG


def validate_public_default_config(config: dict[str, Any]) -> None:
    """Reject configs that cannot represent the public 3D default contract."""

    if config.get("canonical_public_default") is not True:
        raise ValueError("Only the canonical 3D hybrid config may be the public default.")
    if config.get("public_model_name") != DEFAULT_PUBLIC_MODEL_NAME:
        raise ValueError(
            "The public default must be "
            f"{DEFAULT_PUBLIC_MODEL_NAME!r}; use a separate config for ablations."
        )
    structure = config.get("structure_expert", {})
    inference = config.get("inference", {})
    adapter = config.get("interaction_adapter", {})
    fusion = config.get("fusion", {})
    bacc_residual = config.get("bacc_residual", {})
    if not structure.get("required", False):
        raise ValueError("The public default requires the 3D structure expert.")
    if not inference.get("full_3d_required", False):
        raise ValueError("The public default must require complete 3D inputs.")
    if not inference.get("adapter_required", False):
        raise ValueError("The public default must require the cross-modal adapter.")
    if not adapter.get("required", False):
        raise ValueError("The public default must require the interaction adapter.")
    prott5_weight = float(fusion.get("prott5_weight", -1.0))
    structure_weight = float(fusion.get("structure_weight", -1.0))
    if abs(prott5_weight + structure_weight - 1.0) > 1e-8:
        raise ValueError("Public default fusion weights must sum to one.")
    if bacc_residual.get("enabled") is not True:
        raise ValueError("The public default requires the frozen BACC decision residual.")
    if not bacc_residual.get("artifact"):
        raise ValueError("Enabled BACC residual must declare a deployable artifact.")
    if bacc_residual.get("decision_only") is not True:
        raise ValueError("The public BACC residual must be decision-only.")
    if abs(
        float(bacc_residual.get("teacher_threshold", -1.0))
        - float(config.get("prott5_expert", {}).get("standalone_threshold", -2.0))
    ) > 1e-8:
        raise ValueError("BACC teacher threshold must match the ProtT5 threshold.")
    if abs(
        float(bacc_residual.get("base_threshold", -1.0))
        - float(fusion.get("threshold", -2.0))
    ) > 1e-8:
        raise ValueError("BACC base threshold must match the joint-model threshold.")


def combine_probabilities(
    structure_probabilities: np.ndarray,
    prott5_probabilities: np.ndarray,
    *,
    prott5_weight: float,
) -> np.ndarray:
    """Convexly combine aligned structure and ProtT5 probabilities."""

    structure = np.asarray(structure_probabilities, dtype=np.float64)
    prott5 = np.asarray(prott5_probabilities, dtype=np.float64)
    if structure.shape != prott5.shape:
        raise ValueError(
            f"Probability shapes differ: {structure.shape} versus {prott5.shape}."
        )
    if not 0.0 <= float(prott5_weight) <= 1.0:
        raise ValueError("prott5_weight must lie in [0, 1].")
    return float(prott5_weight) * prott5 + (1.0 - float(prott5_weight)) * structure


def validate_alignment(
    structure_sequences: list[str],
    structure_labels: list[int | None],
    prott5_payload: dict[str, Any],
) -> None:
    """Require immutable row, sequence and available-label alignment."""

    prott5_sequences = list(prott5_payload.get("sequences", []))
    if structure_sequences != prott5_sequences:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(structure_sequences, prott5_sequences)
                )
                if left != right
            ),
            min(len(structure_sequences), len(prott5_sequences)),
        )
        raise ValueError(f"3D/ProtT5 sequence alignment mismatch at row {mismatch}.")
    prott5_labels = prott5_payload.get("labels")
    if torch.is_tensor(prott5_labels):
        labels = prott5_labels.long().tolist()
        for index, (left, right) in enumerate(zip(structure_labels, labels)):
            if left is not None and int(left) != int(right):
                raise ValueError(f"3D/ProtT5 label alignment mismatch at row {index}.")


def _predict_structure(
    processed_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    dataset = PeptideTensorDataset(processed_path, require_labels=False)
    checkpoint, selected, _, fallback_used = resolve_inference_checkpoint(
        checkpoint_path, dataset.records, device, requested_threshold=None
    )
    config = checkpoint["config"]
    model = GHXToxModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    required_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
    validate_plm_feature_dim(dataset.records, required_dim, processed_path)
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, collate_fn=collate_peptides
    )
    probabilities: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
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
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            probabilities.append(torch.sigmoid(output["logits"]).float().cpu())
            gates.append(output["global_gate"].float().cpu())
    return {
        "sample_ids": sample_ids,
        "sequences": sequences,
        "labels": labels,
        "probabilities": torch.cat(probabilities).numpy(),
        "gates": torch.cat(gates).numpy(),
        "checkpoint": selected,
        "fallback_used": fallback_used,
    }


def _predict_prott5(
    payload: dict[str, Any],
    checkpoint_dirs: list[str],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    features = payload["features"].float().numpy()
    members = []
    seed_means = []
    for checkpoint_dir in checkpoint_dirs:
        directory = Path(checkpoint_dir)
        seed_members = []
        for fold in range(5):
            checkpoint_path = directory / f"fold{fold}_best_model.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing ProtT5 checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_config = dict(checkpoint["model"])
            model_config["window_sizes"] = tuple(model_config["window_sizes"])
            model = PooledPLMTextCNN(**model_config).to(device)
            model.load_state_dict(checkpoint["model_state"])
            predictions = _predict(model, features, batch_size, device)
            members.append(predictions)
            seed_members.append(predictions)
        seed_means.append(np.mean(np.stack(seed_members, axis=0), axis=0))
    if len(members) != 15:
        raise ValueError(f"Expected 15 ProtT5 fold/seed members, found {len(members)}.")
    return {
        "mean": np.mean(np.stack(members, axis=0), axis=0),
        "seed_means": np.stack(seed_means, axis=1),
    }


def _residual_sequence_features(sequence: str) -> list[float]:
    length = max(len(sequence), 1)
    groups = (
        "KRH",
        "DE",
        "AILMFWVY",
        "FWY",
        "CGP",
        "C",
        "GP",
        "STNQ",
    )
    fractions = [sum(sequence.count(letter) for letter in group) / length for group in groups]
    charge_proxy = (
        sum(sequence.count(letter) for letter in "KRH")
        - sum(sequence.count(letter) for letter in "DE")
    ) / length
    return [float(np.log1p(length)), charge_proxy, *fractions]


def _residual_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def _build_bacc_residual_features(
    teacher: np.ndarray,
    base: np.ndarray,
    baseline: np.ndarray,
    interaction: np.ndarray,
    members: np.ndarray,
    sequences: list[str],
    *,
    teacher_threshold: float,
    base_threshold: float,
) -> np.ndarray:
    member_matrix = np.asarray(members, dtype=np.float64)
    teacher_logit = _residual_logit(teacher)
    base_logit = _residual_logit(base)
    sequence = np.asarray(
        [_residual_sequence_features(value) for value in sequences], dtype=np.float64
    )
    values = np.column_stack(
        [
            teacher,
            base,
            baseline,
            interaction,
            member_matrix,
            member_matrix.std(axis=1),
            member_matrix.min(axis=1),
            member_matrix.max(axis=1),
            (member_matrix >= teacher_threshold).sum(axis=1) / 3.0,
            teacher_logit,
            base_logit,
            teacher_logit - base_logit,
            np.abs(teacher - teacher_threshold),
            np.abs(base - base_threshold),
            sequence,
        ]
    )
    if values.shape[1] != 26 or not np.isfinite(values).all():
        raise ValueError("Invalid BACC residual feature matrix.")
    return values


def _apply_bacc_residual(
    artifact_path: str | Path,
    *,
    teacher: np.ndarray,
    base: np.ndarray,
    baseline: np.ndarray,
    interaction: np.ndarray,
    members: np.ndarray,
    sequences: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    artifact = load_json(artifact_path)
    if artifact.get("artifact_type") != "ghxtox_bacc_decision_residual":
        raise ValueError(f"Unsupported BACC residual artifact: {artifact_path}")
    teacher_threshold = float(artifact["teacher_threshold"])
    base_threshold = float(artifact["base_threshold"])
    gate_threshold = float(artifact["gate_threshold"])
    features = _build_bacc_residual_features(
        teacher,
        base,
        baseline,
        interaction,
        members,
        sequences,
        teacher_threshold=teacher_threshold,
        base_threshold=base_threshold,
    )
    scaler = artifact["scaler"]
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    if mean.shape != (features.shape[1],) or scale.shape != mean.shape:
        raise ValueError("BACC residual scaler does not match the feature schema.")
    hidden = (features - mean) / np.where(scale == 0.0, 1.0, scale)
    model = artifact["model"]
    coefs = [np.asarray(value, dtype=np.float64) for value in model["coefs"]]
    intercepts = [np.asarray(value, dtype=np.float64) for value in model["intercepts"]]
    for index, (coef, intercept) in enumerate(zip(coefs, intercepts, strict=True)):
        hidden = hidden @ coef + intercept
        if index < len(coefs) - 1:
            hidden = np.maximum(hidden, 0.0)
    gate_logit = np.clip(hidden.reshape(-1), -60.0, 60.0)
    gate_probability = 1.0 / (1.0 + np.exp(-gate_logit))
    disagreement = (base >= base_threshold) != (teacher >= teacher_threshold)
    gate_probability = np.where(disagreement, gate_probability, 0.0)
    active = disagreement & (gate_probability >= gate_threshold)
    prediction = base >= base_threshold
    prediction[active] = teacher[active] >= teacher_threshold
    return gate_probability, active, prediction


def _metrics_with_decision(
    labels: np.ndarray,
    probabilities: np.ndarray,
    decision: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    metrics = _probability_metrics(labels, probabilities, threshold)
    labels = np.asarray(labels, dtype=np.int64)
    decision = np.asarray(decision, dtype=bool)
    tp = int(np.sum(decision & (labels == 1)))
    tn = int(np.sum(~decision & (labels == 0)))
    fp = int(np.sum(decision & (labels == 0)))
    fn = int(np.sum(~decision & (labels == 1)))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    denominator = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    metrics.update(
        {
            "accuracy": (tp + tn) / max(len(labels), 1),
            "balanced_accuracy": 0.5 * (recall + specificity),
            "precision": precision,
            "sensitivity": recall,
            "sn": recall,
            "recall": recall,
            "specificity": specificity,
            "sp": specificity,
            "f1": f1,
            "mcc": (tp * tn - fp * fn) / denominator,
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
        }
    )
    return metrics


def _apply_interaction_adapter(
    probabilities: np.ndarray,
    sequences: list[str],
    feature_cache_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
    if list(payload.get("sequences", [])) != list(sequences):
        raise ValueError("Hybrid/interaction-adapter sequence alignment mismatch.")
    baseline = torch.as_tensor(probabilities, dtype=torch.float32)
    if baseline.numel() != len(sequences):
        raise ValueError("Hybrid probabilities do not cover the adapter feature cache.")
    payload["base_probabilities"] = baseline
    dataset = CoupledFeatureDataset(payload)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ProtT5StructureAdapter(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_coupled,
    )
    adapted = []
    deltas = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(
                tokens=batch["tokens"],
                spatial=batch["spatial"],
                plddt=batch["plddt"],
                mask=batch["mask"],
                base_probability=batch["base_probability"],
                chemical=batch.get("chemical"),
            )
            adapted.append(torch.sigmoid(output["logits"]).cpu())
            deltas.append(output["interaction_delta"].cpu())
    return torch.cat(adapted).numpy(), torch.cat(deltas).numpy()


def predict_hybrid(
    config_path: str | Path,
    processed_path: str | Path,
    prott5_features_path: str | Path,
    output_path: str | Path,
    *,
    metrics_path: str | Path | None,
    device_name: str,
    batch_size: int,
    adapter_feature_cache_path: str | Path | None = None,
    allow_pre_adapter: bool = False,
    allow_sequence_fallback: bool = False,
    disable_bacc_residual: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path)
    validate_public_default_config(config)
    device = resolve_device(device_name)
    structure = _predict_structure(
        processed_path,
        config["structure_expert"]["checkpoint"],
        device=device,
        batch_size=batch_size,
    )
    prott5_payload = torch.load(
        prott5_features_path, map_location="cpu", weights_only=False
    )
    validate_alignment(structure["sequences"], structure["labels"], prott5_payload)
    prott5_outputs = _predict_prott5(
        prott5_payload,
        list(config["prott5_expert"]["checkpoint_dirs"]),
        device=device,
        batch_size=batch_size,
    )
    prott5_probabilities = prott5_outputs["mean"]
    prott5_seed_probabilities = prott5_outputs["seed_means"]

    fallback_used = bool(structure["fallback_used"])
    inference_config = config.get("inference", {})
    if (
        fallback_used
        and bool(inference_config.get("full_3d_required", False))
        and not allow_sequence_fallback
    ):
        raise RuntimeError(
            "The public 3D default requires complete chemical-site tensors. "
            "Use a *_chemical_sites_final_esm2.pt cache, or pass "
            "--allow-sequence-fallback explicitly."
        )
    adapter_applied = False
    interaction_delta = np.zeros(len(prott5_probabilities), dtype=np.float64)
    baseline_probability = np.zeros(len(prott5_probabilities), dtype=np.float64)
    if fallback_used and (
        config["inference"].get("fallback_mode") == "prott5_only"
        or allow_sequence_fallback
    ):
        probabilities = prott5_probabilities.astype(np.float64)
        baseline_probability = probabilities.copy()
        threshold = float(config["prott5_expert"]["standalone_threshold"])
        prott5_weight = 1.0
    else:
        prott5_weight = float(config["fusion"]["prott5_weight"])
        probabilities = combine_probabilities(
            structure["probabilities"],
            prott5_probabilities,
            prott5_weight=prott5_weight,
        )
        baseline_probability = probabilities.copy()
        threshold = float(config["fusion"]["threshold"])
        adapter_config = config.get("interaction_adapter")
        adapter_required = bool(inference_config.get("adapter_required", False))
        if adapter_required and adapter_feature_cache_path is None and not allow_pre_adapter:
            raise RuntimeError(
                "The public 3D default requires --adapter-feature-cache. "
                "Pass --allow-pre-adapter only for an explicit ablation."
            )
        if adapter_config and adapter_feature_cache_path is not None:
            probabilities, interaction_delta = _apply_interaction_adapter(
                probabilities,
                structure["sequences"],
                adapter_feature_cache_path,
                adapter_config["checkpoint"],
                device=device,
                batch_size=batch_size,
            )
            adapter_applied = True

    bacc_gate_probability = np.zeros(len(probabilities), dtype=np.float64)
    bacc_residual_active = np.zeros(len(probabilities), dtype=bool)
    base_prediction = probabilities >= threshold
    prediction = base_prediction.copy()
    bacc_config = config.get("bacc_residual", {})
    bacc_residual_applied = False
    if (
        not disable_bacc_residual
        and bool(bacc_config.get("enabled", False))
        and not fallback_used
        and adapter_applied
    ):
        artifact_path = bacc_config.get("artifact")
        if not artifact_path:
            raise ValueError("BACC residual is enabled but no artifact was configured.")
        (
            bacc_gate_probability,
            bacc_residual_active,
            prediction,
        ) = _apply_bacc_residual(
            artifact_path,
            teacher=prott5_probabilities,
            base=probabilities,
            baseline=baseline_probability,
            interaction=interaction_delta,
            members=prott5_seed_probabilities,
            sequences=structure["sequences"],
        )
        bacc_residual_applied = True

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_index",
        "sample_id",
        "sequence",
        "label",
        "structure_probability",
        "prott5_probability",
        "member0_probability",
        "member1_probability",
        "member2_probability",
        "baseline_probability",
        "toxicity_probability",
        "base_prediction",
        "prediction",
        "decision_threshold",
        "prott5_weight",
        "global_3d_gate",
        "fallback_used",
        "interaction_delta",
        "interaction_adapter_applied",
        "bacc_gate_probability",
        "bacc_residual_active",
        "bacc_residual_applied",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, probability in enumerate(probabilities):
            writer.writerow(
                {
                    "source_index": index,
                    "sample_id": structure["sample_ids"][index],
                    "sequence": structure["sequences"][index],
                    "label": "" if structure["labels"][index] is None else structure["labels"][index],
                    "structure_probability": f"{float(structure['probabilities'][index]):.10g}",
                    "prott5_probability": f"{float(prott5_probabilities[index]):.10g}",
                    "member0_probability": f"{float(prott5_seed_probabilities[index, 0]):.10g}",
                    "member1_probability": f"{float(prott5_seed_probabilities[index, 1]):.10g}",
                    "member2_probability": f"{float(prott5_seed_probabilities[index, 2]):.10g}",
                    "baseline_probability": f"{float(baseline_probability[index]):.10g}",
                    "toxicity_probability": f"{float(probability):.10g}",
                    "base_prediction": int(base_prediction[index]),
                    "prediction": int(prediction[index]),
                    "decision_threshold": f"{threshold:.10g}",
                    "prott5_weight": f"{prott5_weight:.10g}",
                    "global_3d_gate": f"{float(structure['gates'][index]):.10g}",
                    "fallback_used": int(fallback_used),
                    "interaction_delta": f"{float(interaction_delta[index]):.10g}",
                    "interaction_adapter_applied": int(adapter_applied),
                    "bacc_gate_probability": f"{float(bacc_gate_probability[index]):.10g}",
                    "bacc_residual_active": int(bacc_residual_active[index]),
                    "bacc_residual_applied": int(bacc_residual_applied),
                }
            )

    available_labels = [label for label in structure["labels"] if label is not None]
    metrics = None
    if len(available_labels) == len(structure["labels"]):
        metrics = _metrics_with_decision(
            np.asarray(available_labels, dtype=np.int64),
            probabilities,
            prediction,
            threshold,
        )
    summary = {
        "model": config["name"],
        "protocol": {
            "config": str(config_path),
            "processed": str(processed_path),
            "prott5_features": str(prott5_features_path),
            "structure_checkpoint": structure["checkpoint"],
            "num_prott5_members": 15,
            "prott5_weight": prott5_weight,
            "structure_weight": 1.0 - prott5_weight,
            "threshold": threshold,
            "fallback_used": fallback_used,
            "interaction_adapter_checkpoint": (
                config.get("interaction_adapter", {}).get("checkpoint")
                if adapter_applied
                else None
            ),
            "interaction_adapter_features": (
                str(adapter_feature_cache_path) if adapter_applied else None
            ),
            "interaction_adapter_applied": adapter_applied,
            "bacc_residual_artifact": (
                str(config.get("bacc_residual", {}).get("artifact"))
                if bacc_residual_applied
                else None
            ),
            "bacc_residual_applied": bacc_residual_applied,
            "bacc_residual_active_rows": int(bacc_residual_active.sum()),
            "bacc_residual_disables_flag": disable_bacc_residual,
            "allow_pre_adapter": allow_pre_adapter,
            "allow_sequence_fallback": allow_sequence_fallback,
            "parameters_selected_from_test_labels": False,
        },
        "metrics": metrics,
    }
    if metrics_path is not None:
        save_json(summary, metrics_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--prott5-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--adapter-feature-cache",
        help=(
            "Residue-aligned ProtT5/3D/chemical feature cache required by the public default. "
            "Omit only with --allow-pre-adapter for an explicit ablation."
        ),
    )
    parser.add_argument(
        "--allow-pre-adapter",
        action="store_true",
        help="Explicitly allow the lower-performing pre-adapter 1D+3D ablation.",
    )
    parser.add_argument(
        "--allow-sequence-fallback",
        action="store_true",
        help="Explicitly allow sequence-only fallback when 3D tensors are unavailable.",
    )
    parser.add_argument(
        "--disable-bacc-residual",
        action="store_true",
        help="Disable the frozen OOF-selected BACC decision residual for an ablation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predict_hybrid(
        args.config,
        args.processed,
        args.prott5_features,
        args.output,
        metrics_path=args.metrics,
        device_name=args.device,
        batch_size=args.batch_size,
        adapter_feature_cache_path=args.adapter_feature_cache,
        allow_pre_adapter=args.allow_pre_adapter,
        allow_sequence_fallback=args.allow_sequence_fallback,
        disable_bacc_residual=args.disable_bacc_residual,
    )


if __name__ == "__main__":
    main()
