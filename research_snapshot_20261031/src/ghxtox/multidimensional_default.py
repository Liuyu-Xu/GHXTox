# -*- coding: utf-8 -*-
"""Canonical GHXTox public inference: calibrated LoRA-BiMamba 1D + single-structure 3D.

The module enforces the frozen public contract instead of silently accepting
partial sample intersections, old checkpoints, mixed-generation OOF files, or
non-canonical weights.  Use ``--allow-noncanonical-operating-point`` only for
an explicitly labelled ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.lora_mamba_1d import (
    PUBLIC_1D_EXPERIMENT,
    PUBLIC_1D_SEEDS,
    validate_public_checkpoint_payload,
)
from ghxtox.selective_prediction import apply_selective_policy, load_selective_policy
from ghxtox.utils import (
    DEFAULT_MULTIDIMENSIONAL_CONFIG,
    DEFAULT_MULTIDIMENSIONAL_MODEL_NAME,
    DEFAULT_MULTIDIMENSIONAL_THRESHOLD,
    DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D,
)


DEFAULT_CHECKPOINTS = tuple(
    f"runs/lora_mamba_full/full_seed{seed}.pt" for seed in PUBLIC_1D_SEEDS
)
DEFAULT_OOF_JSONS = tuple(
    f"runs/lora_mamba_full/oof_seed{seed}.json" for seed in PUBLIC_1D_SEEDS
)
DEFAULT_MANIFEST = "data/folds/train_cdhit080_fallback_folds.csv"
DEFAULT_STRUCTURE_CHECKPOINT = (
    "runs/multisource_1_30_full2795_rank_stable_candidate/best_model.pt"
)
DEFAULT_1D_CONFIG = "configs/lora_mamba_1d_autodl.json"
LEGACY_STRUCTURE_COLUMNS = {
    "prott5_probability",
    "interaction_delta",
    "interaction_adapter_applied",
    "bacc_gate_probability",
    "bacc_residual_active",
    "bacc_residual_applied",
}
RETIRED_TEST_CONFORMER_COLUMNS = {
    "base_structure_probability",
    "conformer_available",
    "conformer_logit_delta",
}


def _logit(probabilities: Any) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def fit_platt(logits: Any, labels: Any) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(
        np.asarray(logits, dtype=np.float64).reshape(-1, 1),
        np.asarray(labels, dtype=np.int64),
    )
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        raise ValueError("Platt calibration produced non-finite parameters.")
    if slope <= 0.0:
        raise ValueError("Platt calibration must be monotone increasing.")
    return slope, intercept


def validate_public_default_config(config: dict[str, Any]) -> None:
    """Validate the one and only public-default configuration."""

    if config.get("canonical_public_default") is not True:
        raise ValueError("The multidimensional configuration must be the canonical public default.")
    if config.get("public_model_name") != DEFAULT_MULTIDIMENSIONAL_MODEL_NAME:
        raise ValueError("Unexpected public model name.")
    if config.get("entrypoint") != "ghxtox-predict":
        raise ValueError("The canonical entrypoint must be ghxtox-predict.")
    fusion = config.get("fusion", {})
    weight_1d = float(fusion.get("weight_1d", -1.0))
    weight_3d = float(fusion.get("weight_3d", -1.0))
    threshold = float(fusion.get("threshold", -1.0))
    if abs(weight_1d - DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D) > 1e-12:
        raise ValueError("Canonical 1D weight differs from the frozen OOF value.")
    if abs(weight_1d + weight_3d - 1.0) > 1e-12:
        raise ValueError("Canonical fusion weights must sum to one.")
    if abs(threshold - DEFAULT_MULTIDIMENSIONAL_THRESHOLD) > 1e-12:
        raise ValueError("Canonical decision threshold differs from the frozen OOF value.")
    expert = config.get("composition", {}).get("1d_expert", {})
    if tuple(expert.get("seeds", ())) != PUBLIC_1D_SEEDS:
        raise ValueError(f"Canonical 1D seeds must be {PUBLIC_1D_SEEDS}.")
    if expert.get("experiment_name") != PUBLIC_1D_EXPERIMENT:
        raise ValueError("Canonical 1D experiment name is missing or stale.")
    if config.get("provenance", {}).get("three_d_oof") != (
        "runs/conformer_ensemble_residual_oof/full_calibrated_predictions.csv"
    ):
        raise ValueError("Canonical operating-point 3D OOF provenance is missing or stale.")
    test_policy = config.get("test_time_conformer_policy", {})
    if test_policy.get("status") != "disabled":
        raise ValueError("Canonical public inference must disable test-time conformer ensembles.")
    if test_policy.get("probability_source") != "base_logits":
        raise ValueError("Canonical public inference must use the single-structure base logits.")
    structure = config.get("composition", {}).get("3d_expert", {})
    if structure.get("checkpoint") != DEFAULT_STRUCTURE_CHECKPOINT:
        raise ValueError("Canonical 3D checkpoint is missing or stale.")
    probability_source = str(structure.get("test_probability_column", "")).lower()
    if "default_3d_hybrid" in probability_source or "legacy" in probability_source:
        raise ValueError("Canonical structure probabilities must not come from a legacy container.")
    if "conformer" in probability_source or "source_aware" in probability_source:
        raise ValueError("Canonical test probabilities must not come from a conformer-enabled export.")


def _validate_operating_point(weight: float, threshold: float, allow_noncanonical: bool) -> None:
    if not 0.0 <= float(weight) <= 1.0 or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Fusion weight and threshold must lie in [0, 1].")
    if not allow_noncanonical and (
        abs(float(weight) - DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D) > 1e-12
        or abs(float(threshold) - DEFAULT_MULTIDIMENSIONAL_THRESHOLD) > 1e-12
    ):
        raise ValueError(
            "Refusing a non-canonical operating point. Pass "
            "--allow-noncanonical-operating-point only for a labelled ablation."
        )


def _read_probability_csv(path: str | Path, probability_key: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "label", probability_key}
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"{path}: missing columns {sorted(missing_columns)}.")
        for row_number, row in enumerate(reader, start=2):
            sample_id = str(row["sample_id"])
            if not sample_id or sample_id in records:
                raise ValueError(f"{path}:{row_number}: empty or duplicate sample_id {sample_id!r}.")
            probability = float(row[probability_key])
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{path}:{row_number}: invalid probability {probability!r}.")
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"{path}:{row_number}: label must be 0 or 1.")
            records[sample_id] = {
                "probability": probability,
                "label": label,
                "sequence": str(row.get("sequence", "")),
            }
            order.append(sample_id)
    if not order:
        raise ValueError(f"{path}: no prediction rows.")
    return order, records


def _read_optional_aligned_column(
    path: str | Path, column: str, sample_ids: list[str]
) -> np.ndarray | None:
    """Read an optional diagnostic column without weakening sample-ID alignment."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in set(reader.fieldnames or ()):
            return None
        values: dict[str, float] = {}
        for row_number, row in enumerate(reader, start=2):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in values:
                raise ValueError(f"{path}:{row_number}: empty or duplicate sample_id {sample_id!r}.")
            text = str(row.get(column, "")).strip()
            values[sample_id] = float(text) if text else float("nan")
    if set(values) != set(sample_ids):
        raise ValueError(f"{path}: optional column {column} does not match canonical sample coverage.")
    return np.asarray([values[sample_id] for sample_id in sample_ids], dtype=np.float64)


def _reject_legacy_structure_csv(path: str | Path) -> None:
    """Prevent the public assembler from consuming retired/noncanonical outputs."""

    source = Path(path)
    normalized = str(source).replace("\\", "/").lower()
    if any(
        token in normalized
        for token in (
            "default_3d_hybrid",
            "/_legacy/",
            "hybrid_default",
            "test_multiconformer_archive",
            "multiconformer_extension",
            "source_aware_multiconformer",
        )
    ):
        raise ValueError(
            "The public multidimensional assembler refuses a legacy or retired test-time "
            "multi-conformer prediction path. Use a current single-structure 3D CSV."
        )
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        fields = set(next(csv.reader(handle), []))
    leaked = sorted(fields & LEGACY_STRUCTURE_COLUMNS)
    if leaked:
        raise ValueError(
            "The structure CSV contains previous-hybrid model columns and cannot enter the "
            f"public default: {leaked}."
        )
    retired = sorted(fields & RETIRED_TEST_CONFORMER_COLUMNS)
    if retired:
        raise ValueError(
            "The structure CSV contains retired test-time multi-conformer columns and cannot "
            f"enter the public default: {retired}."
        )


def _aligned_components(calibrated_1d: str | Path, structure_csv: str | Path):
    _reject_legacy_structure_csv(structure_csv)
    ids, one_d = _read_probability_csv(calibrated_1d, "toxicity_probability")
    structure_ids, three_d = _read_probability_csv(structure_csv, "structure_probability")
    if set(ids) != set(structure_ids):
        missing_3d = sorted(set(ids) - set(structure_ids))[:5]
        extra_3d = sorted(set(structure_ids) - set(ids))[:5]
        raise ValueError(
            "1D/3D sample coverage differs; "
            f"missing_3d={missing_3d}, extra_3d={extra_3d}."
        )
    for sample_id in ids:
        if one_d[sample_id]["label"] != three_d[sample_id]["label"]:
            raise ValueError(f"1D/3D label mismatch for {sample_id}.")
        left_sequence = one_d[sample_id]["sequence"]
        right_sequence = three_d[sample_id]["sequence"]
        if left_sequence and right_sequence and left_sequence != right_sequence:
            raise ValueError(f"1D/3D sequence mismatch for {sample_id}.")
    labels = np.asarray([one_d[sample_id]["label"] for sample_id in ids], dtype=np.int64)
    p_1d = np.asarray([one_d[sample_id]["probability"] for sample_id in ids])
    p_3d = np.asarray([three_d[sample_id]["probability"] for sample_id in ids])
    return ids, labels, p_1d, p_3d


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    prediction = (probabilities >= threshold).astype(int)
    true_negative = int(np.sum((prediction == 0) & (labels == 0)))
    false_positive = int(np.sum((prediction == 1) & (labels == 0)))
    specificity = true_negative / max(true_negative + false_positive, 1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "sensitivity": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def _write_predictions(
    output_csv: str | Path,
    sample_ids: list[str],
    labels: np.ndarray,
    p_1d: np.ndarray,
    p_3d: np.ndarray,
    fused: np.ndarray,
    threshold: float,
    weight_1d: float,
    *,
    sequences: list[str] | None = None,
    p_3d_base: np.ndarray | None = None,
    conformer_available: np.ndarray | None = None,
    conformer_delta: np.ndarray | None = None,
    selective_result: dict[str, np.ndarray] | None = None,
) -> None:
    prediction = (fused >= threshold).astype(int)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "source_index", "sample_id", "label", "p_1d", "p_3d",
            "toxicity_probability", "prediction", "decision_threshold", "weight_1d",
        ]
        if sequences is not None:
            header.insert(2, "sequence")
        if p_3d_base is not None:
            header.extend(["p_3d_base", "conformer_available", "conformer_logit_delta"])
        if selective_result is not None:
            required = {"uncertainty_score", "abstained", "selective_prediction", "binary_prediction"}
            if required - set(selective_result):
                raise ValueError("Selective-prediction result is incomplete.")
            if any(np.asarray(selective_result[key]).shape != prediction.shape for key in required):
                raise ValueError("Selective-prediction outputs are not row-aligned.")
            if not np.array_equal(
                np.asarray(selective_result["binary_prediction"], dtype=np.int64), prediction
            ):
                raise ValueError("Selective policy changed the retained binary prediction.")
            header.extend(["uncertainty_score", "abstained", "selective_prediction"])
        writer.writerow(header)
        for index, sample_id in enumerate(sample_ids):
            row: list[Any] = [
                index, sample_id, int(labels[index]), f"{p_1d[index]:.10g}",
                f"{p_3d[index]:.10g}", f"{fused[index]:.10g}", int(prediction[index]),
                f"{threshold:.10g}", f"{weight_1d:.6g}",
            ]
            if sequences is not None:
                row.insert(2, sequences[index])
            if p_3d_base is not None:
                row.extend(
                    [
                        f"{p_3d_base[index]:.10g}",
                        int(bool(conformer_available[index])) if conformer_available is not None else 0,
                        f"{conformer_delta[index]:.10g}" if conformer_delta is not None else "0",
                    ]
                )
            if selective_result is not None:
                row.extend(
                    [
                        f"{float(selective_result['uncertainty_score'][index]):.10g}",
                        int(bool(selective_result["abstained"][index])),
                        str(selective_result["selective_prediction"][index]),
                    ]
                )
            writer.writerow(row)


def _apply_selective_prediction(
    policy_path: str | Path,
    fused: np.ndarray,
    p_1d: np.ndarray,
    p_3d: np.ndarray,
    threshold: float,
    **diagnostics: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    policy = load_selective_policy(policy_path)
    if abs(float(policy["decision_threshold"]) - float(threshold)) > 1e-12:
        raise ValueError(
            "Selective policy and binary classifier use different decision thresholds."
        )
    result = apply_selective_policy(fused, p_1d, p_3d, policy, **diagnostics)
    accepted = ~np.asarray(result["abstained"], dtype=bool)
    summary = {
        "enabled": True,
        "policy": str(policy_path),
        "profile": str(policy.get("profile", "unknown")),
        "target_coverage": float(policy.get("target_coverage", np.nan)),
        "realized_coverage": float(np.mean(accepted)),
        "accepted_n": int(np.sum(accepted)),
        "abstained_n": int(np.sum(~accepted)),
        "retained_binary_probability_and_prediction": True,
    }
    return result, summary


def assemble(
    weight: float,
    threshold: float,
    cal_1d: str | Path,
    structure_csv: str | Path,
    output_csv: str | Path,
    output_metrics: str | Path | None = None,
    *,
    allow_noncanonical: bool = False,
    selective_policy: str | Path | None = None,
) -> dict[str, Any]:
    _validate_operating_point(weight, threshold, allow_noncanonical)
    ids, labels, p_1d, p_3d = _aligned_components(cal_1d, structure_csv)
    fused = float(weight) * p_1d + (1.0 - float(weight)) * p_3d
    selective_result = None
    selective_summary = {"enabled": False}
    if selective_policy is not None:
        selective_result, selective_summary = _apply_selective_prediction(
            selective_policy,
            fused,
            p_1d,
            p_3d,
            threshold,
            one_d_seed_std=_read_optional_aligned_column(cal_1d, "one_d_seed_std", ids),
            mean_plddt=_read_optional_aligned_column(structure_csv, "mean_plddt", ids),
            mean_residue_rmsf=_read_optional_aligned_column(
                structure_csv, "mean_residue_rmsf", ids
            ),
            mean_conformer_rmsd=_read_optional_aligned_column(
                structure_csv, "mean_conformer_rmsd", ids
            ),
            contact_uncertainty_8=_read_optional_aligned_column(
                structure_csv, "contact_uncertainty_8", ids
            ),
            conformer_missing=_read_optional_aligned_column(
                structure_csv, "conformer_missing", ids
            ),
        )
    _write_predictions(
        output_csv,
        ids,
        labels,
        p_1d,
        p_3d,
        fused,
        threshold,
        weight,
        selective_result=selective_result,
    )
    result_metrics = _metrics(labels, fused, threshold)
    if output_metrics:
        destination = Path(output_metrics)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"n": len(ids), "metrics": result_metrics, "selective_prediction": selective_summary}


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_index", "sample_id", "sequence", "label", "fold"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"{path}: fold manifest lacks {sorted(required)}.")
    indices = [int(row["source_index"]) for row in rows]
    ids = [str(row["sample_id"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError(f"{path}: source_index must be contiguous and row-aligned.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate sample_id values.")
    return rows


def _load_oof_ensemble(
    oof_jsons: list[str] | tuple[str, ...], manifest: str | Path
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    if len(oof_jsons) != len(PUBLIC_1D_SEEDS):
        raise ValueError(f"Exactly {len(PUBLIC_1D_SEEDS)} seed OOF files are required.")
    rows = _read_manifest(manifest)
    expected_ids = [str(row["sample_id"]) for row in rows]
    expected_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    expected_folds = [int(row["fold"]) for row in rows]
    probabilities: list[np.ndarray] = []
    seeds: list[int] = []
    for path in oof_jsons:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        seed = int(payload.get("seed", -1))
        if seed in seeds:
            raise ValueError(f"{path}: duplicate OOF seed {seed}.")
        if payload.get("config_name") != PUBLIC_1D_EXPERIMENT:
            raise ValueError(f"{path}: stale or unknown 1D OOF generation.")
        if [str(value) for value in payload.get("sample_ids", [])] != expected_ids:
            raise ValueError(f"{path}: OOF sample IDs/order differ from the fold manifest.")
        labels = np.asarray(payload.get("labels", []), dtype=np.int64)
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f"{path}: OOF labels differ from the fold manifest.")
        if [int(value) for value in payload.get("folds", [])] != expected_folds:
            raise ValueError(f"{path}: OOF folds differ from the fold manifest.")
        values = np.asarray(payload.get("oof", []), dtype=np.float64)
        if values.shape != expected_labels.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: invalid OOF probability vector.")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{path}: OOF probabilities lie outside [0, 1].")
        probabilities.append(values)
        seeds.append(seed)
    if tuple(sorted(seeds)) != PUBLIC_1D_SEEDS:
        raise ValueError(f"OOF seeds must be exactly {PUBLIC_1D_SEEDS}; found {sorted(seeds)}.")
    return np.mean(np.stack(probabilities), axis=0), expected_labels, tuple(seeds)


def _fit_platt_from_oof(
    oof_jsons: list[str] | tuple[str, ...], manifest: str | Path
) -> tuple[float, float, tuple[int, ...]]:
    mean_probability, labels, seeds = _load_oof_ensemble(oof_jsons, manifest)
    slope, intercept = fit_platt(_logit(mean_probability), labels)
    return slope, intercept, seeds


def _predict_1d_seed(records, checkpoint, config, device, tokenizer, expected_seed):
    import torch
    from ghxtox.lora_mamba_1d import (
        LoraMambaCollator,
        LoraMambaDataset,
        _load_plm,
        build_model,
        predict_probs,
        remove_inactive_lora,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_public_checkpoint_payload(
        payload, config, expected_seed=expected_seed, checkpoint_path=checkpoint
    )
    plm = _load_plm(config, device)
    model = build_model(plm, config, device)
    validate_public_checkpoint_payload(
        payload, config, expected_seed=expected_seed, model=model, checkpoint_path=checkpoint
    )
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    expected_missing = set(model.state_dict()) - set(payload["model_state"])
    if unexpected or set(missing) != expected_missing:
        raise ValueError(
            f"{checkpoint}: unsafe partial load; missing={missing[:5]}, unexpected={unexpected[:5]}."
        )
    remove_inactive_lora(model.plm.encoder)
    collator = LoraMambaCollator(tokenizer, max_length=int(config.get("max_length", 128)))
    probabilities = predict_probs(
        model,
        LoraMambaDataset(records),
        collator,
        device,
        int(config.get("batch_size", 16)),
        bool(config.get("fp16", True)),
    )
    return probabilities


def _validate_sequence_records(records) -> None:
    ids = [str(record.sample_id) for record in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Sequence record cache is empty or contains duplicate sample IDs.")
    for record in records:
        if not record.sequence or int(record.length) != len(record.sequence):
            raise ValueError(f"Invalid sequence record for {record.sample_id}.")
        if record.label is not None and int(record.label) not in (0, 1):
            raise ValueError(f"Invalid label for {record.sample_id}.")


def _canonical_path(path: str | Path) -> Path:
    return Path(path).resolve(strict=False)


def _validate_canonical_predict_assets(
    config_1d: str | Path,
    checkpoints: list[str | Path] | tuple[str | Path, ...],
    oof_jsons: list[str | Path] | tuple[str | Path, ...],
    manifest: str | Path,
    structure_checkpoint: str | Path,
) -> None:
    """Keep the public predictor pinned to the audited current-generation assets."""

    expected_groups = (
        ("1D config", [config_1d], [DEFAULT_1D_CONFIG]),
        ("1D checkpoints", list(checkpoints), list(DEFAULT_CHECKPOINTS)),
        ("1D OOF files", list(oof_jsons), list(DEFAULT_OOF_JSONS)),
        ("fold manifest", [manifest], [DEFAULT_MANIFEST]),
        ("3D checkpoint", [structure_checkpoint], [DEFAULT_STRUCTURE_CHECKPOINT]),
    )
    for label, supplied, expected in expected_groups:
        if [_canonical_path(value) for value in supplied] != [
            _canonical_path(value) for value in expected
        ]:
            raise ValueError(
                f"The public predictor requires the canonical {label}; "
                "use a dedicated research/legacy module for ablations."
            )
    for value in [config_1d, *checkpoints, *oof_jsons, manifest, structure_checkpoint]:
        normalized = str(value).replace("\\", "/").lower()
        if any(token in normalized for token in ("/_legacy/", "hybrid_default", "default_3d_hybrid")):
            raise ValueError("Legacy model assets are forbidden in the public predictor.")


def predict(
    processed_path,
    sequence_records_path,
    config_1d,
    checkpoints,
    oof_jsons,
    manifest,
    structure_checkpoint,
    weight,
    threshold,
    device_name,
    batch_size,
    output_csv,
    output_metrics=None,
    *,
    allow_noncanonical=False,
    selective_policy=None,
):
    import torch
    from transformers import T5Tokenizer

    from ghxtox.lora_mamba_1d import load_test_records
    from ghxtox.structure_inference import predict_structure

    _validate_operating_point(weight, threshold, allow_noncanonical)
    _validate_canonical_predict_assets(
        config_1d, checkpoints, oof_jsons, manifest, structure_checkpoint
    )
    config = json.loads(Path(config_1d).read_text(encoding="utf-8-sig"))
    if config.get("experiment", {}).get("name") != PUBLIC_1D_EXPERIMENT:
        raise ValueError("The 1D config is not the canonical LoRA-BiMamba experiment.")
    if len(checkpoints) != len(PUBLIC_1D_SEEDS):
        raise ValueError(f"Exactly {len(PUBLIC_1D_SEEDS)} full checkpoints are required.")
    slope, intercept, oof_seeds = _fit_platt_from_oof(oof_jsons, manifest)
    device = torch.device(
        device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = T5Tokenizer.from_pretrained(
        str(config.get("plm", "Rostlab/prot_t5_xl_half_uniref50-enc")),
        do_lower_case=False,
    )
    records = load_test_records(sequence_records_path)
    _validate_sequence_records(records)
    checkpoint_payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in checkpoints
    ]
    checkpoint_seeds = tuple(int(payload.get("seed", -1)) for payload in checkpoint_payloads)
    if tuple(sorted(checkpoint_seeds)) != PUBLIC_1D_SEEDS or len(set(checkpoint_seeds)) != len(checkpoint_seeds):
        raise ValueError(
            f"Checkpoint seeds must be exactly {PUBLIC_1D_SEEDS}; found {checkpoint_seeds}."
        )
    if tuple(sorted(oof_seeds)) != tuple(sorted(checkpoint_seeds)):
        raise ValueError("Checkpoint and OOF seed generations do not correspond.")
    seed_probabilities = [
        _predict_1d_seed(records, path, config, device, tokenizer, seed)
        for path, seed in zip(checkpoints, checkpoint_seeds)
    ]
    stacked_seed_probabilities = np.stack(seed_probabilities)
    ensemble_probability = np.mean(stacked_seed_probabilities, axis=0)
    one_d_seed_std = np.std(stacked_seed_probabilities, axis=0, ddof=1)
    p_1d = 1.0 / (1.0 + np.exp(-(slope * _logit(ensemble_probability) + intercept)))

    structure = predict_structure(
        processed_path, structure_checkpoint, device=device, batch_size=batch_size
    )
    structure_ids = [str(value) for value in structure["sample_ids"]]
    if len(structure_ids) != len(set(structure_ids)):
        raise ValueError("3D inference returned duplicate sample IDs.")
    structure_map = {
        sample_id: (
            float(structure["probabilities"][index]),
            str(structure["sequences"][index]),
            structure["labels"][index],
        )
        for index, sample_id in enumerate(structure_ids)
    }
    ids = [str(record.sample_id) for record in records]
    if set(ids) != set(structure_ids):
        raise ValueError("1D sequence records and 3D cache do not have identical sample coverage.")
    for record in records:
        _, sequence, label = structure_map[str(record.sample_id)]
        if sequence != record.sequence:
            raise ValueError(f"1D/3D sequence mismatch for {record.sample_id}.")
        if label is not None and record.label is not None and int(label) != int(record.label):
            raise ValueError(f"1D/3D label mismatch for {record.sample_id}.")
    labels = np.asarray([int(record.label) for record in records], dtype=np.int64)
    structure_index = {sample_id: index for index, sample_id in enumerate(structure_ids)}
    p_3d_base = np.asarray(
        [structure["base_probabilities"][structure_index[sample_id]] for sample_id in ids]
    )
    available = np.asarray(
        [structure["conformer_available"][structure_index[sample_id]] for sample_id in ids],
        dtype=bool,
    )
    conformer_delta = np.asarray(
        [structure["conformer_deltas"][structure_index[sample_id]] for sample_id in ids]
    )
    mean_plddt = np.asarray(
        [structure["mean_plddt"][structure_index[sample_id]] for sample_id in ids]
    )
    if available.any() or np.any(np.abs(conformer_delta) > 1e-10):
        raise ValueError(
            "The canonical public predictor is single-structure at test time, but the processed "
            "input activated conformer features. Use an isolated research script for the retired "
            "multi-conformer ablation."
        )
    p_3d = p_3d_base
    fused = float(weight) * p_1d + (1.0 - float(weight)) * p_3d
    selective_result = None
    selective_summary = {"enabled": False}
    if selective_policy is not None:
        selective_result, selective_summary = _apply_selective_prediction(
            selective_policy,
            fused,
            p_1d,
            p_3d,
            threshold,
            one_d_seed_std=one_d_seed_std,
            mean_plddt=mean_plddt,
        )
    _write_predictions(
        output_csv,
        ids,
        labels,
        p_1d,
        p_3d,
        fused,
        threshold,
        weight,
        sequences=[str(record.sequence) for record in records],
        selective_result=selective_result,
    )
    result_metrics = _metrics(labels, fused, threshold)
    summary = {
        "model": DEFAULT_MULTIDIMENSIONAL_MODEL_NAME,
        "n": len(ids),
        "weight_1d": float(weight),
        "threshold": float(threshold),
        "checkpoint_seeds": list(checkpoint_seeds),
        "oof_seeds": list(oof_seeds),
        "checkpoints": [str(path) for path in checkpoints],
        "oof_jsons": [str(path) for path in oof_jsons],
        "structure_checkpoint": str(structure_checkpoint),
        "test_time_structure_protocol": {
            "mode": "single_structure_only",
            "probability_source": "base_logits",
            "conformer_inputs_allowed": False,
            "conformer_available_records": 0,
        },
        "selective_prediction": selective_summary,
        "metrics": result_metrics,
    }
    if output_metrics:
        destination = Path(output_metrics)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GHXTox-Multidimensional-Fusion (public default)")
    parser.add_argument("--public-config", default=DEFAULT_MULTIDIMENSIONAL_CONFIG)
    subcommands = parser.add_subparsers(dest="command", required=True)

    assemble_parser = subcommands.add_parser(
        "assemble", help="assemble the canonical fusion from aligned 1D/3D CSVs (CPU)"
    )
    assemble_parser.add_argument("--cal-1d", required=True)
    assemble_parser.add_argument("--structure-csv", required=True)
    assemble_parser.add_argument("--weight", type=float, default=DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D)
    assemble_parser.add_argument("--threshold", type=float, default=DEFAULT_MULTIDIMENSIONAL_THRESHOLD)
    assemble_parser.add_argument("--allow-noncanonical-operating-point", action="store_true")
    assemble_parser.add_argument("--output-csv", required=True)
    assemble_parser.add_argument("--output-metrics")
    assemble_parser.add_argument(
        "--selective-policy",
        help="Optional frozen OOF policy; appends uncertainty/abstention fields without changing binary output.",
    )

    predict_parser = subcommands.add_parser(
        "predict", help="end-to-end canonical inference on processed sequence/3D inputs (GPU)"
    )
    predict_parser.add_argument("--processed", required=True)
    predict_parser.add_argument(
        "--sequence-records", "--adapter", dest="sequence_records", required=True,
        help="Sequence/label record cache; --adapter is a deprecated compatibility alias.",
    )
    predict_parser.add_argument("--config-1d", default=DEFAULT_1D_CONFIG)
    predict_parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINTS))
    predict_parser.add_argument("--oof-jsons", nargs="+", default=list(DEFAULT_OOF_JSONS))
    predict_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    predict_parser.add_argument("--structure-checkpoint", default=DEFAULT_STRUCTURE_CHECKPOINT)
    predict_parser.add_argument("--weight", type=float, default=DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D)
    predict_parser.add_argument("--threshold", type=float, default=DEFAULT_MULTIDIMENSIONAL_THRESHOLD)
    predict_parser.add_argument("--allow-noncanonical-operating-point", action="store_true")
    predict_parser.add_argument("--device", default="auto")
    predict_parser.add_argument("--batch-size", type=int, default=16)
    predict_parser.add_argument("--output-csv", required=True)
    predict_parser.add_argument("--output-metrics")
    predict_parser.add_argument(
        "--selective-policy",
        help="Optional frozen OOF policy; appends uncertainty/abstention fields without changing binary output.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    public_config = json.loads(Path(args.public_config).read_text(encoding="utf-8-sig"))
    validate_public_default_config(public_config)
    if args.command == "assemble":
        result = assemble(
            args.weight,
            args.threshold,
            args.cal_1d,
            args.structure_csv,
            args.output_csv,
            args.output_metrics,
            allow_noncanonical=args.allow_noncanonical_operating_point,
            selective_policy=args.selective_policy,
        )
    else:
        result = predict(
            args.processed,
            args.sequence_records,
            args.config_1d,
            args.checkpoints,
            args.oof_jsons,
            args.manifest,
            args.structure_checkpoint,
            args.weight,
            args.threshold,
            args.device,
            args.batch_size,
            args.output_csv,
            args.output_metrics,
            allow_noncanonical=args.allow_noncanonical_operating_point,
            selective_policy=args.selective_policy,
        )
    metrics = result["metrics"]
    print(
        f"{args.command} n={result['n']} BACC={metrics['balanced_accuracy']:.4f} "
        f"MCC={metrics['mcc']:.4f} AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f}"
    )


if __name__ == "__main__":
    main()
