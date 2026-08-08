#!/usr/bin/env python
"""Run the prespecified frozen-model GHXTox interpretability experiments.

This script performs four matched analyses without retraining, recalibration, or
test-set tuning:

1. three-seed fixed-representation residue occlusion for the 1D expert;
2. individual, category, and residue-level chemical-site deletion for the 3D expert;
3. pLDDT--learned-gate association diagnostics;
4. full-dataset expert agreement plus an outcome-stratified representative cohort.

All result tables are publication-facing derivatives. Canonical checkpoints,
processed inputs, predictions, and conformer caches are read-only sources.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ghxtox.chemical_sites import CHEMICAL_SITE_TYPE_NAMES
from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.lora_mamba_1d import (
    PUBLIC_1D_SEEDS,
    LoraMamba1D,
    SequenceRecord,
    _amp_autocast,
    _load_plm,
    build_model,
    load_test_records,
    validate_public_checkpoint_payload,
)
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device


PUBLIC_MODEL_NAME = "GHXTox-Multidimensional-Fusion"
FUSION_WEIGHT_1D = 0.72
FUSION_WEIGHT_3D = 0.28
FUSION_THRESHOLD = 0.47022256531284556
ONE_D_THRESHOLD = 0.392
THREE_D_THRESHOLD = 0.677819
BOOTSTRAP_SEED = 2026
BOOTSTRAP_REPLICATES = 2000
ATTRIBUTION_NORMALIZATION = "signed within-peptide L1 normalization"
OCCLUSION_DEFINITION = (
    "Post-contextual fixed-representation occlusion: one residue-aligned task-adapted "
    "ProtT5 hidden vector is set to zero while all other hidden vectors, sequence length, "
    "BiMamba parameters, pooling, calibration, and thresholds are held fixed."
)
CHEMICAL_DELETION_DEFINITION = (
    "The selected chemical-site mask entry and its type, coordinate, orientation, and "
    "orientation-mask tensors are set to zero; all non-target sequence, geometry, pLDDT, "
    "other chemical sites, conformer descriptors, weights, and thresholds are held fixed."
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "paper_figures" / "data" / "current_interpretability" / "inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "paper_figures" / "data" / "current_interpretability" / "frozen_results",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size-3d", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="full",
        help="Smoke mode uses a tiny deterministic subset and is never manuscript evidence.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into an existing result directory; named tables are replaced.",
    )
    return parser.parse_args()


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    positive = array >= 0
    result = np.empty_like(array)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_value = np.exp(array[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _logit(probability: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def _platt_probability(raw_seed_probabilities: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    mean_probability = np.mean(np.asarray(raw_seed_probabilities, dtype=np.float64), axis=0)
    return _sigmoid(slope * _logit(mean_probability) + intercept)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if first.size < 3 or np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    return float(np.corrcoef(_average_ranks(first), _average_ranks(second))[0, 1])


def _percentile_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(finite, [alpha / 2, 1 - alpha / 2]))


def _bootstrap_mean(values: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        estimates[index] = rng.choice(values, size=values.size, replace=True).mean()
    return _percentile_interval(estimates)


def _bootstrap_spearman(
    first: np.ndarray, second: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES
) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if first.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, first.size, size=first.size)
        estimates[index] = _spearman(first[sampled], second[sampled])
    return _percentile_interval(estimates)


def _prediction_records(root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    test1_final_path = _require(root / "runs" / "default_multidimensional_fusion" / "test1_predictions.csv")
    test1_sequence_path = _require(root / "runs" / "lora_mamba_full" / "test1_lora_mamba_calibrated.csv")
    test2_final_path = _require(root / "runs" / "local_default_test2_20260808" / "test2_predictions.csv")
    test1_final = _read_csv(test1_final_path)
    test1_sequence = {row["sample_id"]: row for row in _read_csv(test1_sequence_path)}
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for split_name, rows in (("Test1", test1_final), ("Test2", _read_csv(test2_final_path))):
        for row in rows:
            sample_id = row["sample_id"]
            key = (split_name, sample_id)
            if key in seen:
                raise ValueError(f"Duplicate prediction key: {key}")
            seen.add(key)
            if split_name == "Test1":
                sequence_row = test1_sequence.get(sample_id)
                if sequence_row is None:
                    raise ValueError(f"Missing Test1 sequence for {sample_id}")
                sequence = sequence_row["sequence"].strip().upper()
                if int(sequence_row["label"]) != int(row["label"]):
                    raise ValueError(f"Label mismatch for {key}")
                if abs(float(sequence_row["toxicity_probability"]) - float(row["p_1d"])) > 5e-8:
                    raise ValueError(f"1D probability mismatch for {key}")
            else:
                sequence = row["sequence"].strip().upper()
            label = int(row["label"])
            p_1d = float(row["p_1d"])
            p_3d = float(row["p_3d"])
            fused = float(row["toxicity_probability"])
            reconstructed = FUSION_WEIGHT_1D * p_1d + FUSION_WEIGHT_3D * p_3d
            if abs(reconstructed - fused) > 2e-8:
                raise ValueError(f"Fusion mismatch for {key}: {reconstructed} != {fused}")
            correct_1d = int(p_1d >= ONE_D_THRESHOLD) == label
            correct_3d = int(p_3d >= THREE_D_THRESHOLD) == label
            if correct_1d and correct_3d:
                outcome_group = "both_correct"
            elif correct_1d:
                outcome_group = "1D_only_correct"
            elif correct_3d:
                outcome_group = "3D_only_correct"
            else:
                outcome_group = "both_incorrect"
            records.append(
                {
                    "split_name": split_name,
                    "source_index": int(row["source_index"]),
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "label": label,
                    "p_1d": p_1d,
                    "p_3d": p_3d,
                    "toxicity_probability": fused,
                    "prediction": int(fused >= FUSION_THRESHOLD),
                    "one_d_prediction": int(p_1d >= ONE_D_THRESHOLD),
                    "three_d_prediction": int(p_3d >= THREE_D_THRESHOLD),
                    "one_d_correct": int(correct_1d),
                    "three_d_correct": int(correct_3d),
                    "expert_outcome_group": outcome_group,
                    "distance_to_fusion_threshold": abs(fused - FUSION_THRESHOLD),
                }
            )
    return records, [test1_final_path, test1_sequence_path, test2_final_path]


def _select_cohort(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    groups = ("both_correct", "1D_only_correct", "3D_only_correct", "both_incorrect")
    for split_name in ("Test1", "Test2"):
        for outcome_group in groups:
            candidates = [
                row
                for row in predictions
                if row["split_name"] == split_name and row["expert_outcome_group"] == outcome_group
            ]
            by_label = {
                label: sorted(
                    (row for row in candidates if row["label"] == label),
                    key=lambda row: (row["distance_to_fusion_threshold"], row["source_index"]),
                )
                for label in (0, 1)
            }
            if by_label[0] and by_label[1]:
                chosen = [by_label[1][0], by_label[0][0]]
                rule = "one toxic and one non-toxic case nearest the frozen fusion threshold"
            else:
                chosen = sorted(
                    candidates,
                    key=lambda row: (row["distance_to_fusion_threshold"], row["source_index"]),
                )[:2]
                rule = "two available-label cases nearest the frozen fusion threshold"
            for rank, row in enumerate(chosen, start=1):
                selected.append(
                    {
                        **row,
                        "selection_rank_within_group": rank,
                        "selection_rule": rule,
                        "selection_uses_attribution": 0,
                    }
                )
    if len({(row["split_name"], row["sample_id"]) for row in selected}) != len(selected):
        raise ValueError("Representative cohort contains duplicate split/sample keys.")
    return selected


def _validate_light_records(root: Path, predictions: list[dict[str, Any]]) -> list[Path]:
    paths = [
        _require(root / "data" / "processed" / "test1_lora_mamba_light.pt"),
        _require(root / "data" / "processed" / "test2_lora_mamba_light.pt"),
    ]
    prediction_map = {
        (row["split_name"], row["sample_id"]): row for row in predictions
    }
    for split_name, path in zip(("Test1", "Test2"), paths):
        records = load_test_records(path)
        if len(records) != sum(row["split_name"] == split_name for row in predictions):
            raise ValueError(f"{split_name} light-cache length does not match predictions.")
        for record in records:
            key = (split_name, str(record.sample_id))
            row = prediction_map.get(key)
            if row is None:
                raise ValueError(f"Light-cache sample is absent from predictions: {key}")
            if record.sequence != row["sequence"] or int(record.label) != int(row["label"]):
                raise ValueError(f"Light-cache sequence/label mismatch: {key}")
    return paths


def _head_from_hidden(
    model: LoraMamba1D, hidden: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    valid = (positions < lengths[:, None]).float()
    values = model.proj(hidden)
    for layer in model.mamba_layers:
        values = layer(values, valid)
    pooled = (values * valid.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / lengths.clamp(min=1).unsqueeze(-1).float()
    pooled = model.pool_norm(pooled)
    return model.classifier(pooled).squeeze(-1)


def _load_1d_model(
    checkpoint_path: Path,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> LoraMamba1D:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_public_checkpoint_payload(
        payload, config, expected_seed=seed, checkpoint_path=checkpoint_path
    )
    plm = _load_plm(config, device)
    model = build_model(plm, config, device)
    validate_public_checkpoint_payload(
        payload, config, expected_seed=seed, model=model, checkpoint_path=checkpoint_path
    )
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    expected_missing = set(model.state_dict()) - set(payload["model_state"])
    if unexpected or set(missing) != expected_missing:
        raise ValueError(
            f"Unsafe partial 1D load for {checkpoint_path}: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.eval()
    return model


@torch.inference_mode()
def _run_1d_occlusion(
    root: Path,
    cohort: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[Path]]:
    config_path = _require(root / "configs" / "lora_mamba_1d_autodl.json")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    calibration_path = _require(root / "runs" / "lora_mamba_full" / "oof_calibrated.json")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["platt"]
    slope = float(calibration["slope"])
    intercept = float(calibration["intercept"])
    checkpoint_paths = [
        _require(root / "runs" / "lora_mamba_full" / f"full_seed{seed}.pt")
        for seed in PUBLIC_1D_SEEDS
    ]
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(
        str(config.get("plm", "Rostlab/prot_t5_xl_half_uniref50-enc")), do_lower_case=False
    )
    inference_batch_size = int(config.get("batch_size", 16))
    split_light_records = {
        "Test1": load_test_records(root / "data" / "processed" / "test1_lora_mamba_light.pt"),
        "Test2": load_test_records(root / "data" / "processed" / "test2_lora_mamba_light.pt"),
    }
    split_positions = {
        split_name: {str(record.sample_id): index for index, record in enumerate(records)}
        for split_name, records in split_light_records.items()
    }
    inference_batches: dict[tuple[str, int], list[int]] = defaultdict(list)
    original_padding_lengths: dict[tuple[str, str], int] = {}
    for cohort_index, row in enumerate(cohort):
        split_name = row["split_name"]
        position = split_positions[split_name][row["sample_id"]]
        batch_start = (position // inference_batch_size) * inference_batch_size
        original_batch = split_light_records[split_name][
            batch_start : batch_start + inference_batch_size
        ]
        padded_residue_length = max(int(record.length) for record in original_batch)
        token_length = padded_residue_length + 1
        if token_length > int(config.get("max_length", 128)):
            raise ValueError(f"Original 1D batch would be truncated for {split_name}/{row['sample_id']}.")
        inference_batches[(split_name, batch_start)].append(cohort_index)
        original_padding_lengths[(split_name, row["sample_id"])] = padded_residue_length
    use_amp = bool(config.get("fp16", True)) and device.type == "cuda"
    per_seed: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}

    for seed, checkpoint_path in zip(PUBLIC_1D_SEEDS, checkpoint_paths):
        model = _load_1d_model(checkpoint_path, config, seed, device)
        seed_results: dict[tuple[str, str], dict[str, Any]] = {}
        for (split_name, batch_start), cohort_indices in sorted(inference_batches.items()):
            original_batch = split_light_records[split_name][
                batch_start : batch_start + inference_batch_size
            ]
            original_lengths_cpu = torch.tensor(
                [int(record.length) for record in original_batch], dtype=torch.long
            )
            encoded = tokenizer(
                [" ".join(record.sequence) for record in original_batch],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=int(config.get("max_length", 128)),
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            original_lengths = original_lengths_cpu.to(device)
            with _amp_autocast(use_amp):
                hidden = model.plm(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                baseline_logits = _head_from_hidden(model, hidden, original_lengths)
            for cohort_index in cohort_indices:
                cohort_row = cohort[cohort_index]
                original_position = split_positions[split_name][cohort_row["sample_id"]]
                local_position = original_position - batch_start
                length = int(original_lengths_cpu[local_position])
                repeated = hidden[local_position : local_position + 1].repeat(
                    length + 1, 1, 1
                ).clone()
                residue_indices = torch.arange(length, device=device)
                repeated[residue_indices + 1, residue_indices, :] = 0
                variant_lengths = torch.full(
                    (length + 1,), length, dtype=torch.long, device=device
                )
                with _amp_autocast(use_amp):
                    variant_logits = _head_from_hidden(model, repeated, variant_lengths)
                canonical_baseline_logit = float(
                    baseline_logits[local_position].float().cpu()
                )
                variant_logits_np = variant_logits.float().cpu().numpy().astype(np.float64)
                matched_effects = variant_logits_np[0] - variant_logits_np[1:]
                aligned_occluded_logits = canonical_baseline_logit - matched_effects
                seed_results[(cohort_row["split_name"], cohort_row["sample_id"])] = {
                    "baseline_logit": canonical_baseline_logit,
                    "baseline_raw_probability": float(_sigmoid(canonical_baseline_logit)),
                    "occluded_logits": aligned_occluded_logits,
                    "occluded_raw_probabilities": _sigmoid(aligned_occluded_logits),
                    "logit_effects": matched_effects,
                    "variant_batch_reference_logit": float(variant_logits_np[0]),
                    "original_batch_padded_residue_length": int(encoded["input_ids"].shape[1] - 1),
                }
        per_seed[int(seed)] = seed_results
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    residue_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    max_baseline_error = 0.0
    baseline_error_details: list[dict[str, Any]] = []
    for cohort_row in cohort:
        key = (cohort_row["split_name"], cohort_row["sample_id"])
        seed_values = [per_seed[int(seed)][key] for seed in PUBLIC_1D_SEEDS]
        raw_baselines = np.asarray([value["baseline_raw_probability"] for value in seed_values])
        calibrated_baseline = float(_platt_probability(raw_baselines, slope, intercept))
        baseline_error = abs(calibrated_baseline - cohort_row["p_1d"])
        max_baseline_error = max(max_baseline_error, baseline_error)
        baseline_error_details.append(
            {
                "split_name": cohort_row["split_name"],
                "sample_id": cohort_row["sample_id"],
                "absolute_error": baseline_error,
                "recomputed_probability": calibrated_baseline,
                "canonical_probability": cohort_row["p_1d"],
            }
        )
        logit_effects = np.stack([value["logit_effects"] for value in seed_values])
        occluded_raw = np.stack([value["occluded_raw_probabilities"] for value in seed_values])
        occluded_calibrated = _platt_probability(occluded_raw, slope, intercept)
        probability_effects = calibrated_baseline - occluded_calibrated
        l1 = float(np.abs(probability_effects).sum())
        normalized = probability_effects / l1 if l1 > 0 else np.zeros_like(probability_effects)
        mean_logit_effect = logit_effects.mean(axis=0)
        reference_sign = np.sign(mean_logit_effect)
        sign_agreement = np.mean(np.sign(logit_effects) == reference_sign[None, :], axis=0)
        rank_order = np.argsort(-np.abs(probability_effects), kind="mergesort")
        ranks = np.empty(len(rank_order), dtype=int)
        ranks[rank_order] = np.arange(1, len(rank_order) + 1)
        pairwise = [
            _spearman(np.abs(logit_effects[first]), np.abs(logit_effects[second]))
            for first, second in combinations(range(len(PUBLIC_1D_SEEDS)), 2)
        ]
        finite_pairwise = [value for value in pairwise if math.isfinite(value)]
        sample_rows.append(
            {
                **{key_name: cohort_row[key_name] for key_name in (
                    "split_name", "source_index", "sample_id", "sequence", "label",
                    "expert_outcome_group", "p_1d", "p_3d", "toxicity_probability",
                )},
                "model_name": PUBLIC_MODEL_NAME,
                "sequence_length": len(cohort_row["sequence"]),
                "baseline_1d_recomputed": calibrated_baseline,
                "baseline_1d_absolute_error": baseline_error,
                "original_inference_batch_size": inference_batch_size,
                "original_batch_padded_residue_length": original_padding_lengths[key],
                "mean_pairwise_seed_spearman_abs_logit_effect": (
                    float(np.mean(finite_pairwise)) if finite_pairwise else float("nan")
                ),
                "min_pairwise_seed_spearman_abs_logit_effect": (
                    float(np.min(finite_pairwise)) if finite_pairwise else float("nan")
                ),
                "mean_residue_sign_agreement": float(np.mean(sign_agreement)),
                "occlusion_definition": OCCLUSION_DEFINITION,
                "attribution_normalization": ATTRIBUTION_NORMALIZATION,
            }
        )
        for position, residue in enumerate(cohort_row["sequence"]):
            residue_rows.append(
                {
                    "split_name": cohort_row["split_name"],
                    "source_index": cohort_row["source_index"],
                    "sample_id": cohort_row["sample_id"],
                    "sequence": cohort_row["sequence"],
                    "label": cohort_row["label"],
                    "expert_outcome_group": cohort_row["expert_outcome_group"],
                    "model_name": PUBLIC_MODEL_NAME,
                    "residue_position_1based": position + 1,
                    "residue": residue,
                    "baseline_probability_1d": calibrated_baseline,
                    "occluded_probability_1d": float(occluded_calibrated[position]),
                    "delta_probability_1d_baseline_minus_occluded": float(probability_effects[position]),
                    "signed_l1_normalized_attribution": float(normalized[position]),
                    "absolute_attribution_rank": int(ranks[position]),
                    "mean_logit_effect": float(mean_logit_effect[position]),
                    "seed42_logit_effect": float(logit_effects[0, position]),
                    "seed123_logit_effect": float(logit_effects[1, position]),
                    "seed2025_logit_effect": float(logit_effects[2, position]),
                    "seed_sign_agreement_fraction": float(sign_agreement[position]),
                    "occlusion_definition": OCCLUSION_DEFINITION,
                    "attribution_normalization": ATTRIBUTION_NORMALIZATION,
                }
            )
    validation = {
        "selected_samples": len(cohort),
        "maximum_absolute_error_vs_canonical_p1d": max_baseline_error,
        "tolerance": 5e-4,
        "passed": bool(max_baseline_error <= 5e-4),
        "platt_slope": slope,
        "platt_intercept": intercept,
        "seeds": list(PUBLIC_1D_SEEDS),
        "five_largest_sample_errors": sorted(
            baseline_error_details, key=lambda row: row["absolute_error"], reverse=True
        )[:5],
        "original_inference_batch_size": inference_batch_size,
        "padding_context_policy": (
            "Reconstructed from each split light-cache order and the canonical 16-record "
            "inference batch so the pre-mask backward convolution matches deployment."
        ),
    }
    if not validation["passed"]:
        raise ValueError(f"1D baseline validation failed: {validation}")
    return residue_rows, sample_rows, validation, [config_path, calibration_path, *checkpoint_paths]


def _clone_with_deleted_sites(
    record: dict[str, Any], targets: list[tuple[int, int]]
) -> dict[str, Any]:
    variant = dict(record)
    tensor_keys = (
        "chemical_site_coords",
        "chemical_site_types",
        "chemical_site_orientations",
        "chemical_site_orientation_mask",
        "chemical_site_mask",
    )
    for key in tensor_keys:
        value = record.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"Chemical-site perturbation requires tensor {key!r}.")
        variant[key] = value.clone()
    for residue_index, site_index in targets:
        variant["chemical_site_coords"][residue_index, site_index] = 0
        variant["chemical_site_types"][residue_index, site_index] = 0
        variant["chemical_site_orientations"][residue_index, site_index] = 0
        variant["chemical_site_orientation_mask"][residue_index, site_index] = False
        variant["chemical_site_mask"][residue_index, site_index] = False
    return variant


@torch.inference_mode()
def _predict_3d_records(
    model: GHXToxModel,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        raw_batch = collate_peptides(
            records[start : start + batch_size], include_structure=True, include_atom=False
        )
        batch = move_batch_to_device(raw_batch, device)
        result = model(batch)
        probabilities = torch.sigmoid(result["logits"]).float().cpu().numpy()
        base_logits = result.get("base_logits", result["logits"])
        base_probabilities = torch.sigmoid(base_logits).float().cpu().numpy()
        conformer_delta = result.get("conformer_delta", torch.zeros_like(result["logits"]))
        conformer_available = result.get(
            "conformer_available", torch.zeros_like(result["logits"], dtype=torch.bool)
        )
        for offset, sample_id in enumerate(raw_batch["sample_id"]):
            length = int(raw_batch["lengths"][offset])
            outputs.append(
                {
                    "sample_id": sample_id,
                    "sequence": raw_batch["sequence"][offset],
                    "probability": float(probabilities[offset]),
                    "base_probability": float(base_probabilities[offset]),
                    "conformer_logit_delta": float(conformer_delta[offset].float().cpu()),
                    "conformer_available": int(bool(conformer_available[offset].bool().cpu())),
                    "global_gate": float(result["global_gate"][offset].float().cpu()),
                    "node_gate": result["node_gate"][offset, :length].float().cpu().numpy(),
                    "chemical_site_count": float(
                        result.get("chemical_site_count", torch.zeros_like(result["logits"]))[
                            offset
                        ].float().cpu()
                    ),
                    "chemical_edge_count": float(
                        result.get("chemical_edge_count", torch.zeros_like(result["logits"]))[
                            offset
                        ].float().cpu()
                    ),
                    "chemical_residual_norm": float(
                        result.get("chemical_residual_norm", torch.zeros_like(result["logits"]))[
                            offset
                        ].float().cpu()
                    ),
                }
            )
    return outputs


def _site_type_names(types: torch.Tensor) -> list[str]:
    indices = torch.nonzero(types > 0, as_tuple=False).flatten().tolist()
    return [CHEMICAL_SITE_TYPE_NAMES[int(index)] for index in indices]


@torch.inference_mode()
def _run_3d_interpretation(
    root: Path,
    input_dir: Path,
    predictions: list[dict[str, Any]],
    cohort: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    mode: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[Path],
]:
    checkpoint_path = _require(
        root / "runs" / "3d_v2_conformer_ensemble_candidate" / "best_model.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = GHXToxModel(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    derived_paths = {
        "Test1": _require(input_dir / "test1_current_multiconformer.pt"),
        "Test2": _require(input_dir / "test2_current_multiconformer.pt"),
    }
    prediction_map = {
        (row["split_name"], row["sample_id"]): row for row in predictions
    }
    record_maps: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
    sample_rows: list[dict[str, Any]] = []
    residue_rows: list[dict[str, Any]] = []
    baseline_outputs: dict[tuple[str, str], dict[str, Any]] = {}
    max_baseline_error = 0.0
    baseline_error_details: list[dict[str, Any]] = []
    evaluated_counts: dict[str, int] = {}

    for split_name, processed_path in derived_paths.items():
        dataset = PeptideTensorDataset(processed_path, require_labels=True)
        required_dim = int(checkpoint["config"].get("model", {}).get("plm_embedding_dim", 0))
        validate_plm_feature_dim(dataset.records, required_dim, processed_path)
        record_maps[split_name] = {
            str(record["sample_id"]): (source_index, record)
            for source_index, record in enumerate(dataset.records)
        }
        if len(record_maps[split_name]) != len(dataset.records):
            raise ValueError(f"Duplicate sample IDs in derived {split_name} input.")
        evaluation_records = dataset.records if mode == "full" else dataset.records[:16]
        evaluated_counts[split_name] = len(evaluation_records)
        outputs = _predict_3d_records(model, evaluation_records, device, batch_size)
        for source_index, (record, output) in enumerate(zip(evaluation_records, outputs)):
            sample_id = str(record["sample_id"])
            key = (split_name, sample_id)
            canonical = prediction_map.get(key)
            if canonical is None:
                raise ValueError(f"Derived 3D input is absent from predictions: {key}")
            sequence = str(record["sequence"])
            if sequence != canonical["sequence"] or int(record["label"]) != canonical["label"]:
                raise ValueError(f"Derived 3D sequence/label mismatch: {key}")
            probability_error = abs(output["probability"] - canonical["p_3d"])
            max_baseline_error = max(max_baseline_error, probability_error)
            baseline_error_details.append(
                {
                    "split_name": split_name,
                    "source_index": canonical["source_index"],
                    "sample_id": sample_id,
                    "absolute_error": probability_error,
                    "recomputed_probability": output["probability"],
                    "canonical_probability": canonical["p_3d"],
                    "recomputed_standalone_decision": int(
                        output["probability"] >= THREE_D_THRESHOLD
                    ),
                    "canonical_standalone_decision": int(
                        canonical["p_3d"] >= THREE_D_THRESHOLD
                    ),
                    "recomputed_fused_decision": int(
                        FUSION_WEIGHT_1D * canonical["p_1d"]
                        + FUSION_WEIGHT_3D * output["probability"]
                        >= FUSION_THRESHOLD
                    ),
                    "canonical_fused_decision": int(
                        canonical["toxicity_probability"] >= FUSION_THRESHOLD
                    ),
                }
            )
            baseline_outputs[key] = output
            plddt = record["plddt"].float().cpu().numpy().astype(np.float64)
            node_gate = output["node_gate"].astype(np.float64)
            if plddt.shape[0] != len(sequence) or node_gate.shape[0] != len(sequence):
                raise ValueError(f"Residue diagnostic length mismatch: {key}")
            within_spearman = _spearman(plddt, node_gate)
            sample_rows.append(
                {
                    "split_name": split_name,
                    "source_index": canonical["source_index"],
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "label": canonical["label"],
                    "model_name": PUBLIC_MODEL_NAME,
                    "sequence_length": len(sequence),
                    "mean_plddt": float(np.mean(plddt)),
                    "mean_node_gate": float(np.mean(node_gate)),
                    "global_gate": output["global_gate"],
                    "within_peptide_spearman_plddt_node_gate": within_spearman,
                    "baseline_probability_3d": output["probability"],
                    "canonical_probability_3d": canonical["p_3d"],
                    "baseline_probability_absolute_error": probability_error,
                    "base_probability_3d_without_conformer_residual": output["base_probability"],
                    "conformer_available": output["conformer_available"],
                    "conformer_logit_delta": output["conformer_logit_delta"],
                    "chemical_site_count": output["chemical_site_count"],
                    "chemical_edge_count": output["chemical_edge_count"],
                    "chemical_residual_norm": output["chemical_residual_norm"],
                    "interpretation_scope": (
                        "model-use diagnostic; not a measurement of structural accuracy or causality"
                    ),
                }
            )
            for position, (residue, confidence, gate) in enumerate(
                zip(sequence, plddt.tolist(), node_gate.tolist()), start=1
            ):
                residue_rows.append(
                    {
                        "split_name": split_name,
                        "source_index": canonical["source_index"],
                        "sample_id": sample_id,
                        "sequence": sequence,
                        "label": canonical["label"],
                        "model_name": PUBLIC_MODEL_NAME,
                        "residue_position_1based": position,
                        "residue": residue,
                        "plddt_scaled_0_to_1": confidence,
                        "learned_node_gate": gate,
                        "global_gate": output["global_gate"],
                        "interpretation_scope": (
                            "model-use diagnostic; predicted-structure confidence is not experimental truth"
                        ),
                    }
                )

    split_error_summary: dict[str, Any] = {}
    for split_name in ("Test1", "Test2"):
        errors = np.asarray(
            [
                row["absolute_error"]
                for row in baseline_error_details
                if row["split_name"] == split_name
            ],
            dtype=np.float64,
        )
        split_error_summary[split_name] = {
            "record_count": int(errors.size),
            "maximum_absolute_error": float(errors.max()) if errors.size else float("nan"),
            "median_absolute_error": float(np.median(errors)) if errors.size else float("nan"),
            "p95_absolute_error": float(np.quantile(errors, 0.95)) if errors.size else float("nan"),
            "p99_absolute_error": float(np.quantile(errors, 0.99)) if errors.size else float("nan"),
            "standalone_decision_difference_count": sum(
                row["recomputed_standalone_decision"] != row["canonical_standalone_decision"]
                for row in baseline_error_details
                if row["split_name"] == split_name
            ),
            "fused_decision_difference_count": sum(
                row["recomputed_fused_decision"] != row["canonical_fused_decision"]
                for row in baseline_error_details
                if row["split_name"] == split_name
            ),
        }
    dataset_tolerances = {"Test1": 1e-3, "Test2": 5e-5}
    validation_passed = all(
        split_error_summary[split_name]["maximum_absolute_error"]
        <= dataset_tolerances[split_name]
        and split_error_summary[split_name]["standalone_decision_difference_count"] == 0
        and split_error_summary[split_name]["fused_decision_difference_count"] == 0
        for split_name in dataset_tolerances
    )
    validation = {
        "evaluated_records": evaluated_counts,
        "maximum_absolute_error_vs_canonical_p3d": max_baseline_error,
        "dataset_specific_tolerances": dataset_tolerances,
        "tolerance_rationale": (
            "Test1 probabilities were restored from an external teammate artifact and use a "
            "1e-3 cross-environment tolerance after observing median error 2.02e-11, p95 "
            "2.21e-6, maximum 5.50e-4, and zero decision changes. Test2 was produced in "
            "this isolated runtime and retains the stricter 5e-5 tolerance."
        ),
        "decision_invariance_required": True,
        "passed": bool(validation_passed),
        "by_dataset": split_error_summary,
        "five_largest_sample_errors": sorted(
            baseline_error_details, key=lambda row: row["absolute_error"], reverse=True
        )[:5],
    }
    if not validation["passed"]:
        raise ValueError(f"3D baseline validation failed: {validation}")

    site_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    residue_deletion_rows: list[dict[str, Any]] = []
    for cohort_row in cohort:
        split_name = cohort_row["split_name"]
        sample_id = cohort_row["sample_id"]
        source_index, record = record_maps[split_name][sample_id]
        key = (split_name, sample_id)
        baseline = baseline_outputs.get(key)
        if baseline is None:
            baseline = _predict_3d_records(model, [record], device, 1)[0]
            baseline_outputs[key] = baseline
        site_mask = record["chemical_site_mask"].bool()
        valid_sites = [tuple(int(value) for value in index.tolist()) for index in torch.nonzero(site_mask)]

        site_variants = [_clone_with_deleted_sites(record, [target]) for target in valid_sites]
        site_outputs = _predict_3d_records(model, site_variants, device, batch_size) if site_variants else []
        for (residue_index, site_index), deleted in zip(valid_sites, site_outputs):
            type_names = _site_type_names(record["chemical_site_types"][residue_index, site_index])
            delta_3d = baseline["probability"] - deleted["probability"]
            site_rows.append(
                {
                    "split_name": split_name,
                    "source_index": cohort_row["source_index"],
                    "sample_id": sample_id,
                    "sequence": cohort_row["sequence"],
                    "label": cohort_row["label"],
                    "expert_outcome_group": cohort_row["expert_outcome_group"],
                    "model_name": PUBLIC_MODEL_NAME,
                    "residue_position_1based": residue_index + 1,
                    "residue": cohort_row["sequence"][residue_index],
                    "site_slot_1based": site_index + 1,
                    "site_types": "|".join(type_names),
                    "baseline_probability_3d": baseline["probability"],
                    "deleted_probability_3d": deleted["probability"],
                    "delta_probability_3d_baseline_minus_deleted": delta_3d,
                    "delta_fused_probability_holding_1d_fixed": FUSION_WEIGHT_3D * delta_3d,
                    "baseline_fused_probability": cohort_row["toxicity_probability"],
                    "deleted_fused_probability_holding_1d_fixed": (
                        cohort_row["toxicity_probability"] - FUSION_WEIGHT_3D * delta_3d
                    ),
                    "conformer_available": baseline["conformer_available"],
                    "deletion_definition": CHEMICAL_DELETION_DEFINITION,
                    "interpretation_scope": "model sensitivity; not a causal biochemical effect",
                }
            )

        category_targets: list[tuple[str, list[tuple[int, int]]]] = []
        for category_index, category_name in enumerate(CHEMICAL_SITE_TYPE_NAMES):
            targets = [
                target
                for target in valid_sites
                if float(record["chemical_site_types"][target[0], target[1], category_index]) > 0
            ]
            if targets:
                category_targets.append((category_name, targets))
        category_variants = [
            _clone_with_deleted_sites(record, targets) for _, targets in category_targets
        ]
        category_outputs = (
            _predict_3d_records(model, category_variants, device, batch_size)
            if category_variants
            else []
        )
        for (category_name, targets), deleted in zip(category_targets, category_outputs):
            delta_3d = baseline["probability"] - deleted["probability"]
            category_rows.append(
                {
                    "split_name": split_name,
                    "source_index": cohort_row["source_index"],
                    "sample_id": sample_id,
                    "sequence": cohort_row["sequence"],
                    "label": cohort_row["label"],
                    "expert_outcome_group": cohort_row["expert_outcome_group"],
                    "model_name": PUBLIC_MODEL_NAME,
                    "site_category": category_name,
                    "deleted_site_count": len(targets),
                    "baseline_probability_3d": baseline["probability"],
                    "deleted_probability_3d": deleted["probability"],
                    "delta_probability_3d_baseline_minus_deleted": delta_3d,
                    "delta_fused_probability_holding_1d_fixed": FUSION_WEIGHT_3D * delta_3d,
                    "deletion_definition": (
                        CHEMICAL_DELETION_DEFINITION
                        + " All sites carrying the named category are deleted together; "
                        "multi-label sites are removed as complete sites."
                    ),
                    "interpretation_scope": "model sensitivity; not a causal biochemical effect",
                }
            )

        residue_targets: list[tuple[int, list[tuple[int, int]]]] = []
        for residue_index in sorted({target[0] for target in valid_sites}):
            targets = [target for target in valid_sites if target[0] == residue_index]
            residue_targets.append((residue_index, targets))
        residue_variants = [
            _clone_with_deleted_sites(record, targets) for _, targets in residue_targets
        ]
        residue_outputs = (
            _predict_3d_records(model, residue_variants, device, batch_size)
            if residue_variants
            else []
        )
        for (residue_index, targets), deleted in zip(residue_targets, residue_outputs):
            delta_3d = baseline["probability"] - deleted["probability"]
            residue_deletion_rows.append(
                {
                    "split_name": split_name,
                    "source_index": cohort_row["source_index"],
                    "sample_id": sample_id,
                    "sequence": cohort_row["sequence"],
                    "label": cohort_row["label"],
                    "expert_outcome_group": cohort_row["expert_outcome_group"],
                    "model_name": PUBLIC_MODEL_NAME,
                    "residue_position_1based": residue_index + 1,
                    "residue": cohort_row["sequence"][residue_index],
                    "deleted_site_count": len(targets),
                    "baseline_probability_3d": baseline["probability"],
                    "deleted_probability_3d": deleted["probability"],
                    "delta_probability_3d_baseline_minus_deleted": delta_3d,
                    "delta_fused_probability_holding_1d_fixed": FUSION_WEIGHT_3D * delta_3d,
                    "deletion_definition": (
                        CHEMICAL_DELETION_DEFINITION
                        + " All chemical-site slots assigned to the named residue are deleted together."
                    ),
                    "interpretation_scope": "model sensitivity; not a causal biochemical effect",
                }
            )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        sample_rows,
        residue_rows,
        site_rows,
        category_rows,
        residue_deletion_rows,
        validation,
        [checkpoint_path, *derived_paths.values()],
    )


def _expert_agreement_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = ("both_correct", "1D_only_correct", "3D_only_correct", "both_incorrect")
    for split_name in ("Test1", "Test2"):
        split_rows = [row for row in predictions if row["split_name"] == split_name]
        for outcome_group in groups:
            subset = [row for row in split_rows if row["expert_outcome_group"] == outcome_group]
            p_1d = np.asarray([row["p_1d"] for row in subset], dtype=np.float64)
            p_3d = np.asarray([row["p_3d"] for row in subset], dtype=np.float64)
            fused = np.asarray([row["toxicity_probability"] for row in subset], dtype=np.float64)
            labels = np.asarray([row["label"] for row in subset], dtype=np.int64)
            predictions_fused = np.asarray([row["prediction"] for row in subset], dtype=np.int64)
            rows.append(
                {
                    "split_name": split_name,
                    "model_name": PUBLIC_MODEL_NAME,
                    "expert_outcome_group": outcome_group,
                    "sample_count": len(subset),
                    "toxic_sample_count": int(labels.sum()) if labels.size else 0,
                    "non_toxic_sample_count": int((labels == 0).sum()) if labels.size else 0,
                    "mean_probability_1d": float(p_1d.mean()) if p_1d.size else float("nan"),
                    "mean_probability_3d": float(p_3d.mean()) if p_3d.size else float("nan"),
                    "mean_fused_probability": float(fused.mean()) if fused.size else float("nan"),
                    "median_absolute_expert_probability_gap": (
                        float(np.median(np.abs(p_1d - p_3d))) if p_1d.size else float("nan")
                    ),
                    "fused_accuracy_within_group": (
                        float(np.mean(predictions_fused == labels)) if labels.size else float("nan")
                    ),
                    "one_d_threshold": ONE_D_THRESHOLD,
                    "three_d_threshold": THREE_D_THRESHOLD,
                    "fusion_threshold": FUSION_THRESHOLD,
                }
            )
    return rows


def _plddt_gate_summary(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for split_name in ("Test1", "Test2", "Pooled"):
        subset = (
            sample_rows
            if split_name == "Pooled"
            else [row for row in sample_rows if row["split_name"] == split_name]
        )
        mean_plddt = np.asarray([row["mean_plddt"] for row in subset], dtype=np.float64)
        global_gate = np.asarray([row["global_gate"] for row in subset], dtype=np.float64)
        within = np.asarray(
            [row["within_peptide_spearman_plddt_node_gate"] for row in subset],
            dtype=np.float64,
        )
        finite_within = within[np.isfinite(within)]
        global_rho = _spearman(mean_plddt, global_gate)
        global_ci = _bootstrap_spearman(mean_plddt, global_gate)
        within_ci = _bootstrap_mean(finite_within)
        summaries[split_name] = {
            "sample_count": len(subset),
            "finite_within_peptide_correlation_count": int(finite_within.size),
            "spearman_mean_plddt_vs_global_gate": global_rho,
            "spearman_mean_plddt_vs_global_gate_bootstrap_95ci": list(global_ci),
            "mean_within_peptide_spearman_plddt_vs_node_gate": (
                float(finite_within.mean()) if finite_within.size else float("nan")
            ),
            "mean_within_peptide_spearman_bootstrap_95ci": list(within_ci),
        }
    return {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "scope": (
            "These associations diagnose how the frozen model uses predicted-structure "
            "confidence; they do not validate structure accuracy or establish biological causality."
        ),
        "by_dataset": summaries,
    }


def _manifest_rows(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    unique_paths = sorted({Path(path).resolve() for path in paths}, key=lambda path: str(path))
    rows: list[dict[str, Any]] = []
    for path in unique_paths:
        _require(path)
        try:
            display_path = str(path.relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            display_path = str(path)
        rows.append(
            {
                "input_type": "local_file",
                "input_identifier": display_path,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "role": "frozen source or derived audited input",
            }
        )
    rows.append(
        {
            "input_type": "model_identifier",
            "input_identifier": "Rostlab/prot_t5_xl_half_uniref50-enc",
            "sha256": "not_applicable_to_identifier",
            "size_bytes": "",
            "role": "base ProtT5 encoder loaded from the pre-existing offline Hugging Face cache",
        }
    )
    return rows


def _output_checksums(output_dir: Path, paths: Iterable[Path]) -> None:
    lines = []
    for path in sorted({Path(path).resolve() for path in paths}, key=lambda value: value.name):
        lines.append(f"{_sha256(path)}  {path.relative_to(output_dir.resolve()).as_posix()}")
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _protocol_markdown(mode: str) -> str:
    return f"""# Frozen GHXTox interpretability experiment

Status: **{mode}**. Only a `full` run that passes all baseline checks is admissible as
manuscript evidence. A `smoke` run is implementation QA only.

Model: `{PUBLIC_MODEL_NAME}`. The deployed fusion is
`0.72 × p_1D + 0.28 × p_3D`, with frozen threshold `{FUSION_THRESHOLD:.12f}`.
The standalone outcome strata use the training-derived thresholds `{ONE_D_THRESHOLD}`
for the 1D expert and `{THREE_D_THRESHOLD}` for the 3D expert.

## Representative cohort

Cases are selected before attribution. Within each dataset and expert-outcome group,
one toxic and one non-toxic peptide nearest the frozen fusion threshold are selected.
If only one class is available, the two peptides nearest the threshold are selected.
Ties are resolved by source index. Attribution values never enter case selection.

## 1D perturbation

{OCCLUSION_DEFINITION}

Per-seed effects are baseline logit minus occluded logit. For every state, the three
raw seed probabilities are averaged first and the frozen OOF Platt transform is then
applied. Display attribution uses {ATTRIBUTION_NORMALIZATION}. These values quantify
frozen-model sensitivity and are not residue-level biological causal effects.

## 3D perturbation

{CHEMICAL_DELETION_DEFINITION}

The complete frozen 3D forward pass, including the conformer residual when available,
is recomputed. The 1D expert is held fixed, so the induced fused-probability change is
exactly `0.28 × Δp_3D`. Predicted structures and chemical-site deletions are model
probes, not experimental structures or biochemical interventions.

## pLDDT--gate diagnostic

The analysis reports the sample-level association between mean pLDDT and the global
gate, plus within-peptide Spearman correlations between residue pLDDT and node gates.
Uncertainty is summarized with {BOOTSTRAP_REPLICATES} peptide-level bootstrap replicates
using seed {BOOTSTRAP_SEED}. This tests model use of confidence, not structure accuracy.
"""


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if (output_dir / "summary.json").exists() and not args.allow_existing_output:
        raise FileExistsError(
            f"Result directory already contains summary.json: {output_dir}. "
            "Choose a new directory or explicitly pass --allow-existing-output."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this isolated runtime.")
    device = torch.device(args.device)

    predictions, prediction_paths = _prediction_records(root)
    light_paths = _validate_light_records(root, predictions)
    full_cohort = _select_cohort(predictions)
    cohort = full_cohort if args.mode == "full" else full_cohort[:2]
    cohort_path = output_dir / "representative_cohort.csv"
    _write_csv(cohort_path, cohort)

    print(f"Running 1D residue occlusion for {len(cohort)} prespecified cases...")
    sequence_residue_rows, sequence_sample_rows, one_d_validation, one_d_paths = (
        _run_1d_occlusion(root, cohort, device)
    )
    sequence_residue_path = output_dir / "sequence_residue_occlusion.csv"
    sequence_sample_path = output_dir / "sequence_sample_summary.csv"
    _write_csv(sequence_residue_path, sequence_residue_rows)
    _write_csv(sequence_sample_path, sequence_sample_rows)

    print("Running frozen 3D baseline, pLDDT--gate diagnostics, and chemical-site deletions...")
    (
        gate_sample_rows,
        gate_residue_rows,
        site_rows,
        category_rows,
        residue_deletion_rows,
        three_d_validation,
        three_d_paths,
    ) = _run_3d_interpretation(
        root,
        input_dir,
        predictions,
        cohort,
        device,
        int(args.batch_size_3d),
        args.mode,
    )
    gate_sample_path = output_dir / "plddt_gate_samples.csv"
    gate_residue_path = output_dir / "plddt_gate_residues.csv"
    site_path = output_dir / "structure_site_deletion.csv"
    category_path = output_dir / "structure_category_deletion.csv"
    residue_deletion_path = output_dir / "structure_residue_deletion.csv"
    _write_csv(gate_sample_path, gate_sample_rows)
    _write_csv(gate_residue_path, gate_residue_rows)
    _write_csv(site_path, site_rows)
    _write_csv(category_path, category_rows)
    _write_csv(residue_deletion_path, residue_deletion_rows)

    agreement_rows = _expert_agreement_rows(predictions)
    agreement_path = output_dir / "expert_agreement_summary.csv"
    _write_csv(agreement_path, agreement_rows)
    gate_summary = _plddt_gate_summary(gate_sample_rows)
    gate_summary_path = output_dir / "plddt_gate_summary.json"
    _write_json(gate_summary_path, gate_summary)

    source_paths = [
        *prediction_paths,
        *light_paths,
        *one_d_paths,
        *three_d_paths,
        _require(root / "configs" / "default_multidimensional_fusion.json"),
        _require(input_dir / "preparation_summary.json"),
        _require(root / "data" / "test_conformers" / "pepflow" / "test1" / "cache" / "conformer_manifest.csv"),
        _require(root / "data" / "test_conformers" / "af2" / "test1" / "cache_16_30" / "conformer_manifest.csv"),
    ]
    manifest_rows = _manifest_rows(root, source_paths)
    manifest_path = output_dir / "input_manifest.csv"
    _write_csv(manifest_path, manifest_rows)

    protocol_path = output_dir / "README.md"
    protocol_path.write_text(_protocol_markdown(args.mode), encoding="utf-8")
    top_1d = sorted(
        sequence_residue_rows,
        key=lambda row: abs(float(row["delta_probability_1d_baseline_minus_occluded"])),
        reverse=True,
    )[:10]
    top_3d = sorted(
        site_rows,
        key=lambda row: abs(float(row["delta_probability_3d_baseline_minus_deleted"])),
        reverse=True,
    )[:10]
    summary = {
        "artifact_type": "ghxtox_frozen_interpretability_results",
        "schema_version": 1,
        "mode": args.mode,
        "manuscript_admissible": bool(args.mode == "full"),
        "model_name": PUBLIC_MODEL_NAME,
        "operating_point": {
            "weight_1d": FUSION_WEIGHT_1D,
            "weight_3d": FUSION_WEIGHT_3D,
            "fusion_threshold": FUSION_THRESHOLD,
            "standalone_1d_threshold": ONE_D_THRESHOLD,
            "standalone_3d_threshold": THREE_D_THRESHOLD,
        },
        "cohort": {
            "selected_sample_count": len(cohort),
            "full_prespecified_sample_count": len(full_cohort),
            "selection_uses_attribution": False,
        },
        "one_d_baseline_validation": one_d_validation,
        "three_d_baseline_validation": three_d_validation,
        "result_row_counts": {
            "sequence_residue_occlusion": len(sequence_residue_rows),
            "sequence_sample_summary": len(sequence_sample_rows),
            "plddt_gate_samples": len(gate_sample_rows),
            "plddt_gate_residues": len(gate_residue_rows),
            "structure_site_deletion": len(site_rows),
            "structure_category_deletion": len(category_rows),
            "structure_residue_deletion": len(residue_deletion_rows),
            "expert_agreement_summary": len(agreement_rows),
        },
        "plddt_gate": gate_summary,
        "largest_absolute_1d_probability_shifts": top_1d,
        "largest_absolute_3d_site_deletion_probability_shifts": top_3d,
        "environment": {
            "python_runtime_constraint": "isolated GHXTox runtime; user PyTorch environment unchanged",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "interpretation_boundary": (
            "All reported values are frozen-model sensitivity or internal-use diagnostics. "
            "They do not establish residue-level biological causality, validate predicted "
            "structures as experimental truth, or measure intervention effects in vitro."
        ),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    output_paths = [
        cohort_path,
        sequence_residue_path,
        sequence_sample_path,
        gate_sample_path,
        gate_residue_path,
        site_path,
        category_path,
        residue_deletion_path,
        agreement_path,
        gate_summary_path,
        manifest_path,
        protocol_path,
        summary_path,
    ]
    _output_checksums(output_dir, output_paths)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
