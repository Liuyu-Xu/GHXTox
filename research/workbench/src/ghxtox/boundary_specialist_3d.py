"""Phase 33 fold-pure BioEmu boundary specialist.

This phase uses only the frozen 600-row decision-boundary BioEmu cohort.  A
fold-specific supervised 3D evidence head is trained on the other four folds,
then a fixed bounded logit residual is applied to the retained Phase 25 OOF
probability only for rows in the frozen boundary cohort.  All other rows remain
an exact Phase 25 fallback.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ghxtox.bootstrap_ci import _metrics
from ghxtox.ensemble_domain_ssl_3d import _length_mask, _sha256, _write_csv
from ghxtox.multiconformer_graph_3d import ConformerSample, build_multiconformer_samples
from ghxtox.oof_source_diagnostics import _read_csv
from ghxtox.partial_unfreeze_3d import (
    _combine_held_evidence,
    _load_initialized_model,
    _predict_evidence,
    _set_seed,
    _train_fold,
)
from ghxtox.utils import save_json


DEFAULT_OUTPUT_ROOT = Path("runs/bioemu_boundary_specialist_3d")
DEFAULT_PROTOCOL = DEFAULT_OUTPUT_ROOT / "protocol.json"
REFERENCE_MCC = 0.8390181404798325
REFERENCE_AUPRC = 0.9281708597568626
REFERENCE_BRIER = 0.05939949004168534
REFERENCE_ECE10 = 0.05110902710928455
REFERENCE_THRESHOLD = 0.677819


def _protocol_input_paths() -> dict[str, Any]:
    return {
        "fold_manifest": "data/folds/train_cdhit080_fallback_folds.csv",
        "legacy_manifests": [],
        "legacy_cache_roots": [],
        "legacy_source_names": [],
        "bioemu_manifest": "data/bioemu_boundary_600/cache_600/conformer_manifest.csv",
        "bioemu_cache_root": "data/bioemu_boundary_600/cache_600",
        "boundary_import_summary": "data/bioemu_boundary_600/cache_600/import_summary.json",
        "boundary_selection_summary": "data/bioemu_boundary_600/selection_summary.json",
        "server_generation_report": "runs/bioemu_boundary_600_server/generation_feasibility.json",
        "reference_oof": "runs/three_d_team_handoff/oof_3d_predictions.csv",
        "phase30_protocol": "runs/ensemble_domain_ssl_3d/protocol.json",
        "phase30_encoder_pattern": "runs/ensemble_domain_ssl_3d/fold_{fold}/encoder.pt",
    }


def _hash_inputs(inputs: dict[str, Any]) -> dict[str, str]:
    paths = [
        inputs["fold_manifest"],
        inputs["bioemu_manifest"],
        inputs["boundary_import_summary"],
        inputs["boundary_selection_summary"],
        inputs["server_generation_report"],
        inputs["reference_oof"],
        inputs["phase30_protocol"],
    ]
    paths.extend(
        inputs["phase30_encoder_pattern"].format(fold=fold) for fold in range(5)
    )
    return {path: _sha256(path) for path in paths}


def _baseline_by_index(reference_oof: str | Path) -> dict[int, dict[str, str]]:
    rows = _read_csv(reference_oof)
    baseline = {int(row["source_index"]): row for row in rows}
    if len(baseline) != len(rows):
        raise ValueError("Phase 25 reference OOF contains duplicate source_index rows.")
    return baseline


def _load_boundary_samples(
    inputs: dict[str, Any],
) -> tuple[list[ConformerSample], dict[str, Any]]:
    return build_multiconformer_samples(
        inputs["fold_manifest"],
        inputs["legacy_manifests"],
        inputs["legacy_cache_roots"],
        inputs["legacy_source_names"],
        inputs["bioemu_manifest"],
        inputs["bioemu_cache_root"],
    )


def _cohort_summary(
    samples: list[ConformerSample], cohort: dict[str, Any]
) -> dict[str, Any]:
    return {
        "canonical_boundary_samples": int(cohort["samples"]),
        "legacy_samples": int(cohort["legacy_samples"]),
        "bioemu_samples": int(cohort["bioemu_samples"]),
        "matched_legacy_bioemu_pairs": int(cohort["paired_samples"]),
        "bioemu_only_samples": int(cohort["bioemu_only"]),
        "folds": {
            str(key): value
            for key, value in sorted(Counter(sample.fold for sample in samples).items())
        },
        "length_strata": {
            key: value
            for key, value in sorted(
                Counter(
                    "11-15"
                    if len(sample.sequence) <= 15
                    else "16-30"
                    if len(sample.sequence) <= 30
                    else "31-50"
                    for sample in samples
                ).items()
            )
        },
        "positive": int(sum(sample.label for sample in samples)),
        "negative": int(len(samples) - sum(sample.label for sample in samples)),
        "source": "unified BioEmu-v1.2 only",
    }


def prepare_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    """Freeze the Phase 33 training/evaluation contract before candidate OOF."""

    output = Path(path)
    inputs = _protocol_input_paths()
    samples, cohort = _load_boundary_samples(inputs)
    import_summary = json.loads(
        Path(inputs["boundary_import_summary"]).read_text(encoding="utf-8")
    )
    server_report = json.loads(
        Path(inputs["server_generation_report"]).read_text(encoding="utf-8")
    )
    if import_summary.get("status") != "ready_for_phase33_boundary_specialist":
        raise ValueError(
            "Boundary cache is not ready for Phase 33 specialist training."
        )
    if not bool(import_summary.get("boundary_specialist_training_authorized")):
        raise PermissionError("Boundary specialist training is not authorized.")
    if server_report.get("decision") != "ready_to_import_phase33_boundary_structures":
        raise ValueError("Server BioEmu report did not pass the import gate.")

    protocol = {
        "schema_version": 1,
        "phase": 33,
        "status": "frozen_before_candidate_oof",
        "scope": "fold-pure unified BioEmu decision-boundary specialist with exact Phase 25 fallback outside the frozen 600 rows",
        "external_test_labels_read": False,
        "inputs": {**inputs, "sha256": _hash_inputs(inputs)},
        "cohort": {
            **_cohort_summary(samples, cohort),
            "generated": int(import_summary["generated"]),
            "reused_from_pilot": int(import_summary["reused"]),
            "total_filtered_conformers": int(
                import_summary["total_filtered_conformers"]
            ),
        },
        "initialization": {
            "encoder": "corresponding outer-fold Phase 30 GeometrySSLModel checkpoint",
            "structure_feature_dim": 16,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "trainable_encoder_modules": ["encoder.layers.1"],
            "frozen_encoder_modules": [
                "encoder.input_projection",
                "encoder.rbf",
                "encoder.layers.0",
                "encoder.norm",
            ],
            "unused_frozen_modules": ["node_decoder", "pair_decoder"],
        },
        "fine_tuning": {
            "toxicity_training_inputs": "BioEmu ensemble tensors and labels only; Phase 25 scores prohibited during training",
            "head": "LayerNorm projection over per-residue conformer mean/std, residue attention plus masked max pooling, one standalone logit",
            "head_hidden_dim": 64,
            "head_dropout": 0.1,
            "epochs": 8,
            "batch_size": 32,
            "training_conformers_per_sample": 2,
            "inference_max_conformers_per_sample": 8,
            "encoder_learning_rate": 0.00001,
            "head_learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 5.0,
            "seed": 20260803,
            "loss": "binary cross entropy with outer-training-fold positive weight equal to negatives divided by positives",
            "early_stopping": False,
            "checkpoint_selection": "fixed final epoch",
            "threshold_search": False,
            "seed_search": False,
            "fusion_formula": "held_phase25_logit + 0.15*tanh(standalone_3d_logit)",
            "maximum_absolute_logit_delta": 0.15,
            "missing_policy": "exact Phase 25 probability fallback outside the frozen 600 boundary rows",
        },
        "evaluation": {
            "decision_threshold": REFERENCE_THRESHOLD,
            "reference": {
                "mcc": REFERENCE_MCC,
                "auprc": REFERENCE_AUPRC,
                "brier": REFERENCE_BRIER,
                "ece_10": REFERENCE_ECE10,
            },
            "length_reference_mcc": {
                "11-15": 0.6504267492346457,
                "16-30": 0.7812252492583115,
                "31-50": 0.8919053414967528,
            },
            "length_reference_auprc": {
                "11-15": 0.6925766243278225,
                "16-30": 0.9135098658052947,
                "31-50": 0.9516045421689131,
            },
            "source_probe": "not applicable: Phase 33 boundary cache contains one unified BioEmu generator source only",
            "external_test_labels_read": False,
        },
        "promotion_gate": {
            "global_mcc_strictly_above": REFERENCE_MCC,
            "global_auprc_at_least": REFERENCE_AUPRC,
            "maximum_brier": REFERENCE_BRIER + 0.001,
            "maximum_ece_10": REFERENCE_ECE10 + 0.002,
            "minimum_fold_mcc_non_decreases": 3,
            "maximum_length_mcc_regression": 0.005,
            "maximum_length_auprc_regression": 0.002,
            "minimum_net_decision_corrections": 1,
            "source_robustness_gate": "not_applicable_unified_bioemu_only",
            "all_conditions_required": True,
        },
        "prohibitions": [
            "Phase 25 score or logit use during supervised training",
            "Test1/Test2 access",
            "post-hoc architecture, seed, epoch, threshold, evidence-weight, subset, checkpoint, or length-stratum selection",
            "full BioEmu generation",
            "3D handoff regeneration unless every promotion gate passes",
        ],
    }
    save_json(protocol, output)
    return protocol


def _load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_candidate_oof":
        raise ValueError("Phase 33 protocol must be frozen before execution.")
    if bool(protocol.get("external_test_labels_read", True)):
        raise ValueError("Phase 33 protocol must keep external labels sealed.")
    for input_path, expected in protocol["inputs"]["sha256"].items():
        actual = _sha256(input_path)
        if actual != expected:
            raise ValueError(
                f"Frozen input hash mismatch for {input_path}: {actual} != {expected}."
            )
    return protocol


def _load_and_validate_inputs(
    protocol_path: str | Path,
) -> tuple[
    dict[str, Any], list[ConformerSample], dict[str, Any], dict[int, dict[str, str]]
]:
    protocol = _load_protocol(protocol_path)
    inputs = protocol["inputs"]
    samples, cohort = _load_boundary_samples(inputs)
    observed = _cohort_summary(samples, cohort)
    for key in (
        "canonical_boundary_samples",
        "legacy_samples",
        "bioemu_samples",
        "matched_legacy_bioemu_pairs",
        "bioemu_only_samples",
    ):
        if int(protocol["cohort"][key]) != int(observed[key]):
            raise ValueError(f"Frozen cohort mismatch for {key}: {observed[key]}.")
    baseline = _baseline_by_index(inputs["reference_oof"])
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
        checkpoint_path = inputs["phase30_encoder_pattern"].format(fold=fold)
        _load_initialized_model(checkpoint_path, fold, protocol)
    return protocol, samples, observed, baseline


def verify_phase33(protocol_path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol, samples, observed, baseline = _load_and_validate_inputs(protocol_path)
    return {
        "status": "verified",
        "phase": int(protocol["phase"]),
        "cohort": observed,
        "reference_rows": len(baseline),
        "folds": sorted({sample.fold for sample in samples}),
        "trainable_encoder_modules": protocol["initialization"][
            "trainable_encoder_modules"
        ],
        "external_test_labels_read": False,
    }


def _load_resumed_fold_result(
    result_path: str | Path,
    *,
    fold: int,
    fold_seed: int,
    expected_held_indices: set[int],
    expected_train_rows: int,
    protocol_sha256: str,
    fine_tuning: dict[str, Any],
) -> dict[str, Any]:
    """Load only a result produced by the identical frozen fold protocol."""

    result = torch.load(result_path, map_location="cpu", weights_only=False)
    metadata = {
        "schema_version": 1,
        "phase": 33,
        "fold": int(fold),
        "seed": int(fold_seed),
        "protocol_sha256": protocol_sha256,
        "fine_tuning": fine_tuning,
    }
    for key, expected in metadata.items():
        if key not in result or result[key] != expected:
            raise ValueError(
                f"Incompatible Phase 33 resume result at {result_path}: {key}. "
                "Use a new output directory or rerun this fold."
            )
    declared_indices = [int(value) for value in result.get("held_source_indices", [])]
    if (
        len(declared_indices) != len(set(declared_indices))
        or set(declared_indices) != expected_held_indices
    ):
        raise ValueError(
            f"Incompatible Phase 33 resume result at {result_path}: held cohort."
        )
    history = result.get("history")
    if not isinstance(history, dict) or (
        int(history.get("fold", -1)) != int(fold)
        or int(history.get("train_rows", -1)) != int(expected_train_rows)
        or int(history.get("held_rows", -1)) != len(expected_held_indices)
    ):
        raise ValueError(
            f"Incompatible Phase 33 resume result at {result_path}: history."
        )
    predictions = result.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(
            f"Incompatible Phase 33 resume result at {result_path}: predictions."
        )
    predicted_indices = [int(row["source_index"]) for row in predictions]
    if (
        len(predicted_indices) != len(set(predicted_indices))
        or set(predicted_indices) != expected_held_indices
    ):
        raise ValueError(
            f"Incompatible Phase 33 resume result at {result_path}: prediction IDs."
        )
    for row in predictions:
        logit = float(row["standalone_logit"])
        probability = float(row["standalone_probability"])
        if not np.isfinite(logit) or not np.isfinite(probability):
            raise ValueError(
                f"Incompatible Phase 33 resume result at {result_path}: non-finite evidence."
            )
        expected_probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))
        if not 0.0 <= probability <= 1.0 or not np.isclose(
            probability, expected_probability, atol=1e-6, rtol=1e-6
        ):
            raise ValueError(
                f"Incompatible Phase 33 resume result at {result_path}: probability/logit mismatch."
            )
    return result


def _evaluate_predictions(
    baseline: dict[int, dict[str, str]],
    samples: list[ConformerSample],
    evidence: dict[int, dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
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
    boundary_indices = {sample.source_index for sample in samples}
    positions = {source_index: offset for offset, source_index in enumerate(ordered)}
    maximum_delta = float(protocol["fine_tuning"]["maximum_absolute_logit_delta"])
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
    by_fold: dict[str, Any] = {}
    fold_non_decreases = 0
    for fold in sorted(np.unique(folds).tolist()):
        selected = folds == fold
        reference_fold = _metrics(
            labels[selected], reference_probabilities[selected], threshold
        )
        candidate_fold = _metrics(
            labels[selected], candidate_probabilities[selected], threshold
        )
        fold_non_decreases += int(candidate_fold["mcc"] >= reference_fold["mcc"])
        by_fold[str(fold)] = {"reference": reference_fold, "candidate": candidate_fold}
    by_length: dict[str, Any] = {}
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

    boundary_mask = np.asarray(
        [source_index in boundary_indices for source_index in ordered]
    )
    reference_decision = reference_probabilities >= threshold
    candidate_decision = candidate_probabilities >= threshold
    reference_correct = reference_decision == labels
    candidate_correct = candidate_decision == labels
    corrections = int((~reference_correct & candidate_correct).sum())
    regressions = int((reference_correct & ~candidate_correct).sum())
    changed = int((reference_decision != candidate_decision).sum())
    boundary_corrections = int(
        (~reference_correct & candidate_correct & boundary_mask).sum()
    )
    boundary_regressions = int(
        (reference_correct & ~candidate_correct & boundary_mask).sum()
    )
    fallback_differences = np.abs(
        candidate_probabilities[~boundary_mask]
        - reference_probabilities[~boundary_mask]
    )

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
        "source_gate_not_applicable_unified_bioemu_only": True,
    }
    gate["all_conditions_passed"] = all(gate.values())

    rows = []
    for offset, source_index in enumerate(ordered):
        value = evidence.get(source_index)
        rows.append(
            {
                "source_index": source_index,
                "sample_id": baseline[source_index]["sample_id"],
                "fold": int(baseline[source_index]["fold"]),
                "label": int(baseline[source_index]["label"]),
                "sequence": baseline[source_index]["sequence"],
                "length": int(lengths[offset]),
                "phase33_boundary_member": int(source_index in boundary_indices),
                "bioemu_boundary_available": int(value is not None),
                "reference_probability": float(reference_probabilities[offset]),
                "standalone_3d_probability": float(standalone[offset])
                if value is not None
                else "",
                "bounded_logit_delta": float(deltas[offset]),
                "candidate_probability": float(candidate_probabilities[offset]),
                "decision_threshold": threshold,
                "reference_prediction": int(reference_decision[offset]),
                "candidate_prediction": int(candidate_decision[offset]),
            }
        )
    report = {
        "overall": {
            "reference": reference_metrics,
            "candidate": candidate_metrics,
            "standalone_boundary_3d": standalone_metrics,
        },
        "folds": by_fold,
        "length_strata": by_length,
        "decision_transitions": {
            "changed": changed,
            "corrected": corrections,
            "regressed": regressions,
            "net": corrections - regressions,
            "boundary_corrected": boundary_corrections,
            "boundary_regressed": boundary_regressions,
            "boundary_net": boundary_corrections - boundary_regressions,
        },
        "fallback": {
            "rows": int((~boundary_mask).sum()),
            "maximum_absolute_probability_difference": float(
                fallback_differences.max(initial=0.0)
            ),
        },
        "fold_mcc_non_decrease_count": int(fold_non_decreases),
        "gate": gate,
    }
    decision = {
        "status": "complete",
        "decision": "promote_phase33_3d_handoff"
        if gate["all_conditions_passed"]
        else "retain_phase25_3d_handoff",
        "promotion_authorized": bool(gate["all_conditions_passed"]),
        "handoff_regeneration_authorized": bool(gate["all_conditions_passed"]),
        "full_bioemu_generation_authorized": False,
        "external_test_labels_read": False,
        "overall": report["overall"],
        "decision_transitions": report["decision_transitions"],
        "gate": gate,
    }
    return rows, report, decision


def run_phase33(
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    output_root: str | Path | None = None,
    *,
    device_name: str = "auto",
    resume: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, samples, observed, baseline = _load_and_validate_inputs(protocol_path)
    output = Path(output_root or Path(protocol_path).parent)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    fine = protocol["fine_tuning"]
    seed = int(fine["seed"])
    protocol_sha256 = _sha256(protocol_path)
    evidence: dict[int, dict[str, Any]] = {}
    fold_histories = []

    for fold in sorted({sample.fold for sample in samples}):
        fold_root = output / f"fold_{fold}"
        fold_root.mkdir(exist_ok=True)
        result_path = fold_root / "result.pt"
        fold_seed = seed + fold
        train_samples = [sample for sample in samples if sample.fold != fold]
        held_samples = [sample for sample in samples if sample.fold == fold]
        if resume and result_path.exists():
            result = _load_resumed_fold_result(
                result_path,
                fold=fold,
                fold_seed=fold_seed,
                expected_held_indices={sample.source_index for sample in held_samples},
                expected_train_rows=len(train_samples),
                protocol_sha256=protocol_sha256,
                fine_tuning=fine,
            )
            for row in result["predictions"]:
                evidence[int(row["source_index"])] = row
            fold_histories.append(result["history"])
            print(f"phase33 fold={fold} resumed", flush=True)
            continue
        _set_seed(fold_seed)
        checkpoint_path = protocol["inputs"]["phase30_encoder_pattern"].format(
            fold=fold
        )
        model = _load_initialized_model(checkpoint_path, fold, protocol)
        history, trainable = _train_fold(model, train_samples, device, fine, fold_seed)
        held_predictions = _predict_evidence(model, held_samples, device, fine)
        for row in held_predictions:
            evidence[int(row["source_index"])] = row
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
                "schema_version": 1,
                "phase": 33,
                "fold": fold,
                "seed": fold_seed,
                "protocol_sha256": protocol_sha256,
                "fine_tuning": fine,
                "held_source_indices": sorted(
                    sample.source_index for sample in held_samples
                ),
                "predictions": held_predictions,
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
        print(f"phase33 fold={fold} complete", flush=True)

    expected_indices = {sample.source_index for sample in samples}
    if set(evidence) != expected_indices:
        raise RuntimeError("Phase 33 did not predict every frozen boundary row.")
    rows, report_body, decision = _evaluate_predictions(
        baseline, samples, evidence, protocol
    )
    _write_csv(rows, output / "oof_predictions.csv")
    report = {
        "status": "complete",
        "protocol": protocol,
        "device": str(device),
        "cohort": observed,
        **report_body,
        "fold_histories": fold_histories,
    }
    save_json(report, output / "summary.json")
    save_json(decision, output / "decision.json")
    return report, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-protocol", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_protocol:
        protocol = prepare_protocol(args.protocol)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "protocol": args.protocol,
                    "phase": protocol["phase"],
                },
                ensure_ascii=False,
            )
        )
        return
    if args.verify_only:
        print(json.dumps(verify_phase33(args.protocol), ensure_ascii=False))
        return
    _, decision = run_phase33(
        args.protocol,
        args.output_root,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps({"decision": decision["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
