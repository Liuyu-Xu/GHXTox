"""Training-OOF-only Quick-to-Full routing analyses for GHXTox.

This module never trains a toxicity expert.  It compares the frozen web Quick
policy with the already-produced canonical Full OOF fusion, then optionally
fits a cross-fitted expected-Brier-gain router using only features available
before ESMFold is executed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, matthews_corrcoef

from ghxtox.unified_inference import read_csv, sha256

EPSILON = 1e-6


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, EPSILON, 1.0 - EPSILON)
    return np.log(values / (1.0 - values))


def _load_policy(path: str | Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if policy.get("artifact_type") != "ghxtox_web_quick_sequence_policy":
        raise ValueError("Expected frozen web Quick sequence policy")
    if policy.get("training_source") != "frozen training OOF only":
        raise ValueError("Quick policy is not training-OOF-only")
    return policy


def _load_seed_mean(paths: list[str | Path], rows: int) -> np.ndarray:
    streams: list[np.ndarray] = []
    for path in paths:
        document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        values = np.asarray(document.get("oof"), dtype=np.float64)
        if values.shape != (rows,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid ordered Quick seed OOF stream: {path}")
        streams.append(values)
    if len(streams) != 3:
        raise ValueError("Quick web policy requires exactly three seed OOF streams")
    return np.mean(np.stack(streams), axis=0)


def quick_sequence_features(sequence: str) -> dict[str, float | int]:
    """Deterministic descriptors available before a structure is generated."""
    if not sequence:
        raise ValueError("Sequence must be non-empty")
    length = len(sequence)
    count = lambda letters: sum(residue in letters for residue in sequence)
    positions = [index for index, residue in enumerate(sequence) if residue == "C"]
    gaps = [right - left for left, right in zip(positions, positions[1:])]
    protonated = lambda pka: 1.0 / (1.0 + 10.0 ** (7.0 - pka))
    deprotonated = lambda pka: 1.0 / (1.0 + 10.0 ** (pka - 7.0))
    charge = (
        protonated(9.69) + count("K") * protonated(10.5) + count("R") * protonated(12.4)
        + count("H") * protonated(6.0) - deprotonated(2.34) - count("D") * deprotonated(3.86)
        - count("E") * deprotonated(4.25) - count("C") * deprotonated(8.33)
        - count("Y") * deprotonated(10.07)
    )
    kyte_doolittle = {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
        "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2,
        "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
    }
    return {
        "sequence_length": length,
        "net_charge_ph7": charge,
        "hydropathy_kd": sum(kyte_doolittle[residue] for residue in sequence) / length,
        "basic_fraction": count("KRH") / length,
        "hydrophobic_fraction": count("AVILMFWY") / length,
        "aromatic_fraction": count("FWY") / length,
        "cys_count": len(positions),
        "cys_fraction": len(positions) / length,
        "cys_span_fraction": ((positions[-1] - positions[0]) / max(length - 1, 1)) if len(positions) >= 2 else 0.0,
        "cys_min_gap_fraction": (min(gaps) / max(length - 1, 1)) if gaps else 0.0,
        "cys_adjacent_pairs": sum(gap == 1 for gap in gaps),
        "cys_terminal_count": int(sequence[0] == "C") + int(sequence[-1] == "C"),
    }


def _validate_fold_groups(rows: list[dict[str, str]]) -> np.ndarray:
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("Routing input requires all five OOF folds")
    groups: dict[str, int] = {}
    for row, fold in zip(rows, folds):
        group = row["group_id"]
        if not group:
            raise ValueError("Routing input has an empty group_id")
        if group in groups and groups[group] != fold:
            raise ValueError("Group leakage across OOF folds")
        groups[group] = int(fold)
    return folds


def build_quick_full_baseline(
    contract_path: str | Path,
    quick_policy_path: str | Path,
    seed_oof_paths: list[str | Path],
    output_dir: str | Path,
    *,
    full_threshold: float,
) -> dict[str, Any]:
    """Materialize the frozen margin baseline and correction ledger."""
    contract_path, quick_policy_path, output_dir = map(Path, (contract_path, quick_policy_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite routing baseline: {output_dir}")
    rows = read_csv(contract_path)
    policy = _load_policy(quick_policy_path)
    folds = _validate_fold_groups(rows)
    raw_mean = _load_seed_mean(seed_oof_paths, len(rows))
    calibration = policy["model"]["calibration"]
    quick = _sigmoid(float(calibration["slope"]) * _logit(raw_mean) + float(calibration["intercept"]))
    expected_sequence = np.asarray([float(row["p_seq"]) for row in rows])
    # The unified contract stores the already Platt-calibrated canonical 1D
    # stream, while the archived seed OOF files are raw model probabilities.
    # Validate the transformed Quick probability, never the raw mean.
    if not np.allclose(quick, expected_sequence, rtol=1e-5, atol=5e-4):
        raise ValueError("Frozen Quick calibration does not reproduce unified p_seq")
    full = np.asarray([float(row["p_fusion"]) for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    quick_threshold = float(policy["decision"]["threshold"])
    reject_band = float(policy["reject"]["reject_band"])
    quick_binary = quick >= quick_threshold
    full_binary = full >= full_threshold
    quick_correct, full_correct = quick_binary == labels, full_binary == labels
    margin = np.abs(quick - quick_threshold)
    recommend_full = margin <= reject_band
    columns = [
        "source_index", "sample_id", "sequence", "label", "fold", "group_id", "p_quick", "p_full",
        "quick_margin", "quick_prediction", "full_prediction", "quick_correct", "full_correct",
        "full_corrects_quick", "full_regresses_quick", "recommend_full_margin", "realized_brier_gain",
        "sequence_seed_dispersion", *list(quick_sequence_features(rows[0]["sequence"]).keys()),
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = output_dir / "quick_full_oof_ledger.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows):
            features = quick_sequence_features(row["sequence"])
            writer.writerow({
                "source_index": row["source_index"], "sample_id": row["sample_id"], "sequence": row["sequence"],
                "label": int(labels[index]), "fold": int(folds[index]), "group_id": row["group_id"],
                "p_quick": repr(float(quick[index])), "p_full": repr(float(full[index])),
                "quick_margin": repr(float(margin[index])), "quick_prediction": int(quick_binary[index]),
                "full_prediction": int(full_binary[index]), "quick_correct": int(quick_correct[index]),
                "full_correct": int(full_correct[index]), "full_corrects_quick": int((not quick_correct[index]) and full_correct[index]),
                "full_regresses_quick": int(quick_correct[index] and (not full_correct[index])),
                "recommend_full_margin": int(recommend_full[index]),
                "realized_brier_gain": repr(float((quick[index] - labels[index]) ** 2 - (full[index] - labels[index]) ** 2)),
                "sequence_seed_dispersion": row["sequence_seed_dispersion"], **features,
            })

    def summary(mask: np.ndarray) -> dict[str, Any]:
        corrected = (~quick_correct & full_correct) & mask
        regressed = (quick_correct & ~full_correct) & mask
        return {
            "n": int(mask.sum()), "quick_error_rate": float((~quick_correct[mask]).mean()),
            "full_error_rate": float((~full_correct[mask]).mean()), "full_corrects_quick": int(corrected.sum()),
            "full_regresses_quick": int(regressed.sum()), "net_corrections": int(corrected.sum() - regressed.sum()),
            "mean_realized_brier_gain": float(np.mean((quick[mask] - labels[mask]) ** 2 - (full[mask] - labels[mask]) ** 2)),
        }

    summary_document = {
        "artifact_type": "ghxtox_quick_full_margin_baseline_oof", "version": 1,
        "training_oof_only": True, "external_test_labels_used": False,
        "contract": str(contract_path.resolve()), "contract_sha256": sha256(contract_path),
        "quick_policy": str(quick_policy_path.resolve()), "quick_policy_sha256": sha256(quick_policy_path),
        "quick_seed_oof": [{"path": str(Path(path).resolve()), "sha256": sha256(path)} for path in seed_oof_paths],
        "quick_definition": "three-seed OOF mean transformed by the frozen web Quick Platt calibration",
        "full_definition": "canonical p_fusion_current_default from the aligned training OOF contract",
        "quick_threshold": quick_threshold, "full_threshold": full_threshold,
        "margin_baseline": {"rule": "abs(p_quick - quick_threshold) <= reject_band", "reject_band": reject_band},
        "all_rows": summary(np.ones(len(rows), dtype=bool)), "margin_recommended_full": summary(recommend_full),
        "margin_direct_quick": summary(~recommend_full),
        "quick_brier": float(brier_score_loss(labels, quick)), "full_brier": float(brier_score_loss(labels, full)),
        "quick_mcc": float(matthews_corrcoef(labels, quick_binary)), "full_mcc": float(matthews_corrcoef(labels, full_binary)),
        "ledger": str(ledger_path.resolve()), "ledger_sha256": sha256(ledger_path),
    }
    _write_json(output_dir / "summary.json", summary_document)
    (output_dir / "README.md").write_text(
        "# Frozen Quick → Full margin baseline\n\n"
        "This is a training-OOF-only correction ledger. `recommend_full_margin=1` means the frozen web Quick policy would recommend canonical Full analysis; it is not an experimental toxicity label.\n",
        encoding="utf-8",
    )
    return summary_document


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> tuple[Ridge, np.ndarray, np.ndarray]:
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale == 0] = 1.0
    model = Ridge(alpha=alpha).fit((features - mean) / scale, target)
    return model, mean, scale


def run_quick_full_router(protocol_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Evaluate a pre-structure, cross-fitted Full-fusion expected-gain router."""
    protocol_path, output_dir = Path(protocol_path), Path(output_dir)
    config = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    if config.get("artifact_type") != "ghxtox_quick_full_adaptive_router_protocol" or config.get("status") != "frozen_before_oof":
        raise ValueError("Quick-to-Full router protocol must be frozen before OOF evaluation")
    ledger_path = Path(config["baseline_ledger"])
    if sha256(ledger_path) != config["baseline_ledger_sha256"]:
        raise ValueError("Baseline ledger hash differs from frozen router protocol")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Quick-to-Full router output: {output_dir}")
    rows = read_csv(ledger_path)
    folds = _validate_fold_groups(rows)
    features_names = list(config["pre_structure_feature_columns"])
    prohibited = {"label", "p_full", "full_correct", "full_corrects_quick", "full_regresses_quick", "realized_brier_gain"}
    if set(features_names) & prohibited:
        raise ValueError("Router contains post-Full fields or labels")
    features = np.asarray([[float(row[name]) for name in features_names] for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    quick = np.asarray([float(row["p_quick"]) for row in rows])
    full = np.asarray([float(row["p_full"]) for row in rows])
    target = (quick - labels) ** 2 - (full - labels) ** 2
    alpha, cost = float(config["model"]["alpha"]), float(config["route_policy"]["full_analysis_cost"])
    gains = np.empty(len(rows), dtype=np.float64)
    for held_fold in range(5):
        train, held = folds != held_fold, folds == held_fold
        model, mean, scale = _fit_ridge(features[train], target[train], alpha)
        gains[held] = model.predict((features[held] - mean) / scale)
    route = gains > cost
    baseline = np.asarray([int(row["recommend_full_margin"]) == 1 for row in rows])
    deployed, baseline_deployed = np.where(route, full, quick), np.where(baseline, full, quick)
    quick_correct = (quick >= float(config["quick_threshold"])) == labels
    full_correct = (full >= float(config["full_threshold"])) == labels
    selected_corrected, selected_regressed = ((~quick_correct & full_correct) & route), ((quick_correct & ~full_correct) & route)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "cross_fitted_routes.csv"
    fields = ["source_index", "sample_id", "fold", "group_id", "p_quick", "p_full", "predicted_brier_gain", "realized_brier_gain", "route_full", "margin_baseline_full"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({key: row[key] for key in fields[:4]} | {
                "p_quick": repr(float(quick[i])), "p_full": repr(float(full[i])),
                "predicted_brier_gain": repr(float(gains[i])), "realized_brier_gain": repr(float(target[i])),
                "route_full": int(route[i]), "margin_baseline_full": int(baseline[i]),
            })
    utility = lambda probability, calls: float(np.mean((probability - labels) ** 2) + cost * calls.mean())
    result = {
        "artifact_type": "ghxtox_quick_full_adaptive_router_oof", "version": 1,
        "training_oof_only": True, "external_test_labels_used": False, "cross_fitted": True,
        "protocol": str(protocol_path.resolve()), "protocol_sha256": sha256(protocol_path),
        "baseline_ledger": str(ledger_path.resolve()), "baseline_ledger_sha256": sha256(ledger_path),
        "features": features_names, "post_full_features_used": False,
        "route_policy": config["route_policy"], "structure_rate": float(route.mean()), "margin_baseline_rate": float(baseline.mean()),
        "router": {"brier": float(brier_score_loss(labels, deployed)), "cost_adjusted_utility": utility(deployed, route), "mean_realized_gain_selected": float(target[route].mean()) if route.any() else None, "corrected": int(selected_corrected.sum()), "regressed": int(selected_regressed.sum()), "net_corrections": int(selected_corrected.sum() - selected_regressed.sum())},
        "margin_baseline": {"brier": float(brier_score_loss(labels, baseline_deployed)), "cost_adjusted_utility": utility(baseline_deployed, baseline), "mean_realized_gain_selected": float(target[baseline].mean()), "corrected": int(((~quick_correct & full_correct) & baseline).sum()), "regressed": int(((quick_correct & ~full_correct) & baseline).sum()), "net_corrections": int((((~quick_correct & full_correct) & baseline).sum()) - ((quick_correct & ~full_correct) & baseline).sum())},
        "quick_only": {"brier": float(brier_score_loss(labels, quick)), "cost_adjusted_utility": utility(quick, np.zeros(len(rows), dtype=bool))},
        "output": str(output_path.resolve()), "output_sha256": sha256(output_path),
    }
    gate = config["promotion_gate"]
    result["promotion"] = {
        "decision": "promote" if (result["router"]["cost_adjusted_utility"] < result["margin_baseline"]["cost_adjusted_utility"] and result["router"]["net_corrections"] > 0 and result["router"]["mean_realized_gain_selected"] > cost) else "retain_margin_baseline",
        "criteria": gate,
    }
    _write_json(output_dir / "summary.json", result)
    return result
