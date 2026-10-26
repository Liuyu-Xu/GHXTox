"""Audit retained ProtTox-TSK MLP errors on aligned training OOF records only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ghxtox.models import PHYS_CHEM_FEATURE_NAMES, SPECTRAL_FEATURE_NAMES, spectral_physchem_feature_vector
from ghxtox.prottox_tsk import CACHE_FORMAT_VERSION, MODEL_NAME, MODEL_VERSION, file_sha256, write_manifest


AUDIT_VERSION = "prottox-tsk-oof-error-audit-v1"


def _cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CACHE_FORMAT_VERSION or payload.get("model_name") != MODEL_NAME or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("Error audit requires a compatible ProtTox-TSK training feature cache")
    records = payload.get("records", [])
    if not records or any(record.get("label") is None for record in records):
        raise ValueError("Error audit cache must include labels for every training record")
    return payload


def _oof(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "sequence_sha256", "label", "fold", "probability"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"OOF CSV must contain {sorted(required)}")
    parsed = [{"sample_id": str(row["sample_id"]), "sequence_sha256": str(row["sequence_sha256"]), "label": int(row["label"]), "fold": int(row["fold"]), "probability": float(row["probability"])} for row in rows]
    if len({row["sample_id"] for row in parsed}) != len(parsed):
        raise ValueError("OOF CSV contains duplicate sample IDs")
    if any(row["label"] not in {0, 1} or not math.isfinite(row["probability"]) or not 0.0 <= row["probability"] <= 1.0 for row in parsed):
        raise ValueError("OOF CSV has invalid labels or probabilities")
    return parsed


def _folds(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"sample_id", "fold"} <= set(rows[0]):
        raise ValueError("Fold CSV must contain sample_id and fold")
    result = {str(row["sample_id"]): int(row["fold"]) for row in rows}
    if len(result) != len(rows) or sorted(set(result.values())) != [0, 1, 2, 3, 4]:
        raise ValueError("Fold CSV must contain unique IDs and exactly folds 0..4")
    return result


def _effect(error_values: list[float], correct_values: list[float]) -> float:
    if len(error_values) < 2 or len(correct_values) < 2:
        return float("nan")
    error, correct = np.asarray(error_values, dtype=np.float64), np.asarray(correct_values, dtype=np.float64)
    pooled = math.sqrt((float(error.var(ddof=1)) + float(correct.var(ddof=1))) / 2.0)
    return 0.0 if pooled <= 1e-12 else float((error.mean() - correct.mean()) / pooled)


def _sign(value: float) -> int:
    return 1 if value > 1e-8 else -1 if value < -1e-8 else 0


def _summarize_comparison(rows: list[dict[str, Any]], *, error_kind: str, correct_kind: str, feature_names: tuple[str, ...]) -> dict[str, Any]:
    error_rows, correct_rows = [row for row in rows if row["error_type"] == error_kind], [row for row in rows if row["error_type"] == correct_kind]
    summaries: list[dict[str, Any]] = []
    for feature in feature_names:
        pooled = _effect([row[feature] for row in error_rows], [row[feature] for row in correct_rows])
        fold_effects: dict[str, float | None] = {}
        for fold in range(5):
            error_values = [row[feature] for row in error_rows if row["fold"] == fold]
            correct_values = [row[feature] for row in correct_rows if row["fold"] == fold]
            value = _effect(error_values, correct_values)
            fold_effects[str(fold)] = None if not math.isfinite(value) else value
        pooled_sign = _sign(pooled)
        matching_folds = sum(_sign(value) == pooled_sign for value in fold_effects.values() if value is not None and pooled_sign != 0)
        eligible_folds = sum(value is not None for value in fold_effects.values())
        summaries.append({"feature": feature, "error_mean": float(np.mean([row[feature] for row in error_rows])) if error_rows else float("nan"), "correct_mean": float(np.mean([row[feature] for row in correct_rows])) if correct_rows else float("nan"), "standardized_mean_difference": pooled, "fold_effects": fold_effects, "eligible_folds": eligible_folds, "matching_sign_folds": matching_folds, "stable": bool(math.isfinite(pooled) and abs(pooled) >= 0.20 and matching_folds >= 4)})
    summaries.sort(key=lambda item: abs(float(item["standardized_mean_difference"])) if math.isfinite(float(item["standardized_mean_difference"])) else -1.0, reverse=True)
    return {"error_kind": error_kind, "correct_kind": correct_kind, "error_records": len(error_rows), "correct_records": len(correct_rows), "features": summaries, "stable_features": [row for row in summaries if row["stable"]]}


def _cys_regime(count: float) -> str:
    value = int(round(count))
    return "0" if value == 0 else "1" if value == 1 else "2-3" if value <= 3 else "4+"


def _cys_regime_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for regime in ("0", "1", "2-3", "4+"):
        selected = [row for row in rows if _cys_regime(row["cysteine_count"]) == regime]
        toxic = [row for row in selected if row["label"] == 1]
        non_toxic = [row for row in selected if row["label"] == 0]
        false_negative = sum(row["error_type"] == "false_negative" for row in toxic)
        false_positive = sum(row["error_type"] == "false_positive" for row in non_toxic)
        summary.append({"cysteine_regime": regime, "records": len(selected), "toxic_records": len(toxic), "non_toxic_records": len(non_toxic), "false_negatives": false_negative, "false_positive": false_positive, "false_negative_rate_among_toxic": false_negative / len(toxic) if toxic else None, "false_positive_rate_among_non_toxic": false_positive / len(non_toxic) if non_toxic else None})
    return summary


def audit(cache_path: str | Path, oof_path: str | Path, fold_path: str | Path, output_dir: str | Path, *, threshold: float = 0.5) -> dict[str, Any]:
    cache_file, oof_file, folds_file = Path(cache_path), Path(oof_path), Path(fold_path)
    cache, oof_rows, folds = _cache(cache_file), _oof(oof_file), _folds(folds_file)
    records = {str(record["sample_id"]): record for record in cache["records"]}
    if len(records) != len(cache["records"]) or set(records) != {row["sample_id"] for row in oof_rows} or set(records) != set(folds):
        raise ValueError("Cache, OOF and fold manifests must contain the exact same unique sample IDs")
    feature_names = tuple(PHYS_CHEM_FEATURE_NAMES) + tuple(SPECTRAL_FEATURE_NAMES)
    ledger: list[dict[str, Any]] = []
    for row in oof_rows:
        record = records[row["sample_id"]]
        if str(record["sequence_sha256"]) != row["sequence_sha256"] or int(record["label"]) != row["label"] or folds[row["sample_id"]] != row["fold"]:
            raise ValueError(f"Identity, label, or fold mismatch for {row['sample_id']}")
        predicted = int(row["probability"] >= threshold)
        error_type = "true_positive" if row["label"] == predicted == 1 else "true_negative" if row["label"] == predicted == 0 else "false_positive" if predicted else "false_negative"
        values = [float(value) for value in torch.as_tensor(record["physchem"]).tolist()] + [float(value) for value in spectral_physchem_feature_vector(str(record["sequence"])).tolist()]
        ledger.append({"sample_id": row["sample_id"], "sequence_sha256": row["sequence_sha256"], "label": row["label"], "fold": row["fold"], "probability": row["probability"], "prediction": predicted, "margin_from_threshold": abs(row["probability"] - threshold), "error_type": error_type, **dict(zip(feature_names, values, strict=True))})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "error_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    counts = {kind: sum(row["error_type"] == kind for row in ledger) for kind in ("true_positive", "true_negative", "false_positive", "false_negative")}
    comparisons = {"false_negative_vs_true_positive": _summarize_comparison(ledger, error_kind="false_negative", correct_kind="true_positive", feature_names=feature_names), "false_positive_vs_true_negative": _summarize_comparison(ledger, error_kind="false_positive", correct_kind="true_negative", feature_names=feature_names)}
    report = {"artifact_type": AUDIT_VERSION, "scope": "training OOF only; Test1/Test2 and website excluded", "threshold": threshold, "source": {"cache": {"path": str(cache_file), "sha256": file_sha256(cache_file), "records": len(ledger)}, "oof": {"path": str(oof_file), "sha256": file_sha256(oof_file)}, "fold_manifest": {"path": str(folds_file), "sha256": file_sha256(folds_file)}}, "error_counts": counts, "cysteine_regimes": _cys_regime_summary(ledger), "feature_contract": list(feature_names), "stability_rule": "absolute pooled standardized mean difference >= 0.20 and matching pooled-sign effect in at least four eligible folds", "comparisons": comparisons}
    write_manifest(output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    report = audit(args.cache, args.oof, args.fold_manifest, args.output_dir, threshold=args.threshold)
    print(json.dumps({key: value["stable_features"][:5] for key, value in report["comparisons"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
