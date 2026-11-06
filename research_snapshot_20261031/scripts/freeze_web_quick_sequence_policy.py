"""Freeze the sequence-only web Quick Prediction policy from 1D OOF data.

This is an offline operator script. It reads only the frozen 1D seed OOF
artifacts, calibrates their three-seed mean, selects a BACC threshold, and
freezes a margin reject band at the requested OOF coverage. It never reads
external test labels and it does not modify the full GHXTox operating point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import csv


def logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values) - np.log1p(-values)


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(logits.reshape(-1, 1), labels)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def best_threshold(labels: np.ndarray, probabilities: np.ndarray, tolerance: float = 0.0005) -> float:
    from sklearn.metrics import balanced_accuracy_score
    candidates = np.sort(np.unique(probabilities))
    scores = [balanced_accuracy_score(labels, probabilities >= value) for value in candidates]
    best = max(scores)
    return float(max(value for value, score in zip(candidates, scores) if score >= best - tolerance))


def load_oof_ensemble(paths: list[str], manifest: str) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    with Path(manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = [str(row["sample_id"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    folds = [int(row["fold"]) for row in rows]
    probabilities, seeds = [], []
    for value in paths:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
        if [str(item) for item in payload["sample_ids"]] != expected_ids or [int(item) for item in payload["folds"]] != folds:
            raise ValueError(f"{value}: sample order or folds do not match the frozen manifest.")
        if not np.array_equal(np.asarray(payload["labels"], dtype=np.int64), labels):
            raise ValueError(f"{value}: labels do not match the frozen manifest.")
        seeds.append(int(payload["seed"]))
        probabilities.append(np.asarray(payload["oof"], dtype=np.float64))
    if tuple(sorted(seeds)) != (42, 123, 2025):
        raise ValueError(f"Expected frozen seeds (42, 123, 2025), found {sorted(seeds)}.")
    return np.mean(np.stack(probabilities), axis=0), labels, tuple(seeds)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/folds/train_cdhit080_fallback_folds.csv")
    parser.add_argument("--oof", nargs="+", default=[
        "runs/lora_mamba_full/oof_seed42.json",
        "runs/lora_mamba_full/oof_seed123.json",
        "runs/lora_mamba_full/oof_seed2025.json",
    ])
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--output", default="configs/web_quick_sequence_policy_v1.json")
    args = parser.parse_args()
    if not 0.5 < args.coverage < 1.0:
        raise SystemExit("--coverage must be in (0.5, 1.0).")

    mean_probability, labels, seeds = load_oof_ensemble(args.oof, args.manifest)
    slope, intercept = fit_platt(logit(mean_probability), labels)
    calibrated = 1.0 / (1.0 + np.exp(-(slope * logit(mean_probability) + intercept)))
    threshold = best_threshold(labels, calibrated)
    margins = np.abs(calibrated - threshold)
    rejected_count = len(margins) - int(round(args.coverage * len(margins)))
    if rejected_count < 1:
        raise SystemExit("Coverage leaves no rejected OOF observations.")
    # Reject the closest `rejected_count` observations. Stable ordering freezes ties.
    order = np.argsort(margins, kind="mergesort")
    reject_band = float(margins[order[rejected_count - 1]])
    high_margin = float(margins[order[int(round(0.20 * len(margins))) - 1]])
    payload = {
        "artifact_type": "ghxtox_web_quick_sequence_policy",
        "version": 1,
        "scope": "sequence-only three-seed 1D web quick prediction; independent from full fusion",
        "training_source": "frozen training OOF only",
        "external_test_labels_used": False,
        "model": {
            "experiment": "lora_mamba_1d_combinationA_autodl",
            "seeds": list(seeds),
            "probability": "Platt-calibrated mean of the three full 1D seed probabilities",
            "calibration": {"slope": float(slope), "intercept": float(intercept)},
        },
        "decision": {
            "threshold": float(threshold),
            "threshold_selection": "maximum OOF balanced accuracy; highest value within 0.0005 of the maximum plateau",
        },
        "reject": {
            "target_coverage": float(args.coverage),
            "realized_training_oof_coverage": float(np.mean(margins > reject_band)),
            "reject_band": reject_band,
            "rule": "abs(p_seq - threshold) <= reject_band",
            "label": "Uncertain",
        },
        "confidence": {
            "review": "rejected by the frozen reject band",
            "moderate": "accepted and abs(p_seq - threshold) <= high_margin",
            "high": "accepted and abs(p_seq - threshold) > high_margin",
            "high_margin": high_margin,
        },
        "inputs": [{"path": str(Path(value).as_posix()), "sha256": sha256(Path(value))} for value in [args.manifest, *args.oof]],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
