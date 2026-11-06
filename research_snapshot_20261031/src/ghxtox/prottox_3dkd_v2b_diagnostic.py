"""Cheap V2-B feasibility diagnostic for teacher pooled representation KD."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _balanced_accuracy(labels, pred):
    positive = labels == 1
    sensitivity = np.mean(pred[positive] == 1) if np.any(positive) else float("nan")
    specificity = np.mean(pred[~positive] == 0) if np.any(~positive) else float("nan")
    return float((sensitivity + specificity) / 2)


def _metrics(labels, probability):
    prediction = (probability >= 0.5).astype(int)
    labels = labels.astype(int)
    return {
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "balanced_accuracy": _balanced_accuracy(labels, prediction),
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "n": int(len(labels)),
        "n_positive": int(np.sum(labels)),
    }


def run_diagnostic(pooled_dir, cache_path, v1_oof_csv, output):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = cache["records"]
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.float64)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    seq_features = torch.cat((tri, phys), dim=1).numpy()
    cache_historical_index = {str(record["historical_sample_id"]): index for index, record in enumerate(records)}
    v1_rows = _read_csv(v1_oof_csv)
    v1_probability = np.full(len(records), np.nan)
    for row in v1_rows:
        v1_probability[cache_historical_index[str(row["historical_sample_id"])]] = float(row["probability"])
    if not np.isfinite(v1_probability).all():
        raise ValueError("V1 fold-base OOF probabilities are incomplete")
    pooled = {}
    for fold in range(5):
        payload = torch.load(Path(pooled_dir) / f"teacher_pooled_outer{fold}.pt", map_location="cpu", weights_only=False)
        sample_ids = [str(sample_id) for sample_id in payload["sample_id"]]
        embedding = payload["embedding"].numpy()
        if len(sample_ids) != len(records):
            raise ValueError("Pooled export row count mismatch")
        aligned = np.full((len(records), embedding.shape[1]), np.nan)
        for position, sample_id in enumerate(sample_ids):
            index = cache_historical_index.get(sample_id)
            if index is None:
                raise ValueError(f"No cache row for pooled sample {sample_id}")
            aligned[index] = embedding[position]
        if not np.isfinite(aligned).all():
            raise ValueError("Pooled alignment failed")
        pooled[fold] = aligned

    # Per-evaluation fold k, the valid teacher-k pooled matrix is pooled[k].
    results = {}
    fold_redundancy = []
    for fold in range(5):
        train_mask = folds != fold
        test_mask = folds == fold
        y_train = labels[train_mask]
        y_test = labels[test_mask]
        v1_errors = ((v1_probability >= 0.5).astype(int) != labels.astype(int)) & test_mask
        fold_entry = {"fold": fold, "probes": {}, "v1_error_subset": {"n": int(np.sum(v1_errors))}}
        for name, matrix in (("tri_physchem", seq_features), ("teacher_pooled", pooled[fold]), ("concat", np.concatenate([seq_features, pooled[fold]], axis=1))):
            scaler = StandardScaler().fit(matrix[train_mask])
            x_train = scaler.transform(matrix[train_mask])
            x_test = scaler.transform(matrix[test_mask])
            probe = LogisticRegression(max_iter=3000, class_weight="balanced")
            probe.fit(x_train, y_train)
            probability = probe.predict_proba(x_test)[:, 1]
            fold_entry["probes"][name] = _metrics(y_test, probability)
            if np.any(v1_errors):
                subset_prob = probability[v1_errors[test_mask]]
                subset_pred = (subset_prob >= 0.5).astype(int)
                subset_labels = labels[test_mask][v1_errors[test_mask]].astype(int)
                fold_entry["v1_error_subset"][f"{name}_accuracy"] = float(np.mean(subset_pred == subset_labels))
                fold_entry["v1_error_subset"][f"{name}_corrected"] = int(np.sum(subset_pred == subset_labels))
        # Redundancy: can TriStat+PhysChem linearly reconstruct teacher pooled[k]?
        ridge = Ridge(alpha=1.0)
        ridge.fit(seq_features[train_mask], pooled[fold][train_mask])
        predicted = ridge.predict(seq_features[test_mask])
        target = pooled[fold][test_mask]
        residual = target - predicted
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((target - target.mean(axis=0)) ** 2))
        fold_entry["redundancy_ridge_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        try:
            cca = CCA(n_components=min(5, pooled[fold].shape[1], seq_features.shape[1]))
            cca.fit(seq_features[train_mask], pooled[fold][train_mask])
            x_c, y_c = cca.transform(seq_features[test_mask], pooled[fold][test_mask])
            correlations = [float(np.corrcoef(x_c[:, i], y_c[:, i])[0, 1]) for i in range(x_c.shape[1])]
            fold_entry["first_canonical_correlation"] = correlations[0]
        except Exception as error:
            fold_entry["first_canonical_correlation"] = None
            fold_entry["cca_error"] = str(error)
        fold_redundancy.append(fold_entry)
        results[fold] = fold_entry
    summary = {
        "artifact_type": "prottox_3dkd_v2b_feasibility_diagnostic",
        "version": "diagnostic-v1",
        "pooled_dimension": 256,
        "definition": "teacher result['embedding'] pre-classifier pooled representation",
        "probe": "logistic regression, class_weight=balanced, per-fold StandardScaler",
        "folds": results,
        "selection_scope": "training OOF only; Test1/Test2 excluded",
        "caveat": "Descriptive feasibility only; does not authorize V2-B training by itself.",
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--v1-oof-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_diagnostic(args.pooled_dir, args.cache, args.v1_oof_csv, args.output)


if __name__ == "__main__":
    main()
