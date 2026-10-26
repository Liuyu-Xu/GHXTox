"""V6-0 Retrieval Memory Diagnostic (outer-safe bank, no model training)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


K_NEIGHBORS = (5, 11)


def _entropy_bits(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def run_diagnostic(cache_path, v1_oof_csv, output):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = cache["records"]
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records]).numpy()
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records]).numpy()
    features = np.concatenate([tri, phys], axis=1)
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.float64)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    rows = list(csv.DictReader(Path(v1_oof_csv).open("r", encoding="utf-8-sig", newline="")))
    v1_probability = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    if len(v1_probability) != len(records):
        raise ValueError("v1 OOF probability length mismatch")
    v1_pred = (v1_probability >= 0.5).astype(int)
    folds_out = []
    all_local_p = {}
    for k in K_NEIGHBORS:
        all_local_p[k] = np.full(len(records), np.nan)
    for fold in range(5):
        bank_mask = folds != fold
        query_mask = folds == fold
        scaler = StandardScaler().fit(features[bank_mask])
        bank_features = scaler.transform(features[bank_mask])
        query_features = scaler.transform(features[query_mask])
        bank_features = bank_features / np.linalg.norm(bank_features, axis=1, keepdims=True).clip(min=1e-12)
        query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True).clip(min=1e-12)
        similarity = query_features @ bank_features.T  # [Q, bank]
        fold_entry = {"fold": fold, "bank_n": int(np.sum(bank_mask)), "query_n": int(np.sum(query_mask))}
        query_labels = labels[query_mask]
        for k in K_NEIGHBORS:
            top_indices = np.argsort(-similarity, axis=1)[:, :k]
            top_sim = np.take_along_axis(similarity, top_indices, axis=1)
            neighbor_labels = labels[bank_mask][top_indices]
            distance = np.clip(1.0 - top_sim, 1e-6, None)
            weights = 1.0 / (1.0 + distance)
            local_p = np.sum(weights * neighbor_labels, axis=1) / np.sum(weights, axis=1)
            density = float(np.mean(weights))
            local_entropy = _entropy_bits(local_p)
            reliability = density * (1.0 - local_entropy)
            all_local_p[k][query_mask] = local_p
            fold_entry[k] = {
                "auroc": float(roc_auc_score(query_labels, local_p)),
                "auprc": float(average_precision_score(query_labels, local_p)),
                "mean_local_p": float(np.mean(local_p)),
                "mean_density": density,
                "mean_reliability": float(np.mean(reliability)),
                "spearman_vs_v1": float(spearmanr(local_p, v1_probability[query_mask]).statistic),
                "v1_error_direction_correct": None,
            }
        folds_out.append(fold_entry)
        # Direction correctness on v1 errors for k=5
        v1_errors = (v1_pred != labels.astype(int)) & query_mask
        if np.any(v1_errors):
            local_p5 = all_local_p[5][v1_errors]
            error_labels = labels[v1_errors]
            # FP: v1 predicts toxic but label 0 -> correct local direction if local_p < 0.5
            # FN: v1 predicts non-toxic but label 1 -> correct if local_p > 0.5
            expected_direction = np.where(error_labels == 1, 1.0, 0.0)
            correct = (local_p5 > 0.5) == (expected_direction > 0.5)
            folds_out[-1][5]["v1_error_direction_correct"] = float(np.mean(correct))
            folds_out[-1][5]["v1_error_n"] = int(np.sum(v1_errors))
    summary = {
        "artifact_type": "retrieval_memory_v6_diagnostic",
        "version": "v6-0",
        "method": "cosine similarity over standardized TriStat+PhysChem; weighted kNN local toxicity probability",
        "folds": folds_out,
        "aggregate": {},
        "selection_scope": "training OOF only; Test1/Test2 excluded",
    }
    for k in K_NEIGHBORS:
        summary["aggregate"][f"k{k}"] = {
            metric: float(np.mean([entry[k][metric] for entry in folds_out]))
            for metric in ("auroc", "auprc", "spearman_vs_v1")
        }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--v1-oof-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_diagnostic(args.cache, args.v1_oof_csv, args.output)


if __name__ == "__main__":
    main()
