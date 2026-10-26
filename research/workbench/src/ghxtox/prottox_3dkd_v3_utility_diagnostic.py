"""V3-0 Teacher Utility Predictability Diagnostic (frozen protocol)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ghxtox.prottox_3dkd_v2a_targets import _load_v1_fold_checkpoint


HELPFUL_THRESHOLD = 0.01
RIDGE_ALPHA = 10.0


def _brier(label, probability):
    return float(np.mean((label - probability) ** 2))


def _mcc(label, probability):
    return float(matthews_corrcoef(label.astype(int), (probability >= 0.5).astype(int)))


def _entropy(probability):
    raw = np.asarray(probability, dtype=np.float64)
    p = np.where(np.isfinite(raw), np.clip(raw, 1e-12, 1 - 1e-12), 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def run_diagnostic(effect_dir, cache_path, v1_fold_template, output):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = cache["records"]
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    seq_features = torch.cat((tri, phys), dim=1).numpy()
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.float64)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    historical_index = {str(record["historical_sample_id"]): index for index, record in enumerate(records)}

    teacher_probability = {}
    for fold in range(5):
        payload = torch.load(Path(effect_dir) / f"v2c_component_effects_outer{fold}.pt", map_location="cpu", weights_only=False)
        z_full = payload["z_full"].numpy().reshape(-1)
        values = np.full(len(records), np.nan)
        for position, sample_id in enumerate([str(value) for value in payload["sample_id"]]):
            values[historical_index[sample_id]] = z_full[position]
        teacher_probability[fold] = 1.0 / (1.0 + np.exp(-values))

    base_probability = {}
    for fold in range(5):
        probabilities = []
        for seed in (20260905, 20260906):
            path = Path(str(v1_fold_template).format(seed=seed, k=fold))
            model, mean, std, _metadata, _payload = _load_v1_fold_checkpoint(path)
            with torch.inference_mode():
                logits = model((torch.as_tensor(seq_features) - mean) / std.clamp_min(1e-6))
            probabilities.append(torch.sigmoid(logits).numpy())
        base_probability[fold] = (probabilities[0] + probabilities[1]) / 2.0

    utility_by_fold = {}
    for fold in range(5):
        utility_by_fold[fold] = (labels - base_probability[fold]) ** 2 - (labels - teacher_probability[fold]) ** 2

    # Descriptive crossfitted table (not used by probes): sample in fold j uses teacher_j/base_j.
    crossfitted_utility = np.full(len(records), np.nan)
    for index in range(len(records)):
        crossfitted_utility[index] = utility_by_fold[int(folds[index])][index]
    prevalence_overall = float(np.mean(crossfitted_utility > 0.01))

    probe_results = {}
    for eval_fold in range(5):
        train_mask = folds != eval_fold
        test_mask = folds == eval_fold
        y_utility = utility_by_fold[eval_fold]
        base_p = base_probability[eval_fold]
        margin = np.abs(base_p - 0.5)
        entropy = _entropy(base_p)
        feature_b = np.column_stack((seq_features, base_p, margin, entropy))
        fold_entry = {}
        for feature_name, matrix in (("representation_only", seq_features), ("deployment_aware_confidence", feature_b)):
            if not np.isfinite(matrix).all():
                raise ValueError(f"{feature_name} contains non-finite values for probe fold {eval_fold}")
            scaler = StandardScaler().fit(matrix[train_mask])
            x_train = scaler.transform(matrix[train_mask])
            x_test = scaler.transform(matrix[test_mask])
            ridge = Ridge(alpha=RIDGE_ALPHA)
            ridge.fit(x_train, y_utility[train_mask])
            ridge_pred = ridge.predict(x_test)
            helpful = y_utility > 0.01
            logistic = LogisticRegression(max_iter=3000)
            logistic.fit(x_train, helpful[train_mask])
            helpful_prob = logistic.predict_proba(x_test)[:, 1]
            fold_entry[feature_name] = {
                "helpful_prevalence_valid": float(np.mean(helpful[test_mask])),
                "auroc_helpful": float(roc_auc_score(helpful[test_mask], helpful_prob)),
                "auprc_helpful": float(average_precision_score(helpful[test_mask], helpful_prob)),
                "spearman_utility": float(spearmanr(y_utility[test_mask], ridge_pred).statistic),
                "r2_utility": float(1 - np.sum((y_utility[test_mask] - ridge_pred) ** 2) / np.sum((y_utility[test_mask] - y_utility[test_mask].mean()) ** 2)),
                "sign_accuracy": float(np.mean(np.sign(y_utility[test_mask]) == np.sign(ridge_pred))),
            }
        probe_results[eval_fold] = fold_entry
        print(f"probe outer fold {eval_fold} done", flush=True)

    def mean_across(name, key):
        return float(np.mean([probe_results[fold][name][key] for fold in range(5)]))

    def count_above(name, key, threshold):
        return int(np.sum([probe_results[fold][name][key] > threshold for fold in range(5)]))

    oracle_p = np.empty(len(labels))
    for index in range(len(labels)):
        fold = int(folds[index])
        oracle_p[index] = teacher_probability[fold][index] if crossfitted_utility[index] > 0 else base_probability[fold][index]
    oracle = {
        "base_brier": float(np.mean([_brier(labels[folds == fold], base_probability[fold][folds == fold]) for fold in range(5)])),
        "oracle_brier": float(np.mean([_brier(labels[folds == fold], oracle_p[folds == fold]) for fold in range(5)])),
        "base_mcc": float(np.mean([_mcc(labels[folds == fold], base_probability[fold][folds == fold]) for fold in range(5)])),
        "oracle_mcc": float(np.mean([_mcc(labels[folds == fold], oracle_p[folds == fold]) for fold in range(5)])),
        "helpful_prevalence_overall": prevalence_overall,
    }
    gate = {
        "deployment_aware_confidence": {
            "helpful_prevalence_ge_0.05": prevalence_overall >= 0.05,
            "mean_auroc_ge_0.65": mean_across("deployment_aware_confidence", "auroc_helpful") >= 0.65,
            "at_least_4_of_5_auroc_gt_0.60": count_above("deployment_aware_confidence", "auroc_helpful", 0.60) >= 4,
            "mean_spearman_ge_0.20": mean_across("deployment_aware_confidence", "spearman_utility") >= 0.20,
        }
    }
    gate["deployment_aware_confidence"]["passed"] = all(gate["deployment_aware_confidence"].values())
    result = {
        "artifact_type": "prottox_3dkd_v3_utility_diagnostic",
        "version": "v3-0",
        "utility_definition": "u = (y - p_base)^2 - (y - p_teacher)^2",
        "helpful_definition": "u > 0.01",
        "probe_folds": probe_results,
        "aggregate": {
            "representation_only": {key: mean_across("representation_only", key) for key in ("helpful_prevalence_valid", "auroc_helpful", "auprc_helpful", "spearman_utility", "r2_utility", "sign_accuracy")},
            "deployment_aware_confidence": {key: mean_across("deployment_aware_confidence", key) for key in ("helpful_prevalence_valid", "auroc_helpful", "auprc_helpful", "spearman_utility", "r2_utility", "sign_accuracy")},
            "auroc_fold_counts_gt_0.60": {
                "representation_only": count_above("representation_only", "auroc_helpful", 0.60),
                "deployment_aware_confidence": count_above("deployment_aware_confidence", "auroc_helpful", 0.60),
            },
        },
        "oracle_upper_bound": oracle,
        "gate": gate,
        "selection_scope": "training OOF only; Test1/Test2 excluded; no frontend or v1 change",
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--v1-fold-template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_diagnostic(args.effect_dir, args.cache, args.v1_fold_template, args.output)


if __name__ == "__main__":
    main()
