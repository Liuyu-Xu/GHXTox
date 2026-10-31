"""V4-0 diagnostic: student uncertainty vs teacher helpfulness (OOF only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef

from ghxtox.prottox_3dkd_v2a_targets import _load_v1_fold_checkpoint


HELPFUL_THRESHOLD = 0.01


def _entropy(probability):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _logit(probability):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def run_diagnostic(effect_dir, cache_path, v1_fold_template, output):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = cache["records"]
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    seq_features = torch.cat((tri, phys), dim=1)
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.float64)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    historical_index = {str(record["historical_sample_id"]): index for index, record in enumerate(records)}
    teacher_probability = np.full(len(records), np.nan)
    base_probability = np.full(len(records), np.nan)
    for fold in range(5):
        payload = torch.load(Path(effect_dir) / f"v2c_component_effects_outer{fold}.pt", map_location="cpu", weights_only=False)
        z_full = payload["z_full"].numpy().reshape(-1)
        for position, sample_id in enumerate([str(value) for value in payload["sample_id"]]):
            index = historical_index[sample_id]
            teacher_probability[index] = 1.0 / (1.0 + np.exp(-z_full[position]))
        probabilities = []
        for seed in (20260905, 20260906):
            path = Path(str(v1_fold_template).format(seed=seed, k=fold))
            model, mean, std, _metadata, _payload = _load_v1_fold_checkpoint(path)
            with torch.inference_mode():
                probabilities.append(torch.sigmoid(model((seq_features - mean) / std.clamp_min(1e-6))).numpy())
        base_probability[folds == fold] = ((probabilities[0] + probabilities[1]) / 2.0)[folds == fold]
    if not np.isfinite(teacher_probability).all() or not np.isfinite(base_probability).all():
        raise ValueError("Teacher/base probability alignment incomplete")
    utility = (labels - base_probability) ** 2 - (labels - teacher_probability) ** 2
    helpful = utility > 0.01
    margin = np.abs(base_probability - 0.5)
    entropy = _entropy(base_probability)
    logit_magnitude = np.abs(_logit(base_probability))
    base_pred = (base_probability >= 0.5).astype(int)
    teacher_pred = (teacher_probability >= 0.5).astype(int)
    base_correct = base_pred == labels.astype(int)
    teacher_correct = teacher_pred == labels.astype(int)
    quantile_edges = np.quantile(margin, [0.2, 0.4, 0.6, 0.8])
    group_index = np.digitize(margin, quantile_edges, right=False)  # 0..4
    groups = []
    for certainty_quintile in range(1, 6):
        group = 5 - certainty_quintile  # group index 4 = largest margin = most certain
        mask = group_index == group
        if not np.any(mask):
            continue
        base_mcc = float(matthews_corrcoef(labels[mask].astype(int), base_pred[mask]))
        teacher_mcc = float(matthews_corrcoef(labels[mask].astype(int), teacher_pred[mask]))
        teacher_wins = int(np.sum(teacher_correct[mask] & ~base_correct[mask]))
        teacher_loses = int(np.sum(base_correct[mask] & ~teacher_correct[mask]))
        groups.append(
            {
                "uncertainty_quintile": certainty_quintile,
                "margin_range": [float(np.min(margin[mask])), float(np.max(margin[mask]))],
                "n": int(np.sum(mask)),
                "mean_margin": float(np.mean(margin[mask])),
                "mean_entropy": float(np.mean(entropy[mask])),
                "helpful_prevalence": float(np.mean(helpful[mask])),
                "mean_delta_brier": float(np.mean(utility[mask])),
                "base_mcc": base_mcc,
                "teacher_mcc": teacher_mcc,
                "base_tp_tn_fp_fn": [
                    int(np.sum((base_pred[mask] == 1) & (labels[mask] == 1))),
                    int(np.sum((base_pred[mask] == 0) & (labels[mask] == 0))),
                    int(np.sum((base_pred[mask] == 1) & (labels[mask] == 0))),
                    int(np.sum((base_pred[mask] == 0) & (labels[mask] == 1))),
                ],
                "teacher_tp_tn_fp_fn": [
                    int(np.sum((teacher_pred[mask] == 1) & (labels[mask] == 1))),
                    int(np.sum((teacher_pred[mask] == 0) & (labels[mask] == 0))),
                    int(np.sum((teacher_pred[mask] == 1) & (labels[mask] == 0))),
                    int(np.sum((teacher_pred[mask] == 0) & (labels[mask] == 1))),
                ],
                "teacher_wins_over_base": teacher_wins,
                "teacher_loses_to_base": teacher_loses,
                "net_corrections": teacher_wins - teacher_loses,
            }
        )
    overall = {
        "helpful_prevalence": float(np.mean(helpful)),
        "mean_utility": float(np.mean(utility)),
        "spearman_margin_vs_utility": float(spearmanr(margin, utility).statistic),
        "spearman_entropy_vs_utility": float(spearmanr(entropy, utility).statistic),
        "spearman_logit_magnitude_vs_utility": float(spearmanr(logit_magnitude, utility).statistic),
        "net_corrections_overall": int(np.sum(teacher_correct & ~base_correct) - np.sum(base_correct & ~teacher_correct)),
    }
    result = {
        "artifact_type": "prottox_3dkd_v4_uncertainty_diagnostic",
        "version": "v4-0",
        "uncertainty_definition": "|p_base - 0.5| quintiles; Q1 most certain, Q5 most uncertain",
        "utility_definition": "u=(y-p_base)^2-(y-p_teacher)^2; helpful if u>0.01",
        "quintiles": groups,
        "overall": overall,
        "monotonic_check": {
            "q1_to_q5_helpful_prevalence": [group["helpful_prevalence"] for group in groups],
            "q1_to_q5_mean_delta_brier": [group["mean_delta_brier"] for group in groups],
            "q1_to_q5_net_corrections": [group["net_corrections"] for group in groups],
        },
        "selection_scope": "training OOF descriptive diagnostic only; Test1/Test2 excluded",
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
