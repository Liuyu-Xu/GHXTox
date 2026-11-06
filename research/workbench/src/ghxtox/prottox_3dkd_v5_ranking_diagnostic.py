"""V5-0 ranking diagnostic: teacher pairwise ordering value under outer-safe OOF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from ghxtox.prottox_3dkd_v2a_targets import _load_v1_fold_checkpoint


def _pair_order_accuracy(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan"), 0
    ranks = rankdata(scores)
    rank_sum = float(np.sum(ranks[labels == 1]))
    u = rank_sum - len(positive) * (len(positive) + 1) / 2.0
    pairs = len(positive) * len(negative)
    accuracy = u / pairs
    return accuracy, pairs


def _correction_rate(pos_base, neg_base, pos_teacher, neg_teacher):
    pairs = len(pos_base) * len(neg_base)
    if pairs == 0:
        return float("nan"), 0
    # base misordered pair: p_toxic < p_nontoxic
    corrected = 0
    for p_t in pos_teacher:
        corrected += int(np.sum(p_t > neg_teacher)) + 0.5 * int(np.sum(p_t == neg_teacher))
    return corrected / pairs, pairs


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
            teacher_probability[historical_index[sample_id]] = 1.0 / (1.0 + np.exp(-z_full[position]))
        probabilities = []
        for seed in (20260905, 20260906):
            path = Path(str(v1_fold_template).format(seed=seed, k=fold))
            model, mean, std, _metadata, _payload = _load_v1_fold_checkpoint(path)
            with torch.inference_mode():
                probabilities.append(torch.sigmoid(model((seq_features - mean) / std.clamp_min(1e-6))).numpy())
        base_probability[folds == fold] = ((probabilities[0] + probabilities[1]) / 2.0)[folds == fold]
    if not np.isfinite(teacher_probability).all() or not np.isfinite(base_probability).all():
        raise ValueError("Alignment incomplete")
    folds_result = []
    for fold in range(5):
        mask = folds == fold
        y = labels[mask]
        p_base = base_probability[mask]
        p_teacher = teacher_probability[mask]
        base_order, base_pairs = _pair_order_accuracy(y, p_base)
        teacher_order, teacher_pairs = _pair_order_accuracy(y, p_teacher)
        pos_base = p_base[y == 1]
        neg_base = p_base[y == 0]
        misordered_mask = pos_base[:, None] < neg_base[None, :]
        # teacher correction rate over all base-misordered pairs
        pos_teacher = p_teacher[y == 1]
        neg_teacher = p_teacher[y == 0]
        correction_numerator = 0
        for index, p_t in enumerate(pos_teacher):
            correction_numerator += int(np.sum((p_t > neg_teacher)[misordered_mask[index]])) + 0.5 * int(np.sum((p_t == neg_teacher)[misordered_mask[index]]))
        total_misordered = int(np.sum(misordered_mask))
        correction_rate = correction_numerator / total_misordered if total_misordered else float("nan")
        folds_result.append(
            {
                "fold": fold,
                "n": int(np.sum(mask)),
                "base_auroc": float(roc_auc_score(y, p_base)),
                "teacher_auroc": float(roc_auc_score(y, p_teacher)),
                "base_auprc": float(average_precision_score(y, p_base)),
                "teacher_auprc": float(average_precision_score(y, p_teacher)),
                "base_pair_order_accuracy": base_order,
                "teacher_pair_order_accuracy": teacher_order,
                "total_pairs": int(base_pairs),
                "v1_misordered_pairs": total_misordered,
                "teacher_corrects_misordered_fraction": correction_rate,
            }
        )
    overall = {
        "teacher_auroc_mean": float(np.mean([item["teacher_auroc"] for item in folds_result])),
        "base_auroc_mean": float(np.mean([item["base_auroc"] for item in folds_result])),
        "teacher_pair_order_accuracy_mean": float(np.mean([item["teacher_pair_order_accuracy"] for item in folds_result])),
        "base_pair_order_accuracy_mean": float(np.mean([item["base_pair_order_accuracy"] for item in folds_result])),
        "teacher_corrects_misordered_fraction_mean": float(np.mean([item["teacher_corrects_misordered_fraction"] for item in folds_result])),
    }
    result = {
        "artifact_type": "prottox_3dkd_v5_ranking_diagnostic",
        "version": "v5-0",
        "scope": "outer-safe per-sample own-fold teacher/base; training OOF only",
        "folds": folds_result,
        "overall": overall,
        "stability_note": "report per-fold values; no selection or tuning after inspection",
        "test1_test2_used": False,
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
