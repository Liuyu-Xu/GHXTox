"""Pre-registered head-to-head analysis: proxy_teacher_kd vs mlp_teacher_kd."""
from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

import numpy as np


SEEDS = (20260905, 20260906)
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20260909
STRATUM_QUANTILES = {"predicted_mean_plddt": 0.25, "predicted_low_plddt_fraction": 0.75, "teacher_mlp_disagreement": 0.75}
BRIER_TOLERANCE = 0.002
MCC_FLOOR = -0.002


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fast_metrics(labels, scores):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pred = scores >= 0.5
    tp = float(np.sum((pred == 1) & (labels == 1)))
    tn = float(np.sum((pred == 0) & (labels == 0)))
    fp = float(np.sum((pred == 1) & (labels == 0)))
    fn = float(np.sum((pred == 0) & (labels == 1)))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    brier = float(np.mean((labels - scores) ** 2))
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    positive_total = int(np.sum(y))
    if positive_total == 0:
        return {"mcc": mcc, "auprc": float("nan"), "brier": brier}
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    precision = tps / np.maximum(1.0, tps + fps)
    recall = tps / positive_total
    positive_indices = np.flatnonzero(y)
    recall_breaks = np.concatenate(([0.0], recall[positive_indices]))
    ap = float(np.sum(precision[positive_indices] * np.diff(recall_breaks)))
    return {"mcc": mcc, "auprc": ap, "brier": brier}


def percentile_ci(values):
    values = np.asarray(values)
    return {"lower_ci95": float(np.percentile(values, 2.5)), "upper_ci95": float(np.percentile(values, 97.5)), "mean": float(np.mean(values))}


def load_oof_probabilities(path):
    rows = _read_csv(path)
    return {str(row["sample_id"]): float(row["probability"]) for row in rows}


def build_aligned_arrays(seed_dirs, baseline_dirs, proxy_prediction_csv, teacher_package, fold_manifest_path):
    anchor = _read_csv(seed_dirs["proxy_teacher_kd"][str(SEEDS[0])] / "oof_predictions.csv")
    anchor_ids = [str(row["sample_id"]) for row in anchor]
    labels = np.asarray([int(row["label"]) for row in anchor], dtype=np.float64)
    folds = np.asarray([int(row["fold"]) for row in anchor], dtype=np.int64)

    def column(rows_by_sample, key):
        values = []
        for sample_id in anchor_ids:
            item = rows_by_sample[str(sample_id)]
            values.append(float(item) if isinstance(item, (int, float)) else float(item[key]))
        return np.asarray(values, dtype=np.float64)

    arrays = {"labels": labels, "folds": folds}
    model_paths = {}
    for arm, seed_map in seed_dirs.items():
        for seed_text, directory in seed_map.items():
            arrays[f"{arm}_seed{seed_text}"] = column(load_oof_probabilities(directory / "oof_predictions.csv"), "probability")
            model_paths.setdefault(arm, {})[seed_text] = directory
    for seed_text, directory in baseline_dirs.items():
        arrays[f"mlp_baseline_seed{seed_text}"] = column(load_oof_probabilities(directory / "oof_predictions.csv"), "probability")
        model_paths.setdefault("mlp_baseline", {})[seed_text] = directory
    proxy_rows = _read_csv(proxy_prediction_csv)
    proxy_by_sample = {str(row["sample_id"]): row for row in proxy_rows}
    arrays["predicted_mean_plddt"] = np.asarray([float(proxy_by_sample[str(sample_id)]["mean_plddt"]) for sample_id in anchor_ids])
    arrays["predicted_low_plddt_fraction"] = np.asarray([float(proxy_by_sample[str(sample_id)]["low_plddt_fraction"]) for sample_id in anchor_ids])

    fold_rows = _read_csv(fold_manifest_path)
    historical_to_student = {str(row["historical_sample_id"]): str(row["sample_id"]) for row in fold_rows}
    teacher_rows = []
    with tarfile.open(teacher_package, "r:gz") as tar:
        member = "runs/topology_strict_oof_v1/topology_oof/oof_predictions.csv"
        with tar.extractfile(member) as handle:
            reader = csv.DictReader(handle.read().decode("utf-8").splitlines())
            for row in reader:
                if row["variant"] == "baseline":
                    teacher_rows.append(row)
    teacher_by_student = {}
    for row in teacher_rows:
        student_id = historical_to_student.get(str(row["sample_id"]))
        if student_id is None:
            raise ValueError(f"No student mapping for teacher sample {row['sample_id']}")
        teacher_by_student[student_id] = float(row["probability"])
    if len(teacher_by_student) != len(anchor_ids):
        raise ValueError("Teacher held-fold probabilities do not cover every student record")
    arrays["teacher_held_probability"] = np.asarray([teacher_by_student[str(sample_id)] for sample_id in anchor_ids])
    return arrays, model_paths


def strata_from_arrays(arrays):
    thresholds = {}
    strata = {}
    values = arrays["predicted_mean_plddt"]
    thresholds["predicted_mean_plddt"] = float(np.quantile(values, STRATUM_QUANTILES["predicted_mean_plddt"]))
    strata["low_predicted_mean_plddt"] = values <= thresholds["predicted_mean_plddt"]
    values = arrays["predicted_low_plddt_fraction"]
    thresholds["predicted_low_plddt_fraction"] = float(np.quantile(values, STRATUM_QUANTILES["predicted_low_plddt_fraction"]))
    strata["high_predicted_low_plddt_fraction"] = values >= thresholds["predicted_low_plddt_fraction"]
    disagreement = np.abs(arrays["teacher_held_probability"] - arrays["mlp_baseline_seed20260905"])
    thresholds["teacher_mlp_disagreement"] = float(np.quantile(disagreement, STRATUM_QUANTILES["teacher_mlp_disagreement"]))
    strata["teacher_mlp_disagreement"] = disagreement >= thresholds["teacher_mlp_disagreement"]
    return thresholds, strata


def bootstrap_deltas(labels, proxy_by_seed, mlp_by_seed, index_pool, n_resamples, rng):
    positive = index_pool[labels[index_pool] == 1]
    negative = index_pool[labels[index_pool] == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Bootstrap pool must contain both classes for label stratification")
    deltas = []
    for _ in range(n_resamples):
        sampled = np.concatenate(
            [
                positive[rng.integers(0, len(positive), size=len(positive))],
                negative[rng.integers(0, len(negative), size=len(negative))],
            ]
        )
        seed_deltas = []
        for seed in SEEDS:
            proxy_metrics = fast_metrics(labels[sampled], proxy_by_seed[seed][sampled])
            mlp_metrics = fast_metrics(labels[sampled], mlp_by_seed[seed][sampled])
            seed_deltas.append(
                {
                    "mcc": proxy_metrics["mcc"] - mlp_metrics["mcc"],
                    "auprc": proxy_metrics["auprc"] - mlp_metrics["auprc"],
                    "brier": proxy_metrics["brier"] - mlp_metrics["brier"],
                }
            )
        deltas.append(
            {
                "mcc": np.mean([item["mcc"] for item in seed_deltas]),
                "auprc": np.mean([item["auprc"] for item in seed_deltas]),
                "brier": np.mean([item["brier"] for item in seed_deltas]),
            }
        )
    return {metric: percentile_ci([item[metric] for item in deltas]) for metric in ("mcc", "auprc", "brier")}


def fold_stability(arrays):
    rows = []
    proxy_wins = {"mcc": 0, "auprc": 0, "brier": 0}
    total = 0
    for seed in SEEDS:
        for fold in range(5):
            mask = arrays["folds"] == fold
            proxy_metrics = fast_metrics(arrays["labels"][mask], arrays[f"proxy_teacher_kd_seed{seed}"][mask])
            mlp_metrics = fast_metrics(arrays["labels"][mask], arrays[f"mlp_teacher_kd_seed{seed}"][mask])
            delta = {metric: proxy_metrics[metric] - mlp_metrics[metric] for metric in ("mcc", "auprc", "brier")}
            for metric in ("mcc", "auprc", "brier"):
                proxy_wins[metric] += int(delta[metric] > 0 if metric != "brier" else delta[metric] < 0)
            rows.append({"seed": seed, "fold": fold, "delta": delta})
            total += 1
    return {"comparisons": rows, "proxy_wins": proxy_wins, "total_comparisons": total}


def decide(overall, subgroups):
    overall_mcc = overall["mcc"]
    overall_brier = overall["brier"]
    overall_auprc = overall["auprc"]
    stable_overall = (overall_mcc["lower_ci95"] > 0 or (overall_auprc["lower_ci95"] > 0 and overall_mcc["lower_ci95"] >= MCC_FLOOR)) and overall_brier["upper_ci95"] <= BRIER_TOLERANCE
    stable_subgroups = 0
    subgroup_evidence = {}
    for name, values in subgroups.items():
        mcc = values["mcc"]
        auprc = values["auprc"]
        brier = values["brier"]
        supported = (mcc["lower_ci95"] > 0 or auprc["lower_ci95"] > 0) and brier["upper_ci95"] <= BRIER_TOLERANCE
        subgroup_evidence[name] = supported
        stable_subgroups += int(supported)
    overall_supported = stable_overall and overall_mcc["lower_ci95"] > 0
    subgroup_supported = stable_subgroups >= 2 and overall_mcc["mean"] >= MCC_FLOOR and overall_brier["upper_ci95"] <= BRIER_TOLERANCE
    if overall_supported or subgroup_supported:
        decision = "proxy_teacher_kd"
        reason = "overall stable MCC advantage" if overall_supported else "predefined hard-subgroup advantage"
    else:
        decision = "mlp_teacher_kd"
        reason = "no reproducible proxy advantage over the simpler distilled MLP"
    return {"decision": decision, "reason": reason, "overall_supported": overall_supported, "subgroup_supported": subgroup_supported, "subgroup_evidence": subgroup_evidence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-kd", nargs=2, action="append", required=True, metavar=("SEED", "DIR"))
    parser.add_argument("--mlp-kd", nargs=2, action="append", required=True, metavar=("SEED", "DIR"))
    parser.add_argument("--mlp-baseline-dir", required=True, action="append")
    parser.add_argument("--proxy-prediction-csv", required=True)
    parser.add_argument("--teacher-package", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.proxy_kd) != len(SEEDS) or len(args.mlp_kd) != len(SEEDS) or len(args.mlp_baseline_dir) != len(SEEDS):
        raise ValueError("Exactly two seeds per arm are required")
    seed_dirs = {
        "proxy_teacher_kd": {str(seed): Path(directory) for seed, directory in args.proxy_kd},
        "mlp_teacher_kd": {str(seed): Path(directory) for seed, directory in args.mlp_kd},
    }
    baseline_dirs = {str(seed): Path(directory) for seed, directory in zip(SEEDS, args.mlp_baseline_dir, strict=True)}
    arrays, _model_paths = build_aligned_arrays(seed_dirs, baseline_dirs, args.proxy_prediction_csv, args.teacher_package, args.fold_manifest)
    thresholds, strata = strata_from_arrays(arrays)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    proxy_probs = np.concatenate([arrays[f"proxy_teacher_kd_seed{seed}"] for seed in SEEDS])
    mlp_probs = np.concatenate([arrays[f"mlp_teacher_kd_seed{seed}"] for seed in SEEDS])
    proxy_by_seed = {seed: arrays[f"proxy_teacher_kd_seed{seed}"] for seed in SEEDS}
    mlp_by_seed = {seed: arrays[f"mlp_teacher_kd_seed{seed}"] for seed in SEEDS}
    overall = bootstrap_deltas(arrays["labels"], proxy_by_seed, mlp_by_seed, np.arange(len(arrays["labels"])), N_BOOTSTRAP, rng)
    subgroups = {}
    for name, mask in strata.items():
        index_pool = np.flatnonzero(mask)
        subgroups[name] = bootstrap_deltas(arrays["labels"], proxy_by_seed, mlp_by_seed, index_pool, N_BOOTSTRAP, rng)
    folds = fold_stability(arrays)
    decision = decide(overall, subgroups)
    point_deltas = {
        metric: {
            "seed_20260905": fast_metrics(arrays["labels"], arrays["proxy_teacher_kd_seed20260905"])[metric] - fast_metrics(arrays["labels"], arrays["mlp_teacher_kd_seed20260905"])[metric],
            "seed_20260906": fast_metrics(arrays["labels"], arrays["proxy_teacher_kd_seed20260906"])[metric] - fast_metrics(arrays["labels"], arrays["mlp_teacher_kd_seed20260906"])[metric],
        }
        for metric in ("mcc", "auprc", "brier")
    }
    output = {
        "artifact_type": "prottox_3d_distill_head2head_analysis",
        "version": 1,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "stratum_quantiles": STRATUM_QUANTILES,
        "stratum_thresholds": thresholds,
        "stratum_sizes": {name: int(np.sum(mask)) for name, mask in strata.items()},
        "overall_point_deltas_proxy_minus_mlp": point_deltas,
        "overall_bootstrap": overall,
        "subgroup_bootstrap": subgroups,
        "fold_stability": folds,
        "decision_rule": {
            "brier_tolerance": BRIER_TOLERANCE,
            "mcc_floor_for_auprc_route": MCC_FLOOR,
            "minimum_supporting_subgroups": 2,
        },
        "decision": decision,
        "selection_scope": "training OOF only; Test1/Test2 excluded",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
