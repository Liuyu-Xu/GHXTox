"""Paper analysis utilities for the HyPepTox-Fuse balanced benchmark.

The independent HyPepTox-Fuse split used here is balanced, so the headline
tables intentionally omit balanced accuracy.  This script consumes saved metric
JSON files and prediction CSV files; it does not train or tune models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HEADLINE_METRICS = (
    "accuracy",
    "sn",
    "sp",
    "f1",
    "mcc",
    "auroc",
    "auprc",
    "brier",
    "ece_10",
)

HIGHER_IS_BETTER = {
    "accuracy": True,
    "sn": True,
    "sp": True,
    "f1": True,
    "mcc": True,
    "auroc": True,
    "auprc": True,
    "brier": False,
    "ece_10": False,
}

METRIC_LABELS = {
    "accuracy": "ACC",
    "sn": "SN",
    "sp": "SP",
    "f1": "F1",
    "mcc": "MCC",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "brier": "Brier",
    "ece_10": "ECE-10",
}

HYPEPTOX_FUSE_TARGET = {
    "accuracy": 0.9067,
    "sn": 0.8832,
    "sp": 0.9303,
    "f1": 0.9045,
    "mcc": 0.8143,
    "auroc": 0.9624,
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    metrics_path: str
    predictions_path: str | None = None
    role: str = "ablation"


DEFAULT_RUNS = (
    RunSpec(
        "3D-v2 baseline",
        "runs/hypeptox_fuse_3d_v2/test_metrics_full.json",
        "runs/hypeptox_fuse_3d_v2/test_predictions_full.csv",
    ),
    RunSpec(
        "ESM2-BiLSTM + AugMix",
        "runs/hypeptox_fuse_esm2_bilstm_augmix_noleak/test_metrics.json",
        "runs/hypeptox_fuse_esm2_bilstm_augmix_noleak/test_predictions.csv",
    ),
    RunSpec(
        "ESM2-Geometry + AugMix",
        "runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak/test_metrics.json",
        "runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak/test_predictions.csv",
    ),
    RunSpec(
        "3D-v2 + AugMix",
        "runs/hypeptox_fuse_3d_v2_augmix_noleak/test_metrics.json",
        "runs/hypeptox_fuse_3d_v2_augmix_noleak/test_predictions.csv",
    ),
    RunSpec(
        "ESM2-BiLSTM 3-seed ensemble",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_bilstm_3seed/test_metrics.json",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_bilstm_3seed/test_predictions.csv",
    ),
    RunSpec(
        "ESM2-Geometry 3-seed ensemble",
        "runs/hypeptox_fuse_ensemble_augmix_geometry_3seed/test_metrics.json",
        None,
    ),
    RunSpec(
        "ESM2 6-model ensemble",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_6model/test_metrics.json",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_6model/test_predictions.csv",
    ),
    RunSpec(
        "ESM2+3D 7-model ensemble",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_3dv2_7model/test_metrics.json",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_3dv2_7model/test_predictions.csv",
    ),
    RunSpec(
        "ESM2+3D 7-model ensemble + validation threshold",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_3dv2_7model/test_metrics_thr0386.json",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_3dv2_7model/test_predictions_thr0386.csv",
        role="final",
    ),
)

CALIBRATION_RUNS = (
    RunSpec(
        "7-model + temperature scaling",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_3dv2_7model/calibration_temperature_summary.json",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_3dv2_7model/calibrated_temperature/test_predictions_thr0386_calibrated.csv",
        role="calibration",
    ),
    RunSpec(
        "7-model + Platt scaling",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_3dv2_7model/calibration_platt_summary.json",
        "runs/hypeptox_fuse_ensemble_augmix_esm2_geometry_3dv2_7model/calibrated_platt/test_predictions_thr0386_calibrated.csv",
        role="calibration",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    threshold = payload.get("threshold", float("nan"))
    if "applied" in payload and payload["applied"]:
        payload = payload["applied"][0]["after"]
    result: dict[str, float] = {}
    for key in HEADLINE_METRICS:
        aliases = {
            "accuracy": ("accuracy", "acc"),
            "sn": ("sn", "sensitivity", "recall", "sen"),
            "sp": ("sp", "specificity", "spe"),
            "auroc": ("auroc", "auc"),
        }.get(key, (key,))
        value = next((payload[name] for name in aliases if name in payload), None)
        result[key] = float(value) if value is not None else float("nan")
    result["threshold"] = float(payload.get("threshold", threshold))
    if "num_samples" in payload:
        result["num_samples"] = float(payload["num_samples"])
    return result


def _format_float(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_run_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(DEFAULT_RUNS + CALIBRATION_RUNS):
        metrics_path = root / spec.metrics_path
        if not metrics_path.exists():
            continue
        metrics = _extract_metrics(_read_json(metrics_path))
        row: dict[str, Any] = {
            "order": index,
            "model": spec.name,
            "role": spec.role,
            "metrics_path": spec.metrics_path,
            "predictions_path": spec.predictions_path or "",
            "threshold": metrics.get("threshold", float("nan")),
        }
        for metric in HEADLINE_METRICS:
            row[metric] = metrics[metric]
        rows.append(row)
    return rows


def _headline_rows(final_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric in HEADLINE_METRICS:
        ours = float(final_row[metric])
        target = HYPEPTOX_FUSE_TARGET.get(metric)
        if target is None:
            delta = None
            direction = "not_reported_by_hypeptox_fuse"
        else:
            raw_delta = ours - target
            delta = raw_delta if HIGHER_IS_BETTER[metric] else -raw_delta
            if delta > 0:
                direction = "ours_better"
            elif delta < 0:
                direction = "hypeptox_fuse_better"
            else:
                direction = "tie"
        rows.append(
            {
                "metric": METRIC_LABELS[metric],
                "metric_key": metric,
                "ours": _format_float(ours),
                "hypeptox_fuse": _format_float(target),
                "advantage_positive_is_better": _format_float(delta),
                "direction": direction,
                "note": "" if target is not None else "HyPepTox-Fuse paper table did not report this metric.",
            }
        )
    return rows


def _metric_values(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(np.int64, copy=False)
    probs = probs.astype(np.float64, copy=False)
    preds = probs >= threshold
    positives = labels == 1
    negatives = labels == 0
    tp = float(np.sum(preds & positives))
    tn = float(np.sum(~preds & negatives))
    fp = float(np.sum(preds & negatives))
    fn = float(np.sum(~preds & positives))
    total = max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
    values = {
        "accuracy": (tp + tn) / total,
        "sn": recall,
        "sp": specificity,
        "f1": f1,
        "mcc": ((tp * tn) - (fp * fn)) / denominator,
        "brier": float(np.mean((probs - labels) ** 2)),
        "ece_10": _ece(labels, probs),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if np.unique(labels).size == 2:
            values["auroc"] = float(roc_auc_score(labels, probs))
            values["auprc"] = float(average_precision_score(labels, probs))
        else:
            values["auroc"] = float("nan")
            values["auprc"] = float("nan")
    except Exception:
        values["auroc"] = float("nan")
        values["auprc"] = float("nan")
    return values


def _ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == n_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        if not np.any(mask):
            continue
        bin_predictions = (probs[mask] >= 0.5).astype(np.int64)
        accuracy = float(np.mean(bin_predictions == labels[mask]))
        confidence = float(np.mean(np.maximum(probs[mask], 1.0 - probs[mask])))
        ece += float(mask.mean()) * abs(accuracy - confidence)
    return ece


def _read_prediction_csv(path: Path, threshold_override: float | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    threshold = threshold_override
    if threshold is None and "decision_threshold" in rows[0]:
        threshold = float(rows[0]["decision_threshold"])
    if threshold is None:
        threshold = 0.5
    return {
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "labels": np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
        "probs": np.asarray([float(row["toxicity_probability"]) for row in rows], dtype=np.float64),
        "threshold": float(threshold),
    }


def _align_predictions(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_lookup = {
        sample_id: (label, prob)
        for sample_id, label, prob in zip(
            candidate["sample_ids"], candidate["labels"], candidate["probs"]
        )
    }
    labels = []
    ref_probs = []
    cand_probs = []
    missing = []
    for sample_id, label, prob in zip(reference["sample_ids"], reference["labels"], reference["probs"]):
        value = candidate_lookup.get(sample_id)
        if value is None:
            missing.append(sample_id)
            continue
        cand_label, cand_prob = value
        if int(cand_label) != int(label):
            raise ValueError(f"Label mismatch for sample {sample_id}")
        labels.append(int(label))
        ref_probs.append(float(prob))
        cand_probs.append(float(cand_prob))
    if missing:
        raise ValueError(f"Missing {len(missing)} reference samples in candidate predictions")
    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "reference_probs": np.asarray(ref_probs, dtype=np.float64),
        "candidate_probs": np.asarray(cand_probs, dtype=np.float64),
    }


def _stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(group.size == 0 for group in groups):
        raise ValueError("Both classes are required for stratified bootstrap.")
    return np.concatenate([rng.choice(group, size=group.size, replace=True) for group in groups])


def _improvement(reference: float, candidate: float, metric: str) -> float:
    return reference - candidate if HIGHER_IS_BETTER[metric] else candidate - reference


def _paired_bootstrap_rows(
    root: Path,
    run_rows: list[dict[str, Any]],
    reference_name: str,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    reference_row = next(row for row in run_rows if row["model"] == reference_name)
    reference = _read_prediction_csv(root / str(reference_row["predictions_path"]))
    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(run_rows):
        if row["model"] == reference_name or not row.get("predictions_path"):
            continue
        path = root / str(row["predictions_path"])
        if not path.exists():
            continue
        candidate = _read_prediction_csv(path)
        aligned = _align_predictions(reference, candidate)
        labels = aligned["labels"]
        ref_point = _metric_values(labels, aligned["reference_probs"], reference["threshold"])
        cand_point = _metric_values(labels, aligned["candidate_probs"], candidate["threshold"])
        rng = np.random.default_rng(seed + row_index)
        samples = {metric: np.empty(iterations, dtype=np.float64) for metric in HEADLINE_METRICS}
        for iteration in range(iterations):
            indices = _stratified_indices(labels, rng)
            ref_values = _metric_values(
                labels[indices],
                aligned["reference_probs"][indices],
                reference["threshold"],
            )
            cand_values = _metric_values(
                labels[indices],
                aligned["candidate_probs"][indices],
                candidate["threshold"],
            )
            for metric in HEADLINE_METRICS:
                samples[metric][iteration] = _improvement(ref_values[metric], cand_values[metric], metric)
        for metric in HEADLINE_METRICS:
            values = samples[metric]
            point = _improvement(ref_point[metric], cand_point[metric], metric)
            output_rows.append(
                {
                    "reference_model": reference_name,
                    "candidate_model": row["model"],
                    "metric": METRIC_LABELS[metric],
                    "metric_key": metric,
                    "reference_value": _format_float(ref_point[metric]),
                    "candidate_value": _format_float(cand_point[metric]),
                    "improvement_positive_is_better": _format_float(point),
                    "ci95_lower": _format_float(float(np.quantile(values, 0.025))),
                    "ci95_upper": _format_float(float(np.quantile(values, 0.975))),
                    "bootstrap_probability_reference_better": _format_float(float(np.mean(values > 0.0))),
                    "iterations": iterations,
                }
            )
    return output_rows


def _plot_outputs(
    root: Path,
    output_dir: Path,
    run_rows: list[dict[str, Any]],
    headline_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve, roc_curve
    except Exception as exc:
        (output_dir / "plot_skipped.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return

    colors = {
        "blue": "#176B87",
        "red": "#C44E52",
        "green": "#55A868",
        "gold": "#CCB974",
        "purple": "#8172B2",
        "gray": "#667085",
        "ink": "#17202A",
        "grid": "#D8DEE5",
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    reported = [row for row in headline_rows if row["hypeptox_fuse"]]
    x = np.arange(len(reported))
    deltas = [float(row["advantage_positive_is_better"]) for row in reported]
    fig, ax = plt.subplots(figsize=(7.6, 3.6), constrained_layout=True)
    ax.bar(
        x,
        deltas,
        color=[colors["green"] if value >= 0 else colors["red"] for value in deltas],
    )
    ax.axhline(0.0, color=colors["ink"], linewidth=0.8)
    ax.set_xticks(x, [row["metric"] for row in reported])
    ax.set_ylabel("Ours minus HyPepTox-Fuse")
    ax.set_title("Headline metric deltas without BACC")
    ax.grid(axis="y", color=colors["grid"], alpha=0.7)
    fig.savefig(output_dir / "headline_delta_vs_hypeptox_fuse.png", dpi=240)
    plt.close(fig)

    metric_keys = ("accuracy", "sn", "sp", "f1", "mcc", "auroc", "auprc")
    ablation_rows = [row for row in run_rows if row["role"] != "calibration"]
    fig, ax = plt.subplots(figsize=(10.4, 4.8), constrained_layout=True)
    positions = np.arange(len(ablation_rows))
    offsets = np.linspace(-0.33, 0.33, len(metric_keys))
    for offset, metric in zip(offsets, metric_keys):
        ax.plot(
            positions + offset,
            [float(row[metric]) for row in ablation_rows],
            marker="o",
            linewidth=1.5,
            label=METRIC_LABELS[metric],
        )
    ax.set_xticks(positions, [row["model"] for row in ablation_rows], rotation=30, ha="right")
    ax.set_ylim(0.74, 0.975)
    ax.set_ylabel("Metric value")
    ax.set_title("Ablation and ensemble progression")
    ax.grid(axis="y", color=colors["grid"], alpha=0.7)
    ax.legend(ncol=4, frameon=False)
    fig.savefig(output_dir / "ablation_progression.png", dpi=240)
    plt.close(fig)

    calibration_rows = [row for row in run_rows if "7-model" in row["model"] and row["role"] in {"final", "calibration"}]
    if calibration_rows:
        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        width = 0.34
        x = np.arange(len(calibration_rows))
        ax.bar(x - width / 2, [float(row["brier"]) for row in calibration_rows], width=width, label="Brier", color=colors["blue"])
        ax.bar(x + width / 2, [float(row["ece_10"]) for row in calibration_rows], width=width, label="ECE-10", color=colors["gold"])
        ax.set_xticks(x, [row["model"].replace("ESM2+3D ", "") for row in calibration_rows], rotation=25, ha="right")
        ax.set_ylabel("Lower is better")
        ax.set_title("Calibration metrics")
        ax.grid(axis="y", color=colors["grid"], alpha=0.7)
        ax.legend(frameon=False)
        fig.savefig(output_dir / "calibration_brier_ece.png", dpi=240)
        plt.close(fig)

    final_row = next((row for row in run_rows if row["role"] == "final"), None)
    if final_row and final_row.get("predictions_path"):
        predictions = _read_prediction_csv(root / str(final_row["predictions_path"]))
        labels = predictions["labels"]
        probs = predictions["probs"]
        fpr, tpr, _ = roc_curve(labels, probs)
        precision, recall, _ = precision_recall_curve(labels, probs)
        frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10, strategy="uniform")
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)
        axes[0].plot(fpr, tpr, color=colors["blue"], linewidth=2.0)
        axes[0].plot([0, 1], [0, 1], color=colors["gray"], linestyle="--", linewidth=1.0)
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        axes[1].plot(recall, precision, color=colors["red"], linewidth=2.0)
        axes[1].set(xlabel="Recall", ylabel="Precision", title="PR")
        axes[2].plot(mean_pred, frac_pos, marker="o", color=colors["purple"], linewidth=2.0)
        axes[2].plot([0, 1], [0, 1], color=colors["gray"], linestyle="--", linewidth=1.0)
        axes[2].set(xlabel="Mean predicted probability", ylabel="Fraction positive", title="Calibration")
        for axis in axes:
            axis.grid(color=colors["grid"], alpha=0.7)
        fig.savefig(output_dir / "final_roc_pr_calibration.png", dpi=240)
        plt.close(fig)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = _load_run_rows(root)
    if not run_rows:
        raise FileNotFoundError("No saved HyPepTox-Fuse run metrics were found.")
    final_row = next((row for row in run_rows if row["role"] == "final"), run_rows[-1])
    headline_rows = _headline_rows(final_row)

    metric_fields = ["order", "model", "role", "threshold", *HEADLINE_METRICS, "metrics_path", "predictions_path"]
    _write_csv(output_dir / "ablation_table_no_bacc.csv", run_rows, metric_fields)
    _write_csv(
        output_dir / "headline_comparison_no_bacc.csv",
        headline_rows,
        ["metric", "metric_key", "ours", "hypeptox_fuse", "advantage_positive_is_better", "direction", "note"],
    )
    paired_rows = _paired_bootstrap_rows(
        root,
        run_rows,
        str(final_row["model"]),
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    _write_csv(
        output_dir / "paired_bootstrap_vs_final.csv",
        paired_rows,
        [
            "reference_model",
            "candidate_model",
            "metric",
            "metric_key",
            "reference_value",
            "candidate_value",
            "improvement_positive_is_better",
            "ci95_lower",
            "ci95_upper",
            "bootstrap_probability_reference_better",
            "iterations",
        ],
    )
    summary = {
        "protocol": {
            "benchmark": "HyPepTox-Fuse balanced independent set",
            "headline_metrics": [METRIC_LABELS[name] for name in HEADLINE_METRICS],
            "balanced_accuracy_policy": "omitted from headline tables because the independent set is balanced",
            "hyptox_fuse_limitation": "Only paper-level point metrics are available; paired statistical tests require per-sample predictions.",
            "bootstrap_iterations": int(args.bootstrap_iterations),
            "seed": int(args.seed),
        },
        "final_model": final_row,
        "hyptox_fuse_target": HYPEPTOX_FUSE_TARGET,
        "outputs": {
            "headline_comparison": "headline_comparison_no_bacc.csv",
            "ablation_table": "ablation_table_no_bacc.csv",
            "paired_bootstrap": "paired_bootstrap_vs_final.csv",
            "figures": [
                "headline_delta_vs_hypeptox_fuse.png",
                "ablation_progression.png",
                "calibration_brier_ece.png",
                "final_roc_pr_calibration.png",
            ],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot_outputs(root, output_dir, run_rows, headline_rows)
    print(f"HyPepTox-Fuse paper analysis saved to {output_dir}")
    print(
        "Final model: "
        f"ACC={float(final_row['accuracy']):.4f} "
        f"SN={float(final_row['sn']):.4f} "
        f"SP={float(final_row['sp']):.4f} "
        f"F1={float(final_row['f1']):.4f} "
        f"MCC={float(final_row['mcc']):.4f} "
        f"AUROC={float(final_row['auroc']):.4f} "
        f"AUPRC={float(final_row['auprc']):.4f}"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate no-BACC HyPepTox-Fuse comparison, ablation tables, and analysis figures."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="reports/hypeptox_fuse_paper_analysis")
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    run_analysis(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
