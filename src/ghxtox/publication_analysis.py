"""Generate publication-oriented benchmark, calibration, and ablation figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from ghxtox.plm_textcnn import _probability_metrics
from ghxtox.utils import save_json


CORE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "mcc",
    "auroc",
    "auprc",
    "brier",
    "ece_10",
)
DISPLAY_NAMES = {
    "accuracy": "ACC",
    "balanced_accuracy": "BACC",
    "sensitivity": "SN",
    "specificity": "SP",
    "mcc": "MCC",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "brier": "Brier",
    "ece_10": "ECE-10",
}
COLORS = {
    "ProtT5-3Seed": "#087F8C",
    "Archived 1D+3D": "#C44536",
    "ToxPLTC": "#D89B23",
    "ToxMSRC": "#59636E",
    "grid": "#D6DADF",
    "ink": "#20262E",
    "muted": "#68727D",
    "paper": "#FFFFFF",
}


PAPER_BENCHMARKS: dict[str, dict[str, dict[str, float]]] = {
    "Test1": {
        "ToxMSRC": {
            "accuracy": 0.9405,
            "balanced_accuracy": 0.9217,
            "sensitivity": 0.8781,
            "specificity": 0.9653,
            "mcc": 0.8520,
            "auroc": 0.9650,
            "auprc": math.nan,
            "brier": math.nan,
            "ece_10": math.nan,
        },
        "ToxPLTC": {
            "accuracy": 0.9378,
            "balanced_accuracy": 0.9302,
            "sensitivity": 0.9125,
            "specificity": 0.9479,
            "mcc": 0.8496,
            "auroc": 0.9709,
            "auprc": 0.9426,
            "brier": math.nan,
            "ece_10": math.nan,
        },
    },
    "Test2": {
        "ToxMSRC": {
            "accuracy": 0.9416,
            "balanced_accuracy": 0.8689,
            "sensitivity": 0.7826,
            "specificity": 0.9552,
            "mcc": 0.6550,
            "auroc": 0.9430,
            "auprc": math.nan,
            "brier": math.nan,
            "ece_10": math.nan,
        },
        "ToxPLTC": {
            "accuracy": 0.9261,
            "balanced_accuracy": 0.8804,
            "sensitivity": 0.8261,
            "specificity": 0.9347,
            "mcc": 0.6197,
            "auroc": 0.9295,
            "auprc": 0.7353,
            "brier": math.nan,
            "ece_10": math.nan,
        },
    },
}


@dataclass(frozen=True)
class PredictionData:
    labels: np.ndarray
    probabilities: np.ndarray
    sequences: list[str]
    threshold: float
    predictions: np.ndarray | None = None


@dataclass(frozen=True)
class AblationSpec:
    name: str
    path: str
    threshold: float | None
    probability_column: str = "toxicity_probability"
    prediction_column: str | None = None


ABLATIONS = (
    AblationSpec(
        "ProtT5, no BSMOTE",
        "runs/plm_textcnn_prott5_none/oof_predictions.csv",
        0.377916157245636,
    ),
    AblationSpec(
        "ProtT5 + BSMOTE, 1 seed",
        "runs/plm_textcnn_prott5_bsmote/oof_predictions.csv",
        0.5012562274932861,
    ),
    AblationSpec(
        "ESM2 + BSMOTE, 3 seeds",
        "runs/plm_textcnn_esm2_bsmote_multiseed/oof_predictions.csv",
        0.60,
    ),
    AblationSpec(
        "ProtT5 + BSMOTE, 3 seeds",
        "runs/plm_textcnn_prott5_bsmote_multiseed/oof_predictions.csv",
        0.45,
    ),
    AblationSpec(
        "BACC-priority ESM2 + ProtT5",
        "runs/plm_textcnn_esm2_prott5_fusion_bacc_oof/crossfit_predictions.csv",
        None,
        probability_column="fusion_probability",
        prediction_column="prediction",
    ),
)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def _load_prediction(path: str | Path, threshold: float) -> PredictionData:
    rows = _read_rows(path)
    required = {"label", "sequence", "toxicity_probability"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Prediction file {path} is missing columns: {sorted(missing)}")
    return PredictionData(
        labels=np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
        probabilities=np.asarray(
            [float(row["toxicity_probability"]) for row in rows], dtype=np.float64
        ),
        sequences=[row["sequence"] for row in rows],
        threshold=float(threshold),
        predictions=(
            np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
            if "prediction" in rows[0]
            else None
        ),
    )


def _metrics_with_decisions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    base = _probability_metrics(labels, probabilities, threshold=threshold)
    positives = labels == 1
    negatives = ~positives
    decisions = np.asarray(predictions, dtype=bool)
    tp = float(np.sum(decisions & positives))
    tn = float(np.sum(~decisions & negatives))
    fp = float(np.sum(decisions & negatives))
    fn = float(np.sum(~decisions & positives))
    sensitivity = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
    base.update(
        {
            "accuracy": (tp + tn) / max(len(labels), 1),
            "balanced_accuracy": 0.5 * (sensitivity + specificity),
            "sensitivity": sensitivity,
            "sn": sensitivity,
            "specificity": specificity,
            "sp": specificity,
            "mcc": (tp * tn - fp * fn) / denominator,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
    )
    return base


def _core_metrics(data: PredictionData) -> dict[str, float]:
    if data.predictions is not None:
        return _metrics_with_decisions(
            data.labels,
            data.probabilities,
            data.predictions,
            threshold=data.threshold,
        )
    return _probability_metrics(data.labels, data.probabilities, threshold=data.threshold)


def _validate_alignment(first: PredictionData, second: PredictionData) -> None:
    if first.sequences != second.sequences:
        raise ValueError("Model prediction sequences are not rowwise aligned.")
    if not np.array_equal(first.labels, second.labels):
        raise ValueError("Model prediction labels are not rowwise aligned.")


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(COLORS["paper"])
    axis.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(COLORS["muted"])
    axis.spines["bottom"].set_color(COLORS["muted"])
    axis.tick_params(colors=COLORS["ink"], labelsize=9)


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _write_core_table(
    output_dir: Path,
    results: dict[str, dict[str, dict[str, float]]],
) -> None:
    with (output_dir / "core_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "model", *CORE_METRICS])
        writer.writeheader()
        for dataset, models in results.items():
            for model, values in models.items():
                writer.writerow(
                    {
                        "dataset": dataset,
                        "model": model,
                        **{
                            metric: "NR" if not np.isfinite(values[metric]) else f"{values[metric]:.9g}"
                            for metric in CORE_METRICS
                        },
                    }
                )


def _plot_benchmark(
    output_dir: Path,
    results: dict[str, dict[str, dict[str, float]]],
) -> None:
    metrics = CORE_METRICS[:7]
    x = np.arange(len(metrics))
    figure, axes = plt.subplots(2, 1, figsize=(11.8, 7.8), constrained_layout=True)
    for axis, dataset in zip(axes, ("Test1", "Test2"), strict=True):
        _style_axis(axis)
        for model in ("ToxMSRC", "ToxPLTC", "ProtT5-3Seed", "Archived 1D+3D"):
            values = np.asarray([results[dataset][model][metric] for metric in metrics])
            axis.plot(
                x,
                values,
                marker="o",
                markersize=5.5,
                linewidth=2.2 if model == "ProtT5-3Seed" else 1.5,
                color=COLORS[model],
                label=model,
                alpha=1.0 if model == "ProtT5-3Seed" else 0.88,
            )
        axis.set_xticks(x, [DISPLAY_NAMES[metric] for metric in metrics])
        axis.set_ylim((0.82, 0.995) if dataset == "Test1" else (0.60, 0.99))
        axis.set_ylabel("Score")
        axis.set_title(f"{dataset} | same split benchmark", loc="left", fontsize=12, fontweight="bold")
        bacc = results[dataset]["ProtT5-3Seed"]["balanced_accuracy"]
        axis.annotate(
            f"Default BACC {bacc:.4f}",
            xy=(1, bacc),
            xytext=(1.2, min(bacc + 0.035, axis.get_ylim()[1] - 0.01)),
            arrowprops={"arrowstyle": "-", "color": COLORS["ProtT5-3Seed"]},
            color=COLORS["ProtT5-3Seed"],
            fontsize=9,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.025))
    figure.suptitle("Core discrimination metrics", fontsize=15, fontweight="bold", x=0.01, ha="left")
    _save_figure(figure, output_dir, "figure1_core_benchmark")


def _calibration_bins(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float]]:
    bins: list[dict[str, float]] = []
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == 9 else probabilities < upper
        )
        if not np.any(selected):
            continue
        bins.append(
            {
                "bin": float(index),
                "lower": lower,
                "upper": upper,
                "mean_probability": float(np.mean(probabilities[selected])),
                "observed_frequency": float(np.mean(labels[selected])),
                "count": float(np.sum(selected)),
            }
        )
    return bins


def _plot_calibration(
    output_dir: Path,
    model_data: dict[str, dict[str, PredictionData]],
    results: dict[str, dict[str, dict[str, float]]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    with (output_dir / "calibration_bins.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "model",
            "bin",
            "lower",
            "upper",
            "mean_probability",
            "observed_frequency",
            "count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for axis, dataset in zip(axes, ("Test1", "Test2"), strict=True):
            _style_axis(axis)
            axis.plot([0, 1], [0, 1], linestyle="--", color=COLORS["muted"], linewidth=1.0)
            for model in ("ProtT5-3Seed", "Archived 1D+3D"):
                bins = _calibration_bins(
                    model_data[dataset][model].labels,
                    model_data[dataset][model].probabilities,
                )
                writer.writerows(
                    {"dataset": dataset, "model": model, **row} for row in bins
                )
                axis.plot(
                    [row["mean_probability"] for row in bins],
                    [row["observed_frequency"] for row in bins],
                    marker="o",
                    linewidth=2.0,
                    color=COLORS[model],
                    label=(
                        f"{model} | Brier {results[dataset][model]['brier']:.3f}, "
                        f"ECE {results[dataset][model]['ece_10']:.3f}"
                    ),
                )
            axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed toxicity rate")
            axis.set_title(f"{dataset} reliability", loc="left", fontsize=12, fontweight="bold")
            axis.legend(loc="upper left", fontsize=8, frameon=False)
    figure.suptitle("Probability calibration", fontsize=15, fontweight="bold", x=0.01, ha="left")
    _save_figure(figure, output_dir, "figure2_calibration")


def _stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    return np.concatenate([rng.choice(group, size=group.size, replace=True) for group in groups])


def _mcnemar_exact(first_correct: np.ndarray, second_correct: np.ndarray) -> dict[str, float]:
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index) * 0.5**discordant
            for index in range(min(first_only, second_only) + 1)
        )
        p_value = min(1.0, 2.0 * tail)
    return {
        "prott5_only_correct": float(first_only),
        "joint_only_correct": float(second_only),
        "discordant": float(discordant),
        "exact_two_sided_p": float(p_value),
    }


def _paired_bootstrap(
    first: PredictionData,
    second: PredictionData,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    _validate_alignment(first, second)
    first_point = _core_metrics(first)
    second_point = _core_metrics(second)
    higher = CORE_METRICS[:7]
    lower = CORE_METRICS[7:]
    samples = {metric: np.empty(iterations, dtype=np.float64) for metric in CORE_METRICS}
    rng = np.random.default_rng(seed)
    for iteration in range(iterations):
        indices = _stratified_indices(first.labels, rng)
        first_values = _probability_metrics(
            first.labels[indices], first.probabilities[indices], threshold=first.threshold
        )
        second_values = _probability_metrics(
            second.labels[indices], second.probabilities[indices], threshold=second.threshold
        )
        for metric in higher:
            samples[metric][iteration] = first_values[metric] - second_values[metric]
        for metric in lower:
            samples[metric][iteration] = second_values[metric] - first_values[metric]
    intervals = {
        metric: {
            "advantage": (
                first_point[metric] - second_point[metric]
                if metric in higher
                else second_point[metric] - first_point[metric]
            ),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "probability_prott5_better": float(np.mean(values > 0.0)),
        }
        for metric, values in samples.items()
    }
    first_correct = (first.probabilities >= first.threshold) == first.labels
    second_correct = (second.probabilities >= second.threshold) == second.labels
    return {
        "protocol": {
            "iterations": int(iterations),
            "seed": int(seed),
            "sampling": "paired class-stratified percentile bootstrap",
            "advantage_semantics": "positive always favors ProtT5; Brier/ECE signs reversed",
        },
        "metrics": intervals,
        "mcnemar": _mcnemar_exact(first_correct, second_correct),
    }


def _plot_bootstrap(output_dir: Path, comparisons: dict[str, dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), constrained_layout=True)
    with (output_dir / "paired_bootstrap.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "metric", "advantage", "lower", "upper", "probability_prott5_better"],
        )
        writer.writeheader()
        for axis, dataset in zip(axes, ("Test1", "Test2"), strict=True):
            _style_axis(axis)
            y = np.arange(len(CORE_METRICS))
            values = comparisons[dataset]["metrics"]
            centers = np.asarray([values[metric]["advantage"] for metric in CORE_METRICS])
            lowers = np.asarray([values[metric]["lower"] for metric in CORE_METRICS])
            uppers = np.asarray([values[metric]["upper"] for metric in CORE_METRICS])
            colors = ["#087F8C" if lower > 0 else "#C44536" if upper < 0 else "#7A858F" for lower, upper in zip(lowers, uppers)]
            for row, (center, lower, upper, color) in enumerate(
                zip(centers, lowers, uppers, colors, strict=True)
            ):
                axis.errorbar(
                    center,
                    row,
                    xerr=np.asarray([[center - lower], [upper - center]]),
                    fmt="none",
                    ecolor=color,
                    elinewidth=2.0,
                    capsize=3,
                )
            axis.scatter(centers, y, c=colors, s=35, zorder=3)
            axis.axvline(0.0, color=COLORS["ink"], linewidth=1.0)
            axis.set_yticks(y, [DISPLAY_NAMES[metric] for metric in CORE_METRICS])
            axis.invert_yaxis()
            axis.set_xlabel("ProtT5 advantage (95% paired bootstrap CI)")
            axis.set_title(f"{dataset}", loc="left", fontsize=12, fontweight="bold")
            for metric in CORE_METRICS:
                writer.writerow({"dataset": dataset, "metric": metric, **values[metric]})
    figure.suptitle("Paired uncertainty: ProtT5-3Seed vs archived 1D+3D", fontsize=14, fontweight="bold", x=0.01, ha="left")
    _save_figure(figure, output_dir, "figure3_paired_bootstrap")


def _load_ablation(spec: AblationSpec) -> dict[str, float]:
    rows = _read_rows(spec.path)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [float(row[spec.probability_column]) for row in rows], dtype=np.float64
    )
    if spec.prediction_column is None:
        if spec.threshold is None:
            raise ValueError(f"A threshold is required for ablation {spec.name}.")
        return _probability_metrics(labels, probabilities, threshold=spec.threshold)
    predictions = np.asarray([int(row[spec.prediction_column]) for row in rows], dtype=np.int64)
    return _metrics_with_decisions(labels, probabilities, predictions)


def _plot_ablation(output_dir: Path, ablations: dict[str, dict[str, float]]) -> None:
    metrics = CORE_METRICS[1:]
    names = list(ablations)
    raw = np.asarray([[ablations[name][metric] for metric in metrics] for name in names])
    desirability = raw.copy()
    desirability[:, -2:] *= -1.0
    scaled = np.zeros_like(desirability)
    for column in range(desirability.shape[1]):
        low = float(np.min(desirability[:, column]))
        high = float(np.max(desirability[:, column]))
        scaled[:, column] = 0.5 if high == low else (desirability[:, column] - low) / (high - low)
    figure, axis = plt.subplots(figsize=(12.2, 5.1), constrained_layout=True)
    image = axis.imshow(scaled, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(metrics)), [DISPLAY_NAMES[metric] for metric in metrics])
    axis.set_yticks(np.arange(len(names)), names)
    axis.tick_params(axis="x", rotation=0)
    axis.set_title("OOF ablation matrix | color ranks each metric independently", loc="left", fontsize=13, fontweight="bold")
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            axis.text(
                column,
                row,
                f"{raw[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if scaled[row, column] < 0.17 or scaled[row, column] > 0.83 else COLORS["ink"],
            )
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Within-column desirability")
    _save_figure(figure, output_dir, "figure4_oof_ablation")
    with (output_dir / "oof_ablation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *CORE_METRICS])
        writer.writeheader()
        for name, values in ablations.items():
            writer.writerow({"model": name, **{metric: values[metric] for metric in CORE_METRICS}})


def _plot_roc_pr_curves(
    output_dir: Path,
    model_data: dict[str, dict[str, PredictionData]],
    results: dict[str, dict[str, dict[str, float]]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), constrained_layout=True)
    for row, dataset in enumerate(("Test1", "Test2")):
        roc_axis, pr_axis = axes[row]
        _style_axis(roc_axis)
        _style_axis(pr_axis)
        labels = model_data[dataset]["ProtT5-3Seed"].labels
        prevalence = float(np.mean(labels))
        roc_axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            color=COLORS["muted"],
            linewidth=1.0,
            label="Random",
        )
        pr_axis.axhline(
            prevalence,
            linestyle="--",
            color=COLORS["muted"],
            linewidth=1.0,
            label=f"Prevalence {prevalence:.3f}",
        )
        for model in ("ProtT5-3Seed", "Archived 1D+3D"):
            data = model_data[dataset][model]
            false_positive_rate, true_positive_rate, _ = roc_curve(
                data.labels, data.probabilities
            )
            precision, recall, _ = precision_recall_curve(
                data.labels, data.probabilities
            )
            linewidth = 2.3 if model == "ProtT5-3Seed" else 1.8
            roc_axis.plot(
                false_positive_rate,
                true_positive_rate,
                color=COLORS[model],
                linewidth=linewidth,
                label=f"{model} | AUROC {results[dataset][model]['auroc']:.3f}",
            )
            pr_axis.plot(
                recall,
                precision,
                color=COLORS[model],
                linewidth=linewidth,
                label=f"{model} | AUPRC {results[dataset][model]['auprc']:.3f}",
            )
        roc_axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="False positive rate",
            ylabel="True positive rate",
        )
        pr_axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.02),
            xlabel="Recall",
            ylabel="Precision",
        )
        roc_axis.set_title(f"{dataset} ROC", loc="left", fontsize=12, fontweight="bold")
        pr_axis.set_title(f"{dataset} precision-recall", loc="left", fontsize=12, fontweight="bold")
        roc_axis.legend(loc="lower right", fontsize=8, frameon=False)
        pr_axis.legend(loc="lower left", fontsize=8, frameon=False)
    figure.suptitle(
        "Frozen-test ranking performance",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    _save_figure(figure, output_dir, "figure5_roc_pr_curves")


def generate_publication_analysis(
    output_dir: str | Path,
    *,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 20260802,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "Test1": {
            "ProtT5-3Seed": (
                "runs/plm_textcnn_prott5_bsmote_multiseed/test1_predictions.csv",
                0.45,
            ),
            "Archived 1D+3D": (
                "runs/local_joint_validation/test1/predictions.csv",
                0.60,
            ),
        },
        "Test2": {
            "ProtT5-3Seed": (
                "runs/plm_textcnn_prott5_bsmote_multiseed/test2_predictions.csv",
                0.45,
            ),
            "Archived 1D+3D": (
                "runs/local_joint_validation/test2/predictions.csv",
                0.60,
            ),
        },
    }
    model_data = {
        dataset: {
            model: _load_prediction(path, threshold)
            for model, (path, threshold) in models.items()
        }
        for dataset, models in paths.items()
    }
    for dataset in model_data.values():
        _validate_alignment(dataset["ProtT5-3Seed"], dataset["Archived 1D+3D"])
    results = {
        dataset: {
            **PAPER_BENCHMARKS[dataset],
            **{model: _core_metrics(data) for model, data in models.items()},
        }
        for dataset, models in model_data.items()
    }
    _write_core_table(output, results)
    _plot_benchmark(output, results)
    _plot_calibration(output, model_data, results)
    comparisons = {
        dataset: _paired_bootstrap(
            models["ProtT5-3Seed"],
            models["Archived 1D+3D"],
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + index,
        )
        for index, (dataset, models) in enumerate(model_data.items())
    }
    _plot_bootstrap(output, comparisons)
    ablations = {spec.name: _load_ablation(spec) for spec in ABLATIONS}
    _plot_ablation(output, ablations)
    _plot_roc_pr_curves(output, model_data, results)
    summary = {
        "protocol": {
            "paper_benchmarks": {
                "ToxMSRC": "2025ToxMSRC.docx; identical Train/Test1/Test2 split",
                "ToxPLTC": "C:/Users/xly74/Downloads/2026JCIM.docx; identical split",
            },
            "test_labels_used_for_selection": False,
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(bootstrap_seed),
            "paper_missing_metrics": "NR; never imputed",
        },
        "core_metrics": results,
        "paired_comparison": comparisons,
        "oof_ablations": ablations,
    }
    save_json(summary, output / "analysis_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/publication_analysis")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = generate_publication_analysis(
        args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary["paired_comparison"], indent=2))


if __name__ == "__main__":
    main()
