"""Generate the mentor-facing report using only the public 3D default predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_curve, roc_curve

from ghxtox._legacy.plm_textcnn import _probability_metrics
from ghxtox.publication_analysis import (
    ABLATIONS,
    COLORS,
    CORE_METRICS,
    DISPLAY_NAMES,
    PAPER_BENCHMARKS,
    PredictionData,
    _calibration_bins,
    _core_metrics,
    _load_ablation,
    _load_prediction,
    _metrics_with_decisions,
    _save_figure,
    _stratified_indices,
    _style_axis,
)
from ghxtox.utils import load_json, save_json


DEFAULT_PATHS = {
    "Test1": "runs/default_multidimensional_fusion/test1_predictions.csv",
    "Test2": "runs/default_multidimensional_fusion/test2_predictions.csv",
}
DEFAULT_THRESHOLD = 0.47022256531284556
DEFAULT_MODEL_KEY = "GHXTox-Multidimensional-Fusion"
LEGACY_MODEL_KEY = "GHXTox-3D-Chemical-Adapter-Residual"
COLORS[DEFAULT_MODEL_KEY] = COLORS["Archived 1D+3D"]
COLORS[LEGACY_MODEL_KEY] = COLORS["Archived 1D+3D"]
PAPER_METRICS = CORE_METRICS[:7]
DEFAULT_COMPONENT_SUMMARIES = {
    "structure": "runs/3d_v2_multiseed/ensemble/summary.json",
    "adapter": "runs/prott5_structure_chemical_cross_attention_oof/summary.json",
    "residual": "runs/bacc_prioritized_decision_residual/summary.json",
}


def _default_metrics(
    results: dict[str, dict[str, dict[str, float]]], dataset: str
) -> dict[str, float]:
    """Read the current default while accepting legacy report fixtures."""

    if DEFAULT_MODEL_KEY in results[dataset]:
        return results[dataset][DEFAULT_MODEL_KEY]
    return results[dataset]["ProtT5-3Seed"]


def _normalise_component_metrics(values: dict[str, Any]) -> dict[str, float]:
    sensitivity = float(values.get("sensitivity", values.get("recall")))
    specificity = float(
        values.get(
            "specificity",
            2.0 * float(values["balanced_accuracy"]) - sensitivity,
        )
    )
    normalised = {
        "accuracy": float(values["accuracy"]),
        "balanced_accuracy": float(values["balanced_accuracy"]),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mcc": float(values["mcc"]),
        "auroc": float(values["auroc"]),
        "auprc": float(values["auprc"]),
        "brier": float(values["brier"]),
        "ece_10": float(values["ece_10"]),
    }
    if not all(np.isfinite(value) for value in normalised.values()):
        raise ValueError("Default-component OOF metrics must all be finite.")
    return normalised


def _load_default_component_ablations() -> dict[str, dict[str, float]]:
    structure = load_json(DEFAULT_COMPONENT_SUMMARIES["structure"])
    adapter = load_json(DEFAULT_COMPONENT_SUMMARIES["adapter"])
    residual = load_json(DEFAULT_COMPONENT_SUMMARIES["residual"])
    if residual.get("promotion_authorized") is not True:
        raise ValueError("The BACC residual has not passed its frozen promotion gate.")
    return {
        "ProtT5 sequence expert": _load_ablation(ABLATIONS[3]),
        "3D chemical expert": _normalise_component_metrics(
            structure["reference_metrics"]
        ),
        "Frozen 1D+3D fusion": _normalise_component_metrics(adapter["baseline"]),
        "+ chemical adapter": _normalise_component_metrics(adapter["adapter"]),
        "+ BACC decision residual": _normalise_component_metrics(
            residual["selected"]["metrics"]
        ),
    }


def _bootstrap_intervals(
    data: PredictionData,
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    point = _core_metrics(data)
    samples = {metric: np.empty(iterations, dtype=np.float64) for metric in CORE_METRICS}
    rng = np.random.default_rng(seed)
    for iteration in range(iterations):
        indices = _stratified_indices(data.labels, rng)
        values = (
            _metrics_with_decisions(
                data.labels[indices],
                data.probabilities[indices],
                data.predictions[indices],
                threshold=data.threshold,
            )
            if data.predictions is not None
            else _probability_metrics(
                data.labels[indices],
                data.probabilities[indices],
                threshold=data.threshold,
            )
        )
        for metric in CORE_METRICS:
            samples[metric][iteration] = values[metric]
    return {
        metric: {
            "estimate": float(point[metric]),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for metric, values in samples.items()
    }


def _paper_delta_rows(
    results: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for dataset in ("Test1", "Test2"):
        default = _default_metrics(results, dataset)
        for paper in ("ToxMSRC", "ToxPLTC"):
            for metric in PAPER_METRICS:
                reference = results[dataset][paper][metric]
                rows.append(
                    {
                        "dataset": dataset,
                        "reference": paper,
                        "metric": metric,
                        "default_value": default[metric],
                        "reference_value": reference if np.isfinite(reference) else "NR",
                        "delta": default[metric] - reference if np.isfinite(reference) else "NR",
                    }
                )
    return rows


def _write_tables(
    output_dir: Path,
    results: dict[str, dict[str, dict[str, float]]],
    intervals: dict[str, dict[str, dict[str, float]]],
    delta_rows: list[dict[str, str | float]],
) -> None:
    with (output_dir / "core_metrics_default_only.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "model", *CORE_METRICS])
        writer.writeheader()
        for dataset in ("Test1", "Test2"):
            for model in ("ToxMSRC", "ToxPLTC", DEFAULT_MODEL_KEY):
                values = results[dataset][model]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "model": model,
                        **{
                            metric: (
                                f"{values[metric]:.9g}"
                                if np.isfinite(values[metric])
                                else "NR"
                            )
                            for metric in CORE_METRICS
                        },
                    }
                )
    with (output_dir / "default_bootstrap_ci.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "metric", "estimate", "lower", "upper"]
        )
        writer.writeheader()
        for dataset, metrics in intervals.items():
            for metric, values in metrics.items():
                writer.writerow({"dataset": dataset, "metric": metric, **values})
    with (output_dir / "paper_deltas.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(delta_rows[0]))
        writer.writeheader()
        writer.writerows(delta_rows)


def _plot_benchmark(
    output_dir: Path,
    results: dict[str, dict[str, dict[str, float]]],
) -> None:
    x = np.arange(len(PAPER_METRICS))
    figure, axes = plt.subplots(2, 1, figsize=(11.8, 7.8), constrained_layout=True)
    for axis, dataset in zip(axes, ("Test1", "Test2"), strict=True):
        _style_axis(axis)
        for model in ("ToxMSRC", "ToxPLTC", DEFAULT_MODEL_KEY):
            values = np.asarray([results[dataset][model][metric] for metric in PAPER_METRICS])
            axis.plot(
                x,
                values,
                marker="o",
                markersize=6.0,
                linewidth=2.5 if model == DEFAULT_MODEL_KEY else 1.6,
                color=COLORS[model],
                label=model,
            )
        axis.set_xticks(x, [DISPLAY_NAMES[metric] for metric in PAPER_METRICS])
        axis.set_ylim((0.82, 0.995) if dataset == "Test1" else (0.60, 0.99))
        axis.set_ylabel("Score")
        axis.set_title(
            f"{dataset} | identical data split",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.025),
    )
    figure.suptitle(
        "Default model vs same-dataset studies",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    _save_figure(figure, output_dir, "figure1_default_paper_benchmark")


def _plot_paper_deltas(
    output_dir: Path,
    results: dict[str, dict[str, dict[str, float]]],
) -> None:
    row_labels = [
        "Test1 vs ToxMSRC",
        "Test1 vs ToxPLTC",
        "Test2 vs ToxMSRC",
        "Test2 vs ToxPLTC",
    ]
    values = np.full((len(row_labels), len(PAPER_METRICS)), np.nan, dtype=np.float64)
    for row, (dataset, paper) in enumerate(
        (("Test1", "ToxMSRC"), ("Test1", "ToxPLTC"), ("Test2", "ToxMSRC"), ("Test2", "ToxPLTC"))
    ):
        for column, metric in enumerate(PAPER_METRICS):
            reference = results[dataset][paper][metric]
            if np.isfinite(reference):
                values[row, column] = 100.0 * (
                    _default_metrics(results, dataset)[metric] - reference
                )
    limit = float(np.nanmax(np.abs(values)))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#E6E8EB")
    figure, axis = plt.subplots(figsize=(11.8, 4.0), constrained_layout=True)
    image = axis.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(PAPER_METRICS)), [DISPLAY_NAMES[m] for m in PAPER_METRICS])
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(
        "Default-model improvement over each reference | percentage points",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            display_value = 0.0 if np.isfinite(value) and abs(value) < 0.005 else value
            axis.text(
                column,
                row,
                "NR" if not np.isfinite(value) else f"{display_value:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if np.isfinite(value) and abs(value) > 0.65 * limit else COLORS["ink"],
            )
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Percentage-point gain")
    _save_figure(figure, output_dir, "figure2_default_paper_deltas")


def _plot_default_roc_pr(
    output_dir: Path,
    model_data: dict[str, PredictionData],
    results: dict[str, dict[str, dict[str, float]]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), constrained_layout=True)
    for row, dataset in enumerate(("Test1", "Test2")):
        data = model_data[dataset]
        roc_axis, pr_axis = axes[row]
        _style_axis(roc_axis)
        _style_axis(pr_axis)
        fpr, tpr, _ = roc_curve(data.labels, data.probabilities)
        precision, recall, _ = precision_recall_curve(data.labels, data.probabilities)
        prevalence = float(np.mean(data.labels))
        roc_axis.plot([0, 1], [0, 1], "--", color=COLORS["muted"], linewidth=1.0)
        roc_axis.plot(
            fpr,
            tpr,
            color=COLORS["Archived 1D+3D"],
            linewidth=2.4,
            label=f"AUROC {_default_metrics(results, dataset)['auroc']:.3f}",
        )
        pr_axis.axhline(
            prevalence,
            linestyle="--",
            color=COLORS["muted"],
            linewidth=1.0,
            label=f"Prevalence {prevalence:.3f}",
        )
        pr_axis.plot(
            recall,
            precision,
            color=COLORS["Archived 1D+3D"],
            linewidth=2.4,
            label=f"AUPRC {_default_metrics(results, dataset)['auprc']:.3f}",
        )
        roc_axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate")
        pr_axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
        roc_axis.set_title(f"{dataset} ROC", loc="left", fontsize=12, fontweight="bold")
        pr_axis.set_title(f"{dataset} precision-recall", loc="left", fontsize=12, fontweight="bold")
        roc_axis.legend(loc="lower right", frameon=False)
        pr_axis.legend(loc="lower left", frameon=False)
    figure.suptitle(
        "Frozen default-model ranking performance",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    _save_figure(figure, output_dir, "figure3_default_roc_pr")


def _plot_default_calibration(
    output_dir: Path,
    model_data: dict[str, PredictionData],
    results: dict[str, dict[str, dict[str, float]]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    with (output_dir / "default_calibration_bins.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["dataset", "bin", "lower", "upper", "mean_probability", "observed_frequency", "count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for axis, dataset in zip(axes, ("Test1", "Test2"), strict=True):
            data = model_data[dataset]
            bins = _calibration_bins(data.labels, data.probabilities)
            writer.writerows({"dataset": dataset, **row} for row in bins)
            _style_axis(axis)
            axis.plot([0, 1], [0, 1], "--", color=COLORS["muted"], linewidth=1.0)
            axis.plot(
                [row["mean_probability"] for row in bins],
                [row["observed_frequency"] for row in bins],
                marker="o",
                linewidth=2.2,
                color=COLORS["Archived 1D+3D"],
                label=(
                    f"Brier {_default_metrics(results, dataset)['brier']:.3f} | "
                    f"ECE-10 {_default_metrics(results, dataset)['ece_10']:.3f}"
                ),
            )
            axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed toxicity rate")
            axis.set_title(f"{dataset} reliability", loc="left", fontsize=12, fontweight="bold")
            axis.legend(loc="upper left", frameon=False)
    figure.suptitle(
        "Default-model probability calibration",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    _save_figure(figure, output_dir, "figure4_default_calibration")


def _plot_default_ablation(
    output_dir: Path,
    ablations: dict[str, dict[str, float]],
) -> None:
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
    figure, axis = plt.subplots(figsize=(12.0, 4.5), constrained_layout=True)
    image = axis.imshow(scaled, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(metrics)), [DISPLAY_NAMES[m] for m in metrics])
    axis.set_yticks(np.arange(len(names)), names)
    axis.set_title(
        "Group-aware OOF component ablation",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
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
    figure.colorbar(
        image,
        ax=axis,
        fraction=0.025,
        pad=0.02,
        label="Within-column desirability",
    )
    _save_figure(figure, output_dir, "figure5_default_oof_ablation")
    with (output_dir / "default_oof_ablation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *CORE_METRICS])
        writer.writeheader()
        for name, values in ablations.items():
            writer.writerow({"model": name, **{metric: values[metric] for metric in CORE_METRICS}})


def _plot_default_architecture(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(14.0, 5.0), constrained_layout=True)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    def add_box(
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        color: str,
        *,
        fontsize: float = 9.0,
    ) -> None:
        axis.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=color,
                edgecolor="#27313A",
                linewidth=1.0,
            )
        )
        axis.text(
            x + width / 2.0,
            y + height / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#172027",
        )

    def add_arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        axis.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "-|>", "color": "#45515C", "linewidth": 1.4},
        )

    box_h = 0.18
    add_box(0.01, 0.43, 0.09, box_h, "Peptide\nFASTA", "#DDE3E8")

    top_y = 0.69
    top_boxes = (
        (0.14, 0.14, "ProtT5-XL\npooled + residue", "#A8DADC"),
        (0.32, 0.15, "15-member TextCNN\n3 seeds x 5 folds", "#C8E6C9"),
        (0.51, 0.11, "Sequence\nprobability", "#B8E0B8"),
    )
    lower_y = 0.17
    lower_boxes = (
        (0.14, 0.14, "ESM2 + coordinates\n+ pLDDT", "#DDE3E8"),
        (0.32, 0.15, "C-alpha EGNN +\nall-atom chemical sites", "#A8DADC"),
        (0.51, 0.11, "3D expert\nprobability", "#B8E0B8"),
    )
    for y, boxes in ((top_y, top_boxes), (lower_y, lower_boxes)):
        for x, width, label, color in boxes:
            add_box(x, y, width, box_h, label, color, fontsize=8.7)
        for current, following in zip(boxes, boxes[1:]):
            add_arrow(
                current[0] + current[1],
                y + box_h / 2.0,
                following[0],
                y + box_h / 2.0,
            )
    add_arrow(0.10, 0.52, 0.14, top_y + box_h / 2.0)
    add_arrow(0.10, 0.52, 0.14, lower_y + box_h / 2.0)

    add_box(0.66, 0.43, 0.10, box_h, "0.62 / 0.38\nOOF fusion", "#F7D794")
    add_arrow(0.62, top_y + box_h / 2.0, 0.66, 0.55)
    add_arrow(0.62, lower_y + box_h / 2.0, 0.66, 0.49)
    add_box(
        0.79,
        0.43,
        0.10,
        box_h,
        "ProtT5-to-3D\nchemical adapter",
        "#C8E6C9",
    )
    add_arrow(0.76, 0.52, 0.79, 0.52)
    add_box(0.91, 0.65, 0.08, 0.16, "Toxicity\nprobability", "#B8E0B8")
    add_arrow(0.89, 0.55, 0.91, 0.70)
    add_box(0.91, 0.23, 0.08, 0.16, "BACC gate\ndecision", "#F7D794")
    add_arrow(0.89, 0.49, 0.91, 0.34)
    add_arrow(0.565, top_y, 0.93, 0.39)
    axis.text(
        0.01,
        0.93,
        "GHXTox 3D default model and leakage-controlled validation",
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="center",
        color=COLORS["ink"],
    )
    axis.text(
        0.115,
        0.78,
        "sequence evidence",
        fontsize=9,
        fontweight="bold",
        color=COLORS["muted"],
        rotation=90,
        ha="center",
        va="center",
    )
    axis.text(
        0.115,
        0.26,
        "3D chemical evidence",
        fontsize=9,
        fontweight="bold",
        color=COLORS["muted"],
        rotation=90,
        ha="center",
        va="center",
    )
    axis.text(
        0.84,
        0.14,
        "Decision residual preserves toxicity probability and only changes selected disagreements",
        fontsize=8.5,
        color=COLORS["muted"],
        ha="center",
        va="center",
    )
    _save_figure(figure, output_dir, "figure6_default_architecture")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reproducibility_manifest(output_dir: Path) -> dict[str, Any]:
    paths = [
        Path("configs/default_3d_hybrid.json"),
        Path("configs/default.json"),
        Path("configs/default_prott5.json"),
        Path("data/folds/train_cdhit080_fallback_folds.csv"),
        Path("runs/plm_textcnn_prott5_bsmote_multiseed/oof_predictions.csv"),
        Path(DEFAULT_COMPONENT_SUMMARIES["structure"]),
        Path(DEFAULT_COMPONENT_SUMMARIES["adapter"]),
        Path(DEFAULT_COMPONENT_SUMMARIES["residual"]),
        Path("runs/bacc_prioritized_decision_residual/decision.json"),
        Path("runs/bacc_prioritized_decision_residual/deployable_gate.json"),
        *(Path(path) for path in DEFAULT_PATHS.values()),
        *(
            Path(path)
            .with_name(Path(path).name.replace("predictions", "metrics"))
            .with_suffix(".json")
            for path in DEFAULT_PATHS.values()
        ),
        Path("runs/3d_v2_conformer_ensemble_candidate/best_model.pt"),
        Path("runs/prott5_structure_chemical_cross_attention_candidate/best_model.pt"),
        Path("2025ToxMSRC.docx"),
        Path.home() / "Downloads/2026JCIM.docx",
        Path(__file__),
        output_dir / "REPORT.md",
    ]
    for directory in (
        Path("runs/plm_textcnn_prott5_bsmote"),
        Path("runs/plm_textcnn_prott5_bsmote_seed123"),
        Path("runs/plm_textcnn_prott5_bsmote_seed2025"),
    ):
        paths.extend(sorted(directory.glob("fold*_best_model.pt")))
    artifacts = []
    for path in paths:
        resolved = path.resolve()
        artifacts.append(
            {
                "path": str(path),
                "exists": resolved.is_file(),
                "bytes": resolved.stat().st_size if resolved.is_file() else None,
                "sha256": _sha256(resolved) if resolved.is_file() else None,
            }
        )
    manifest = {
        "report_scope": "Only the public GHXTox 3D default test predictions are compared with papers.",
        "excluded_from_report": "All non-default model test predictions.",
        "default_threshold": DEFAULT_THRESHOLD,
        "test_labels_used_for_model_or_threshold_selection": False,
        "command": "python -m ghxtox.mentor_report --bootstrap-iterations 5000 --output-dir reports/mentor_report",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifacts": artifacts,
    }
    save_json(manifest, output_dir / "reproducibility_manifest.json")
    return manifest


def generate_mentor_report(
    output_dir: str | Path,
    *,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 20260802,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_data = {
        dataset: _load_prediction(path, DEFAULT_THRESHOLD)
        for dataset, path in DEFAULT_PATHS.items()
    }
    results = {
        dataset: {
            **PAPER_BENCHMARKS[dataset],
            DEFAULT_MODEL_KEY: _core_metrics(data),
        }
        for dataset, data in model_data.items()
    }
    intervals = {
        dataset: _bootstrap_intervals(
            data,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + index,
        )
        for index, (dataset, data) in enumerate(model_data.items())
    }
    delta_rows = _paper_delta_rows(results)
    _write_tables(output, results, intervals, delta_rows)
    _plot_benchmark(output, results)
    _plot_paper_deltas(output, results)
    _plot_default_roc_pr(output, model_data, results)
    _plot_default_calibration(output, model_data, results)
    ablations = _load_default_component_ablations()
    _plot_default_ablation(output, ablations)
    _plot_default_architecture(output)
    manifest = _reproducibility_manifest(output)
    summary = {
        "protocol": {
            "report_scope": "default test predictions only",
            "threshold": DEFAULT_THRESHOLD,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
            "test_labels_used_for_selection": False,
            "paper_missing_metrics": "NR; never imputed",
            "public_model": DEFAULT_MODEL_KEY,
        },
        "core_metrics": results,
        "default_bootstrap_ci": intervals,
        "paper_deltas": delta_rows,
        "oof_ablations": ablations,
        "bacc_residual": load_json(DEFAULT_COMPONENT_SUMMARIES["residual"]),
        "reproducibility_manifest": manifest,
    }
    save_json(summary, output / "mentor_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/mentor_report")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = generate_mentor_report(
        args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary["protocol"], indent=2))


if __name__ == "__main__":
    main()
