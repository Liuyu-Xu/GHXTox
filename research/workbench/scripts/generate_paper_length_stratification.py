"""Generate paper figures for length-stratified multidimensional predictions.

The comparison deliberately separates three effects:

1. the previous frozen multidimensional default;
2. the current LoRA-ProtT5--BiMamba fusion with the single-structure 3D expert;
3. the canonical current default with the multi-structure-enabled 3D expert.

The final two variants share the current 1D probabilities, 0.72/0.28 fusion
weights, and the OOF-frozen threshold.  Their only difference is the 3D input
probability.  This prevents improvements in the 1D expert from being
misattributed to conformer-ensemble enhancement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)


LENGTH_BINS = (("11--15", 11, 15), ("16--30", 16, 30), ("31--50", 31, 50))
MODEL_LABELS = (
    "Previous multidimensional default",
    "Current + single-structure 3D",
    "Current multi-structure-enabled default (fallback)",
)
METRICS = (
    ("balanced_accuracy", "BACC"),
    ("mcc", "MCC"),
    ("auprc", "AUPRC"),
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def _analyze_dataset(
    dataset: str,
    current_dir: Path,
    single_structure_dir: Path,
    previous_dir: Path,
) -> dict[str, object]:
    current_rows = _rows(current_dir / f"{dataset}_predictions.csv")
    single_rows = {
        row["sample_id"]: row
        for row in _rows(single_structure_dir / f"{dataset}_predictions.csv")
    }
    previous_rows = {
        row["sample_id"]: row
        for row in _rows(previous_dir / f"{dataset}_predictions.csv")
    }

    sample_ids = [row["sample_id"] for row in current_rows]
    if set(sample_ids) != set(single_rows) or set(sample_ids) != set(previous_rows):
        raise ValueError(f"{dataset}: sample IDs differ across comparison inputs.")

    labels = np.asarray([int(row["label"]) for row in current_rows], dtype=int)
    sequences = [single_rows[sample_id]["sequence"] for sample_id in sample_ids]
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=int)
    p_1d = np.asarray([float(row["p_1d"]) for row in current_rows], dtype=float)
    p_3d_multi = np.asarray([float(row["p_3d"]) for row in current_rows], dtype=float)
    p_3d_single = np.asarray(
        [float(single_rows[sample_id]["toxicity_probability"]) for sample_id in sample_ids],
        dtype=float,
    )
    p_previous = np.asarray(
        [float(previous_rows[sample_id]["toxicity_probability"]) for sample_id in sample_ids],
        dtype=float,
    )
    pred_previous = np.asarray(
        [int(previous_rows[sample_id]["prediction"]) for sample_id in sample_ids],
        dtype=int,
    )

    weights = {round(float(row["weight_1d"]), 12) for row in current_rows}
    thresholds = {round(float(row["decision_threshold"]), 12) for row in current_rows}
    if len(weights) != 1 or len(thresholds) != 1:
        raise ValueError(f"{dataset}: current operating point is not constant.")
    weight_1d = float(next(iter(weights)))
    threshold = float(next(iter(thresholds)))
    if not np.isclose(weight_1d, 0.72, atol=1e-12):
        raise ValueError(f"{dataset}: expected current 1D weight 0.72, got {weight_1d}.")

    p_single = weight_1d * p_1d + (1.0 - weight_1d) * p_3d_single
    p_multi = weight_1d * p_1d + (1.0 - weight_1d) * p_3d_multi
    exported_multi = np.asarray(
        [float(row["toxicity_probability"]) for row in current_rows], dtype=float
    )
    if not np.allclose(p_multi, exported_multi, atol=2e-10, rtol=0.0):
        raise ValueError(f"{dataset}: canonical probabilities do not match 0.72/0.28 fusion.")

    pred_single = (p_single >= threshold).astype(int)
    pred_multi = (p_multi >= threshold).astype(int)
    strata: dict[str, object] = {}
    for label, lower, upper in LENGTH_BINS:
        selected = (lengths >= lower) & (lengths <= upper)
        y = labels[selected]
        variants = {
            MODEL_LABELS[0]: _metric_values(
                y, p_previous[selected], pred_previous[selected]
            ),
            MODEL_LABELS[1]: _metric_values(y, p_single[selected], pred_single[selected]),
            MODEL_LABELS[2]: _metric_values(y, p_multi[selected], pred_multi[selected]),
        }
        strata[label] = {
            "n": int(selected.sum()),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "models": variants,
            "multi_minus_single": {
                key: variants[MODEL_LABELS[2]][key] - variants[MODEL_LABELS[1]][key]
                for key, _ in METRICS
            },
        }

    difference = np.abs(p_3d_multi - p_3d_single)
    return {
        "dataset": dataset,
        "n": len(current_rows),
        "weight_1d": weight_1d,
        "weight_3d": 1.0 - weight_1d,
        "threshold": threshold,
        "sources": {
            "previous_default": str(previous_dir / f"{dataset}_predictions.csv"),
            "current_1d_and_multi_3d": str(
                current_dir / f"{dataset}_predictions.csv"
            ),
            "single_structure_3d": str(
                single_structure_dir / f"{dataset}_predictions.csv"
            ),
        },
        "three_d_probability_comparison": {
            "max_absolute_multi_minus_single": float(difference.max(initial=0.0)),
            "count_above_1e_minus_5": int(np.sum(difference > 1e-5)),
            "interpretation": (
                "The held-out multi-structure-enabled 3D stream is numerically "
                "equivalent to the single-structure fallback at the metric level."
            ),
        },
        "strata": strata,
    }


def _plot_dataset(result: dict[str, object], output_dir: Path) -> None:
    dataset = str(result["dataset"])
    strata = result["strata"]
    bin_labels = list(strata)
    x = np.arange(len(bin_labels), dtype=float)
    width = 0.24
    colors = ("#808080", "#3B6FB6", "#D9822B")
    hatches = ("", "", "//")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.25), sharey=True)
    for axis, (metric_key, metric_label) in zip(axes, METRICS):
        for model_index, model_label in enumerate(MODEL_LABELS):
            values = [
                float(strata[length_label]["models"][model_label][metric_key])
                for length_label in bin_labels
            ]
            positions = x + (model_index - 1) * width
            bars = axis.bar(
                positions,
                values,
                width,
                label=model_label,
                color=colors[model_index],
                edgecolor="#222222",
                linewidth=0.55,
                hatch=hatches[model_index],
                zorder=3,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.3f}" for value in values],
                padding=2,
                fontsize=7.2,
                rotation=90,
            )
        axis.set_title(metric_label)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [
                (
                    f"{length_label} aa\n"
                    f"n={strata[length_label]['n']}, +={strata[length_label]['n_positive']}"
                )
                for length_label in bin_labels
            ]
        )
        axis.set_xlabel("Peptide length stratum")
        axis.set_ylim(0.0, 1.06)
        axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.6, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.suptitle(
        f"{dataset.capitalize()}: length-stratified multidimensional performance",
        y=1.055,
        fontsize=12,
        fontweight="bold",
    )
    max_difference = result["three_d_probability_comparison"][
        "max_absolute_multi_minus_single"
    ]
    fig.text(
        0.5,
        0.012,
        (
            "The two current variants hold the 1D stream, 0.72/0.28 fusion and "
            "OOF-frozen threshold fixed. No effective held-out conformer residual "
            f"was present (maximum 3D probability difference = {max_difference:.2e})."
        ),
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.065, 0.98, 0.91))

    stem = output_dir / f"figure6_{dataset}_length_multistructure"
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_csv(results: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "length_bin",
        "n",
        "n_positive",
        "n_negative",
        "model",
        "balanced_accuracy",
        "mcc",
        "auprc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for length_label, stratum in result["strata"].items():
                for model_label, metrics in stratum["models"].items():
                    writer.writerow(
                        {
                            "dataset": result["dataset"],
                            "length_bin": length_label,
                            "n": stratum["n"],
                            "n_positive": stratum["n_positive"],
                            "n_negative": stratum["n_negative"],
                            "model": model_label,
                            **{key: f"{float(metrics[key]):.10f}" for key, _ in METRICS},
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-dir", default="runs/default_multidimensional_fusion", type=Path
    )
    parser.add_argument(
        "--single-structure-dir", default="runs/3d_v2_default", type=Path
    )
    parser.add_argument(
        "--previous-dir", default="runs/default_3d_hybrid", type=Path
    )
    parser.add_argument(
        "--output-dir", default="reports/publication_analysis", type=Path
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        _analyze_dataset(
            dataset,
            args.current_dir,
            args.single_structure_dir,
            args.previous_dir,
        )
        for dataset in ("test1", "test2")
    ]
    for result in results:
        _plot_dataset(result, args.output_dir)

    payload = {
        "status": "completed",
        "comparison_scope": (
            "Latest 1D and operating point held fixed for the single-structure "
            "versus multi-structure-enabled comparison."
        ),
        "results": results,
    }
    json_path = args.output_dir / "figure6_length_multistructure_metrics.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        results, args.output_dir / "figure6_length_multistructure_metrics.csv"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
