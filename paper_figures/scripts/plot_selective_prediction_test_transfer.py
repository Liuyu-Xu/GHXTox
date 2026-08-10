#!/usr/bin/env python
"""Plot frozen-threshold selective-prediction transfer to Test1 and Test2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


INTERNAL_METHODS = (
    "full_binary_predictor",
    "frozen_threshold_accepted_set",
    "equal_count_random_rejection",
)
PUBLIC_NAMES = {
    "full_binary_predictor": "Full binary predictor",
    "frozen_threshold_accepted_set": "Frozen-threshold accepted set",
    "equal_count_random_rejection": "Equal-count random rejection",
}
METRICS = (
    "accepted_error_rate",
    "accepted_mcc",
    "accepted_balanced_accuracy",
    "accepted_sensitivity",
    "accepted_specificity",
)
METRIC_LABELS = {
    "accepted_error_rate": "Accepted error",
    "accepted_mcc": "MCC",
    "accepted_balanced_accuracy": "Balanced accuracy",
    "accepted_sensitivity": "Sensitivity",
    "accepted_specificity": "Specificity",
}
SPLITS = ("Test1", "Test2")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _root()
    result = (
        root
        / "runs"
        / "selective_prediction_frozen_threshold_transfer_current_default_20260810_v2"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-csv", type=Path, default=result / "method_point_estimates.csv")
    parser.add_argument(
        "--intervals-csv", type=Path, default=result / "method_bootstrap_intervals.csv"
    )
    parser.add_argument(
        "--contrasts-csv", type=Path, default=result / "paired_bootstrap_contrasts.csv"
    )
    parser.add_argument("--split-summary-csv", type=Path, default=result / "split_summary.csv")
    parser.add_argument("--summary-json", type=Path, default=result / "summary.json")
    parser.add_argument(
        "--protocol-json",
        type=Path,
        default=root
        / "configs"
        / "selective_prediction_frozen_threshold_transfer_current_default_v2.json",
    )
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=root
        / "runs"
        / "selective_prediction_current_default_oof_20260810_v2"
        / "selected_policy.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "paper_figures"
        / "output"
        / "selective_prediction_frozen_threshold_transfer_current_default_v2",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty source table: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _one(
    rows: list[dict[str, str]],
    split_name: str,
    method: str | None = None,
    metric: str | None = None,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["split_name"] == split_name
        and (method is None or row.get("method") == method)
        and (metric is None or row.get("metric") == metric)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for split={split_name}, method={method}, metric={metric}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _validate(
    points: list[dict[str, str]],
    intervals: list[dict[str, str]],
    contrasts: list[dict[str, str]],
    split_summary: list[dict[str, str]],
    summary: dict[str, Any],
    protocol: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    if len(points) != 6 or len(intervals) != 36 or len(contrasts) != 20:
        raise ValueError("Unexpected transfer-table row counts.")
    if protocol.get("artifact_type") != (
        "ghxtox_selective_prediction_frozen_threshold_test_transfer_protocol"
    ):
        raise ValueError("Unexpected transfer protocol.")
    if policy.get("profile") != "margin_only" or policy.get("feature_names") != [
        "decision_closeness"
    ]:
        raise ValueError("The figure requires the frozen margin-only policy.")
    if policy.get("test_labels_used") is not False:
        raise ValueError("The policy record violates the no-Test-label boundary.")
    if float(policy["abstention_threshold"]) != float(
        protocol["frozen_policy_contract"]["uncertainty_score_cutoff"]
    ):
        raise ValueError("The plotted cutoff differs from the frozen protocol.")
    decision = summary.get("decision", {})
    for key in (
        "policy_reselected",
        "test_specific_cutoff_used",
        "test_specific_coverage_forced",
        "test_labels_used_for_selection",
        "canonical_binary_model_changed",
        "fusion_or_binary_threshold_changed",
        "new_independent_dataset_claimed",
    ):
        if decision.get(key) is not False:
            raise ValueError(f"Transfer decision scope guard failed: {key}")
    for split_name in SPLITS:
        split_row = _one(split_summary, split_name)
        if split_row["transfer_decision"] != "strong_support":
            raise ValueError(f"Unexpected retained outcome for {split_name}.")
        for method in INTERNAL_METHODS:
            _one(points, split_name, method)
            for metric in ("coverage",) + METRICS:
                row = _one(intervals, split_name, method, metric)
                if int(row["bootstrap_replicates"]) != 5000:
                    raise ValueError("All plotted intervals must use 5,000 replicates.")
        if abs(float(split_row["realized_coverage"]) - 0.9) < 1e-6:
            raise ValueError("The benchmark coverage appears to have been forced to 90%.")


def _public_metric_rows(
    points: list[dict[str, str]], intervals: list[dict[str, str]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split_name in SPLITS:
        for method in INTERNAL_METHODS:
            point = _one(points, split_name, method)
            for metric in ("coverage",) + METRICS:
                interval = _one(intervals, split_name, method, metric)
                result.append(
                    {
                        "split_name": split_name,
                        "model_and_policy": "Frozen GHXTox predictor with OOF-derived margin-only abstention cutoff",
                        "method": PUBLIC_NAMES[method],
                        "accepted_n": point["accepted_n"],
                        "rejected_n": point["rejected_n"],
                        "metric": "Coverage" if metric == "coverage" else METRIC_LABELS[metric],
                        "estimate": point[metric],
                        "ci95_lower": interval["ci95_lower"],
                        "ci95_upper": interval["ci95_upper"],
                        "uncertainty": interval["uncertainty_type"],
                        "replicates": interval["bootstrap_replicates"],
                    }
                )
    return result


def _public_contrast_rows(contrasts: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in contrasts:
        comparator = PUBLIC_NAMES[row["comparator"]]
        definition = row["positive_improvement_definition"]
        for internal, public in PUBLIC_NAMES.items():
            definition = definition.replace(internal, public)
        result.append(
            {
                "split_name": row["split_name"],
                "comparison": f"{comparator} versus {PUBLIC_NAMES['frozen_threshold_accepted_set']}",
                "metric": METRIC_LABELS[row["metric"]],
                "positive_improvement_definition": definition,
                "point_estimate": row["point_estimate"],
                "ci95_lower": row["ci95_lower"],
                "ci95_upper": row["ci95_upper"],
                "bootstrap_probability_gt_0": row["bootstrap_probability_gt_0"],
                "replicates": row["bootstrap_replicates"],
            }
        )
    return result


def _public_split_rows(split_summary: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in split_summary:
        result.append(
            {
                "split_name": row["split_name"],
                "sample_count": row["sample_count"],
                "positive_count": row["positive_count"],
                "negative_count": row["negative_count"],
                "oof_derived_uncertainty_cutoff": row["uncertainty_score_cutoff"],
                "oof_target_coverage_reference": row["oof_target_coverage_reference"],
                "realized_test_coverage": row["realized_coverage"],
                "accepted_n": row["accepted_n"],
                "rejected_n": row["rejected_n"],
                "transfer_decision": "Strong fixed-threshold support",
            }
        )
    return result


def _write_exact_table(path: Path, points: list[dict[str, str]]) -> None:
    lines = [
        "| Split | Method | Coverage | Accepted | Error | MCC | BACC | SN | SP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in SPLITS:
        for method in INTERNAL_METHODS:
            row = _one(points, split_name, method)
            lines.append(
                "| "
                + " | ".join(
                    [
                        split_name,
                        PUBLIC_NAMES[method],
                        f"{100 * float(row['coverage']):.2f}%",
                        row["accepted_n"],
                        f"{100 * float(row['accepted_error_rate']):.2f}%",
                        f"{float(row['accepted_mcc']):.4f}",
                        f"{float(row['accepted_balanced_accuracy']):.4f}",
                        f"{float(row['accepted_sensitivity']):.4f}",
                        f"{float(row['accepted_specificity']):.4f}",
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _draw(
    output: Path,
    points: list[dict[str, str]],
    intervals: list[dict[str, str]],
    split_summary: list[dict[str, str]],
    cutoff: float,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "ghxtox-selective-test-transfer-v1",
        }
    )
    blue = "#0072B2"
    dark = "#3F4650"
    gray = "#8E98A7"
    amber = "#E69F00"
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 3.15),
        gridspec_kw={"width_ratios": [0.78, 1.12, 1.12]},
        constrained_layout=True,
    )

    # Panel A: fixed-cutoff realized coverage, not a test-specific percentile.
    x = np.arange(len(SPLITS), dtype=float)
    coverage = np.asarray(
        [100 * float(_one(split_summary, split_name)["realized_coverage"]) for split_name in SPLITS]
    )
    bars = axes[0].bar(x, coverage, width=0.58, color=blue, edgecolor="none", zorder=3)
    axes[0].axhline(
        90.0,
        color=amber,
        linewidth=1.0,
        linestyle=(0, (3, 2)),
        label="OOF target reference",
        zorder=2,
    )
    for bar, value in zip(bars, coverage):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value - 2.2,
            f"{value:.1f}%",
            ha="center",
            va="top",
            fontsize=7.5,
            fontweight="bold",
            color="white",
        )
    axes[0].text(
        0.5,
        0.04,
        rf"Frozen $U^*={cutoff:.4f}$",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=dark,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    axes[0].set_xticks(x, SPLITS)
    axes[0].set_ylabel("Realized coverage (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].legend(loc="lower left", bbox_to_anchor=(-0.03, 1.01), frameon=False)

    method_style = {
        "full_binary_predictor": (dark, "D"),
        "frozen_threshold_accepted_set": (blue, "o"),
        "equal_count_random_rejection": (gray, "s"),
    }
    offsets = {
        "full_binary_predictor": -0.18,
        "frozen_threshold_accepted_set": 0.0,
        "equal_count_random_rejection": 0.18,
    }
    plot_specs = (
        (axes[1], "accepted_error_rate", 100.0, "Accepted error (%)", (0.0, 8.0)),
        (axes[2], "accepted_mcc", 1.0, "MCC on evaluated predictions", (0.5, 1.0)),
    )
    legend_handles = []
    legend_labels = []
    for panel_index, (ax, metric, scale, ylabel, ylim) in enumerate(plot_specs):
        for method in INTERNAL_METHODS:
            color, marker = method_style[method]
            xs = x + offsets[method]
            ys = []
            lower = []
            upper = []
            for split_name in SPLITS:
                point = _one(points, split_name, method)
                interval = _one(intervals, split_name, method, metric)
                value = scale * float(point[metric])
                ys.append(value)
                lower.append(value - scale * float(interval["ci95_lower"]))
                upper.append(scale * float(interval["ci95_upper"]) - value)
            container = ax.errorbar(
                xs,
                ys,
                yerr=np.asarray([lower, upper]),
                fmt=marker,
                linestyle="none",
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                markersize=5.0,
                elinewidth=0.9,
                capsize=2.5,
                capthick=0.9,
                zorder=3 if method == "frozen_threshold_accepted_set" else 2,
            )
            if panel_index == 0:
                legend_handles.append(container)
                legend_labels.append(PUBLIC_NAMES[method])
        ax.set_xticks(x, SPLITS)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D9DDE3", linewidth=0.45, alpha=0.8, zorder=0)

    for index, ax in enumerate(axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.18,
            1.04,
            chr(ord("A") + index),
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="top",
        )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.67, 1.035),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    stem = output / "selective_prediction_frozen_threshold_transfer"
    metadata = {
        "Title": "Frozen-threshold selective-prediction transfer",
        "Author": "GHXTox study",
        "Subject": "Test1 and Test2 benchmark reliability",
        "Keywords": "selective prediction, abstention, fixed threshold, benchmark transfer",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inputs = (
        args.points_csv,
        args.intervals_csv,
        args.contrasts_csv,
        args.split_summary_csv,
        args.summary_json,
        args.protocol_json,
        args.policy_json,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty figure package: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    points = _read_csv(args.points_csv)
    intervals = _read_csv(args.intervals_csv)
    contrasts = _read_csv(args.contrasts_csv)
    split_summary = _read_csv(args.split_summary_csv)
    summary = _read_json(args.summary_json)
    protocol = _read_json(args.protocol_json)
    policy = _read_json(args.policy_json)
    _validate(points, intervals, contrasts, split_summary, summary, protocol, policy)
    cutoff = float(policy["abstention_threshold"])

    _write_csv(source_dir / "benchmark_transfer_metrics.csv", _public_metric_rows(points, intervals))
    _write_csv(source_dir / "paired_improvements.csv", _public_contrast_rows(contrasts))
    _write_csv(source_dir / "split_summary.csv", _public_split_rows(split_summary))
    _write_exact_table(output / "exact_results_table.md", points)
    _draw(output, points, intervals, split_summary, cutoff)

    roles = (
        "frozen transfer point estimates",
        "95% method intervals from the fixed-mask paired bootstrap",
        "positive-oriented paired transfer contrasts",
        "realized coverage and split-level transfer decision",
        "frozen transfer summary and selection firewall",
        "outcome-independent fixed-threshold transfer protocol",
        "training-OOF-derived immutable selective policy",
    )
    _write_csv(
        output / "input_manifest.csv",
        [
            {"source": str(path.resolve()), "sha256": _sha256(path), "role": role}
            for path, role in zip(inputs, roles)
        ],
    )
    _write_csv(
        output / "internal_to_publication_names.csv",
        [
            {
                "internal_name": internal,
                "publication_name": public,
                "scope": "fixed-threshold Test1/Test2 transfer",
            }
            for internal, public in PUBLIC_NAMES.items()
        ],
    )
    _write_csv(
        output / "panel_manifest.csv",
        [
            {
                "figure": "Supplementary selective-prediction transfer",
                "panel": "A",
                "data_file": "source_data/split_summary.csv",
                "script": "plot_selective_prediction_test_transfer.py",
                "model": "Frozen GHXTox predictor with OOF-derived abstention cutoff",
                "conclusion": "The unchanged OOF cutoff yields split-specific benchmark coverage rather than forced 90% coverage.",
            },
            {
                "figure": "Supplementary selective-prediction transfer",
                "panel": "B",
                "data_file": "source_data/benchmark_transfer_metrics.csv",
                "script": "plot_selective_prediction_test_transfer.py",
                "model": "Frozen GHXTox predictor with OOF-derived abstention cutoff",
                "conclusion": "The frozen cutoff lowers accepted error relative to both full prediction and equal-count random rejection on both benchmarks.",
            },
            {
                "figure": "Supplementary selective-prediction transfer",
                "panel": "C",
                "data_file": "source_data/benchmark_transfer_metrics.csv",
                "script": "plot_selective_prediction_test_transfer.py",
                "model": "Frozen GHXTox predictor with OOF-derived abstention cutoff",
                "conclusion": "The accepted sets have higher MCC on both benchmarks under the unchanged cutoff.",
            },
        ],
    )

    caption = (
        "The training-OOF-derived abstention cutoff transfers without test-specific tuning and "
        "identifies lower-risk accepted subsets on both established benchmarks. (A) Applying the "
        f"same frozen uncertainty-score cutoff (U* = {cutoff:.10f}) yields realized coverages of "
        "88.54% on Test1 (997/1,126) and 86.94% on Test2 (506/582); the dashed 90% line is the "
        "training-OOF target used to derive the cutoff, not a Test-set quota. (B) Accepted error "
        "decreases from 5.24% to 2.71% on Test1 and from 5.33% to 3.16% on Test2, whereas "
        "equal-count random rejection remains at 5.24% and 5.33%, respectively. (C) MCC increases "
        "from 0.870 to 0.933 on Test1 and from 0.667 to 0.778 on Test2. Error bars denote 95% "
        "intervals from 5,000 label-stratified fixed-mask paired bootstrap replicates; in each "
        "replicate the random comparator accepts the same number of resampled observations as the "
        "frozen mask. The score definition, binary threshold, and abstention cutoff were "
        "fixed from training OOF predictions before Test1/Test2 outcomes were examined. Test labels "
        "were used only for post-selection evaluation. These are fixed-policy transfer results on "
        "established benchmarks, not validation on a newly collected independent dataset. The MCC "
        "axis is restricted to 0.5--1.0 for readability; exact metrics are provided in the companion table."
    )
    (output / "caption.md").write_text(caption + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Frozen-threshold selective-prediction transfer figure\n\n"
        "Scientific question: does the already selected training-OOF abstention cutoff retain "
        "utility when applied unchanged to Test1 and Test2?\n\n"
        "The figure uses the canonical single-structure GHXTox predictions and the immutable "
        "`margin_only` policy. The sole feature is normalized decision closeness. The uncertainty "
        f"cutoff is {cutoff:.15f}, with `score <= cutoff` accepted. Test-specific quantiles, "
        "target coverages, feature selection, recalibration, and threshold changes are prohibited. "
        "The realized coverage is therefore an outcome. Labels are evaluation-only.\n\n"
        "Intervals use 5,000 label-stratified fixed-mask paired bootstrap replicates (seed "
        "20260825). The equal-count random control uses 5,000 fixed-cohort masks (seed 20260824), "
        "and one equal-count random mask inside each bootstrap replicate. No p-values or "
        "significance marks are used.\n\n"
        "Artwork is 178 mm wide and supplied as editable SVG, vector PDF, and 600-dpi PNG. "
        "Exact values are in `exact_results_table.md` and `source_data/benchmark_transfer_metrics.csv`. "
        "Regenerate with `python paper_figures/scripts/plot_selective_prediction_test_transfer.py`.\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output / 'selective_prediction_frozen_threshold_transfer.pdf'}")


if __name__ == "__main__":
    main()
