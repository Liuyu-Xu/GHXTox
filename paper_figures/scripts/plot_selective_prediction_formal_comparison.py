#!/usr/bin/env python
"""Render the formal GHXTox selective-versus-random coverage experiment."""

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


INTERNAL_SELECTIVE = "frozen_uncertainty_ranking"
INTERNAL_RANDOM = "random_rejection"
PUBLIC_NAMES = {
    INTERNAL_SELECTIVE: "Frozen uncertainty ranking",
    INTERNAL_RANDOM: "Random rejection",
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


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _root()
    experiment = root / "runs" / "selective_prediction_formal_current_default_20260810_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=experiment / "five_coverage_comparison.csv",
    )
    parser.add_argument(
        "--intervals-csv",
        type=Path,
        default=experiment / "method_bootstrap_intervals.csv",
    )
    parser.add_argument(
        "--contrasts-csv",
        type=Path,
        default=experiment / "paired_bootstrap_contrasts.csv",
    )
    parser.add_argument("--summary-json", type=Path, default=experiment / "summary.json")
    parser.add_argument(
        "--protocol-json",
        type=Path,
        default=root / "configs" / "selective_prediction_formal_current_default_v2.json",
    )
    parser.add_argument(
        "--selected-policy-json",
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
        / "selective_prediction_formal_current_default_v2",
    )
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
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty source table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _exact_row(
    rows: list[dict[str, str]], method: str, coverage: float, metric: str | None = None
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["method"] == method
        and abs(float(row["coverage_target"]) - coverage) < 1e-12
        and (metric is None or row.get("metric") == metric)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for method={method}, coverage={coverage}, metric={metric}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _validate_inputs(
    comparison: list[dict[str, str]],
    intervals: list[dict[str, str]],
    contrasts: list[dict[str, str]],
    summary: dict[str, Any],
    protocol: dict[str, Any],
    selected_policy: dict[str, Any],
) -> list[float]:
    expected_coverages = [1.0, 0.95, 0.9, 0.85, 0.8]
    if summary.get("sample_count") != 6387:
        raise ValueError("The frozen formal cohort must contain 6,387 training-OOF rows.")
    policy = summary.get("selected_uncertainty_policy", {})
    if policy.get("required_profile") != "margin_only":
        raise ValueError("The formal figure requires the frozen margin-only policy.")
    if policy.get("required_feature_names") != ["decision_closeness"]:
        raise ValueError("The frozen uncertainty score must use decision_closeness only.")
    if selected_policy.get("profile") != "margin_only":
        raise ValueError("The policy-selection record does not select margin_only.")
    if selected_policy.get("feature_names") != ["decision_closeness"]:
        raise ValueError("The policy-selection record contains an unexpected feature set.")
    if selected_policy.get("test_labels_used") is not False:
        raise ValueError("The immutable policy record does not satisfy the no-Test-label boundary.")
    scope = summary.get("scope", {})
    prohibited = (
        "allow_policy_refit",
        "allow_feature_reselection",
        "allow_binary_model_change",
        "allow_fusion_or_threshold_change",
        "allow_test1_or_test2_labels",
        "external_validation_claimed",
    )
    if any(scope.get(key) is not False for key in prohibited):
        raise ValueError("The formal no-refit/no-Test scope is not intact.")
    if protocol.get("operating_points", {}).get("coverage_targets") != expected_coverages:
        raise ValueError("The protocol coverage grid differs from the frozen five points.")
    for method in PUBLIC_NAMES:
        for coverage in expected_coverages:
            _exact_row(comparison, method, coverage)
            for metric in METRICS:
                interval = _exact_row(intervals, method, coverage, metric)
                if int(interval["bootstrap_replicates"]) != 5000:
                    raise ValueError("All method intervals must use 5,000 replicates.")
    for coverage in expected_coverages:
        for metric in METRICS:
            matches = [
                row
                for row in contrasts
                if abs(float(row["coverage_target"]) - coverage) < 1e-12
                and row["metric"] == metric
            ]
            if len(matches) != 1 or int(matches[0]["bootstrap_replicates"]) != 5000:
                raise ValueError("Each paired contrast must have one 5,000-replicate row.")
    return expected_coverages


def _public_metric_rows(
    comparison: list[dict[str, str]],
    intervals: list[dict[str, str]],
    coverages: list[float],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    threshold = summary["binary_threshold"]
    for coverage in coverages:
        for method, public_name in PUBLIC_NAMES.items():
            point = _exact_row(comparison, method, coverage)
            for metric in METRICS:
                interval = _exact_row(intervals, method, coverage, metric)
                rows.append(
                    {
                        "split": "Training out-of-fold predictions",
                        "model_and_policy": "Frozen GHXTox predictor with margin-only selective policy",
                        "method": public_name,
                        "coverage_target": coverage,
                        "coverage_realized": point["coverage"],
                        "accepted_n": point["accepted_n"],
                        "rejected_n": point["rejected_n"],
                        "binary_threshold": threshold,
                        "metric": METRIC_LABELS[metric],
                        "estimate": point[metric],
                        "ci95_lower": interval["ci95_lower"],
                        "ci95_upper": interval["ci95_upper"],
                        "uncertainty": interval["uncertainty_type"],
                        "replicates": interval["bootstrap_replicates"],
                    }
                )
    return rows


def _public_contrast_rows(
    contrasts: list[dict[str, str]], coverages: list[float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for coverage in coverages:
        for metric in METRICS:
            row = next(
                item
                for item in contrasts
                if abs(float(item["coverage_target"]) - coverage) < 1e-12
                and item["metric"] == metric
            )
            rows.append(
                {
                    "split": "Training out-of-fold predictions",
                    "coverage_target": coverage,
                    "accepted_n": row["accepted_n"],
                    "rejected_n": row["rejected_n"],
                    "metric": METRIC_LABELS[metric],
                    "positive_improvement_definition": row[
                        "positive_improvement_definition"
                    ].replace("random", "Random rejection").replace(
                        "selective", "Frozen uncertainty ranking"
                    ),
                    "point_estimate": row["point_estimate"],
                    "ci95_lower": row["ci95_lower"],
                    "ci95_upper": row["ci95_upper"],
                    "paired_bootstrap_replicates": row["bootstrap_replicates"],
                }
            )
    return rows


def _write_markdown_table(
    path: Path, comparison: list[dict[str, str]], coverages: list[float]
) -> None:
    headers = [
        "Method",
        "Target coverage",
        "Realized coverage",
        "Accepted",
        "Error",
        "MCC",
        "BACC",
        "SN",
        "SP",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for coverage in coverages:
        for method, public_name in PUBLIC_NAMES.items():
            row = _exact_row(comparison, method, coverage)
            lines.append(
                "| "
                + " | ".join(
                    [
                        public_name,
                        f"{100 * coverage:.0f}%",
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


def _series(
    comparison: list[dict[str, str]],
    intervals: list[dict[str, str]],
    method: str,
    metric: str,
    coverages: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(coverages)
    x = np.asarray(
        [100 * float(_exact_row(comparison, method, coverage)["coverage"]) for coverage in ordered]
    )
    y = np.asarray(
        [float(_exact_row(comparison, method, coverage)[metric]) for coverage in ordered]
    )
    lo = np.asarray(
        [float(_exact_row(intervals, method, coverage, metric)["ci95_lower"]) for coverage in ordered]
    )
    hi = np.asarray(
        [float(_exact_row(intervals, method, coverage, metric)["ci95_upper"]) for coverage in ordered]
    )
    return x, y, lo, hi


def _draw_figure(
    output: Path,
    comparison: list[dict[str, str]],
    intervals: list[dict[str, str]],
    coverages: list[float],
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "ghxtox-selective-formal-v1",
        }
    )
    blue = "#0072B2"
    gray = "#697586"
    amber = "#E69F00"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), constrained_layout=True)

    panels = (
        (axes[0], "accepted_error_rate", "Accepted error (%)", 100.0),
        (axes[1], "accepted_mcc", "MCC on accepted predictions", 1.0),
    )
    legend_handles = []
    legend_labels = []
    for panel_index, (ax, metric, ylabel, scale) in enumerate(panels):
        for method, color, linestyle, marker, linewidth in (
            (INTERNAL_SELECTIVE, blue, "-", "o", 2.0),
            (INTERNAL_RANDOM, gray, (0, (4, 3)), "s", 1.45),
        ):
            x, y, lo, hi = _series(comparison, intervals, method, metric, coverages)
            y, lo, hi = y * scale, lo * scale, hi * scale
            ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=1)
            (line,) = ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                zorder=3,
            )
            if panel_index == 0:
                legend_handles.append(line)
                legend_labels.append(PUBLIC_NAMES[method])

        sel90 = _exact_row(comparison, INTERNAL_SELECTIVE, 0.9)
        rnd90 = _exact_row(comparison, INTERNAL_RANDOM, 0.9)
        x90 = 100 * float(sel90["coverage"])
        y_sel = scale * float(sel90[metric])
        y_rnd = scale * float(rnd90[metric])
        ax.axvline(x90, color=amber, linewidth=0.9, linestyle=(0, (2, 2)), alpha=0.8)
        ax.scatter(
            [x90], [y_sel], s=38, facecolor="white", edgecolor=amber, linewidth=1.4, zorder=5
        )
        if metric == "accepted_error_rate":
            label = f"90%: {y_sel:.2f}% vs {y_rnd:.2f}%"
            xytext = (83.0, 2.1)
        else:
            label = f"90%: {y_sel:.3f} vs {y_rnd:.3f}"
            xytext = (82.2, 0.953)
        ax.annotate(
            label,
            xy=(x90, y_sel),
            xytext=xytext,
            arrowprops={"arrowstyle": "-", "color": amber, "linewidth": 0.8},
            color="#4A4A4A",
            fontsize=7.4,
            ha="left",
            va="center",
        )
        ax.set_xlabel("Coverage of accepted predictions (%)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(78.7, 101.3)
        ax.set_xticks([80, 85, 90, 95, 100])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.16,
            1.04,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="top",
        )

    axes[0].set_ylim(1.35, 6.15)
    axes[1].set_ylim(0.85, 0.965)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.035),
        ncol=2,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.7,
    )
    stem = output / "selective_prediction_random_rejection_comparison"
    metadata = {
        "Title": "Selective prediction compared with equal-coverage random rejection",
        "Author": "GHXTox study",
        "Subject": "Training out-of-fold reliability analysis",
        "Keywords": "selective prediction, risk-coverage, random rejection, GHXTox",
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
        args.comparison_csv,
        args.intervals_csv,
        args.contrasts_csv,
        args.summary_json,
        args.protocol_json,
        args.selected_policy_json,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty figure package: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    comparison = _read_csv(args.comparison_csv)
    intervals = _read_csv(args.intervals_csv)
    contrasts = _read_csv(args.contrasts_csv)
    summary = _read_json(args.summary_json)
    protocol = _read_json(args.protocol_json)
    selected_policy = _read_json(args.selected_policy_json)
    coverages = _validate_inputs(
        comparison, intervals, contrasts, summary, protocol, selected_policy
    )

    public_metrics = _public_metric_rows(comparison, intervals, coverages, summary)
    public_contrasts = _public_contrast_rows(contrasts, coverages)
    _write_csv(source_dir / "five_coverage_metrics.csv", public_metrics)
    _write_csv(source_dir / "paired_improvements.csv", public_contrasts)
    _write_markdown_table(output / "five_coverage_table.md", comparison, coverages)
    _draw_figure(output, comparison, intervals, coverages)

    input_roles = (
        "frozen five-coverage point estimates and equal-coverage random means",
        "frozen 95% method intervals from the paired bootstrap design",
        "frozen positive-oriented paired bootstrap contrasts",
        "formal training-OOF experiment summary and scope",
        "outcome-independent formal protocol",
        "immutable policy-selection record",
    )
    _write_csv(
        output / "input_manifest.csv",
        [
            {"source": str(path.resolve()), "sha256": _sha256(path), "role": role}
            for path, role in zip(inputs, input_roles)
        ],
    )
    _write_csv(
        output / "internal_to_publication_names.csv",
        [
            {
                "internal_name": internal,
                "publication_name": public,
                "scope": "training-OOF selective-prediction comparison",
            }
            for internal, public in PUBLIC_NAMES.items()
        ],
    )
    _write_csv(
        output / "panel_manifest.csv",
        [
            {
                "panel": "A",
                "scientific_question": "Does frozen uncertainty ranking reduce accepted-set error more than equal-coverage random rejection?",
                "metric": "Accepted error (%)",
                "display": "Five-point risk-coverage curves with 95% intervals",
            },
            {
                "panel": "B",
                "scientific_question": "Is the risk reduction accompanied by improved balanced classification performance?",
                "metric": "MCC on accepted predictions",
                "display": "Five-point performance-coverage curves with 95% intervals",
            },
        ],
    )

    primary = next(
        row
        for row in public_contrasts
        if row["coverage_target"] == 0.9 and row["metric"] == "Accepted error"
    )
    sel90 = _exact_row(comparison, INTERNAL_SELECTIVE, 0.9)
    rnd90 = _exact_row(comparison, INTERNAL_RANDOM, 0.9)
    direct_score = selected_policy.get("model", {}).get("type") == "direct_feature_score"
    score_caption = (
        "the direct normalized proximity of the GHXTox fusion probability to the fixed "
        "decision threshold"
        if direct_score
        else "the frozen class-balanced logistic error-ranking score based only on normalized "
        "proximity of the GHXTox fusion probability to the fixed decision threshold"
    )
    score_readme = (
        "the direct normalized decision-closeness value is used as the ranking score"
        if direct_score
        else "the stored class-balanced logistic output is used only as a ranking score"
    )
    caption = (
        "Frozen uncertainty ranking identifies difficult GHXTox predictions more effectively than "
        "equal-coverage random rejection on 6,387 training out-of-fold predictions. (A) Accepted-set "
        "error and (B) Matthews correlation coefficient at target coverages of 100%, 95%, 90%, "
        f"85%, and 80%. The uncertainty score is {score_caption} "
        "(tau = 0.4702225653); it is not a calibrated error probability. Random "
        "rejection uses 5,000 uniformly sampled nested rankings with exactly the same accepted count "
        "at each coverage. Shaded regions denote 95% intervals from 5,000 class-stratified paired "
        "bootstrap replicates; each random-bootstrap evaluation uses one nested random ranking. At "
        f"the prespecified 90% operating point, accepted error was {100 * float(sel90['accepted_error_rate']):.2f}% "
        f"for uncertainty ranking and {100 * float(rnd90['accepted_error_rate']):.2f}% for random rejection; "
        f"the random-minus-selective difference was {100 * float(primary['point_estimate']):.2f} percentage "
        f"points (95% paired interval, {100 * float(primary['ci95_lower']):.2f}--"
        f"{100 * float(primary['ci95_upper']):.2f}). The toxicity probabilities, fusion rule, and "
        "binary threshold were unchanged. Test1 and Test2 labels were not used for policy selection "
        "or evaluation; these results quantify internal training-OOF reliability and do not establish "
        "external generalization."
    )
    (output / "caption.md").write_text(caption + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Formal selective-prediction comparison\n\n"
        "Scientific question: does the frozen uncertainty ranking identify difficult predictions "
        "better than rejecting the same number of samples at random?\n\n"
        "The package uses 6,387 training out-of-fold predictions. The uncertainty policy is fixed "
        "to the previously selected `margin_only` profile. Its only input is "
        "`decision_closeness = 1 - abs(p_GHXTox - tau) / max(tau, 1 - tau)`, with "
        f"`tau = 0.47022256531284556`; {score_readme}. No Test1/Test2 label, policy refit, "
        "feature reselection, model retraining, "
        "fusion change, or threshold change is involved.\n\n"
        "Random rejection comprises 5,000 uniform nested rankings (seed 20260820), with the exact "
        "accepted count used by the selective policy at each target coverage. Intervals and paired "
        "contrasts use 5,000 class-stratified bootstrap replicates (seed 20260821); each replicate "
        "compares the two methods on the same resampled cohort, and the random method receives one "
        "nested random ranking. The primary contrast is random-minus-selective accepted error at "
        "90% coverage. No p-values or significance marks are reported.\n\n"
        "Artwork is 178 mm wide and is provided as editable SVG, vector PDF, and 600-dpi PNG. "
        "Frozen uncertainty ranking is shown as a blue solid line with circular markers; random "
        "rejection is a neutral-gray dashed line with square markers; amber marks the prespecified "
        "90% operating point. Exact values are in `five_coverage_table.md` and "
        "`source_data/five_coverage_metrics.csv`. Regenerate with "
        "`python paper_figures/scripts/plot_selective_prediction_formal_comparison.py`.\n",
        encoding="utf-8",
    )

    checksum_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output / 'selective_prediction_random_rejection_comparison.pdf'}")


if __name__ == "__main__":
    main()
