#!/usr/bin/env python
"""Create the publication-ready GHXTox selective risk--coverage figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PUBLIC_POLICY_NAME = "Frozen selective policy"
SELECTED_INTERNAL_PROFILE = "margin_only"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--risk-csv",
        type=Path,
        default=root / "runs" / "selective_prediction_oof_20260809_v1" / "risk_coverage.csv",
    )
    parser.add_argument(
        "--bootstrap-csv",
        type=Path,
        default=root
        / "runs"
        / "selective_prediction_oof_20260809_v1"
        / "bootstrap_intervals.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=root / "runs" / "selective_prediction_oof_20260809_v1" / "summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "paper_figures" / "output" / "selective_prediction_risk_coverage",
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty source table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    for path in (args.risk_csv, args.bootstrap_csv, args.summary_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    risk_rows = [
        row for row in _read_csv(args.risk_csv) if row["profile"] == SELECTED_INTERNAL_PROFILE
    ]
    risk_rows.sort(key=lambda row: float(row["coverage"]))
    if len(risk_rows) < 3 or abs(float(risk_rows[-1]["coverage"]) - 1.0) > 1e-9:
        raise ValueError("Selected-policy risk curve is incomplete.")
    public_risk_rows = [
        {
            "policy": PUBLIC_POLICY_NAME,
            "coverage": row["coverage"],
            "accepted_error_rate": row["accepted_error_rate"],
            "accepted_mcc": row["accepted_mcc"],
            "toxic_class_coverage": row["toxic_class_coverage"],
            "non_toxic_class_coverage": row["non_toxic_class_coverage"],
            "accepted_n": row["accepted_n"],
            "rejected_n": row["rejected_n"],
        }
        for row in risk_rows
    ]
    _write_csv(source_dir / "risk_coverage.csv", public_risk_rows)

    bootstrap_rows = [
        row
        for row in _read_csv(args.bootstrap_csv)
        if row["profile"] == SELECTED_INTERNAL_PROFILE
        and row["metric"] == "accepted_error_rate"
        and float(row["coverage_target"]) in (0.8, 0.9, 0.95)
    ]
    public_bootstrap_rows = [
        {
            "policy": PUBLIC_POLICY_NAME,
            "metric": "Accepted error rate",
            "coverage": row["coverage_target"],
            "bootstrap_replicates": row["bootstrap_replicates"],
            "ci_lower": row["ci_lower"],
            "ci_median": row["ci_median"],
            "ci_upper": row["ci_upper"],
        }
        for row in bootstrap_rows
    ]
    _write_csv(source_dir / "bootstrap_error_intervals.csv", public_bootstrap_rows)

    coverage = np.asarray([float(row["coverage"]) for row in risk_rows]) * 100.0
    risk = np.asarray([float(row["accepted_error_rate"]) for row in risk_rows]) * 100.0
    full_risk = float(risk[-1])
    point_90 = min(risk_rows, key=lambda row: abs(float(row["coverage"]) - 0.9))
    coverage_90 = float(point_90["coverage"]) * 100.0
    risk_90 = float(point_90["accepted_error_rate"]) * 100.0
    ci_90 = next(row for row in bootstrap_rows if abs(float(row["coverage_target"]) - 0.9) < 1e-12)
    lower_90 = float(ci_90["ci_lower"]) * 100.0
    upper_90 = float(ci_90["ci_upper"]) * 100.0

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    blue = "#0072B2"
    amber = "#E69F00"
    gray = "#666666"
    ax.axhline(
        full_risk,
        color=gray,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        label=f"Full-coverage error ({full_risk:.2f}%)",
        zorder=1,
    )
    ax.fill_between(coverage, risk, full_risk, where=risk <= full_risk, color=blue, alpha=0.09)
    ax.plot(
        coverage,
        risk,
        color=blue,
        linewidth=2.0,
        marker="o",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label=PUBLIC_POLICY_NAME,
        zorder=3,
    )
    ax.errorbar(
        [coverage_90],
        [risk_90],
        yerr=[[risk_90 - lower_90], [upper_90 - risk_90]],
        fmt="o",
        color=amber,
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=7,
        capsize=3,
        linewidth=1.4,
        zorder=5,
    )
    ax.axvline(coverage_90, color=amber, linewidth=0.9, alpha=0.65, linestyle=(0, (2, 2)))
    ax.annotate(
        f"Prespecified operating point\nCoverage: {coverage_90:.1f}%\nAccepted error: {risk_90:.2f}%",
        xy=(coverage_90, risk_90),
        xytext=(73.5, 4.05),
        arrowprops={"arrowstyle": "-", "color": amber, "linewidth": 0.9},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": amber, "linewidth": 0.9},
        ha="left",
        va="center",
        fontsize=8,
    )
    ax.set_xlabel("Coverage of accepted predictions (%)")
    ax.set_ylabel("Selective risk (accepted error, %)")
    ax.set_xlim(49.0, 101.0)
    ax.set_ylim(0.0, max(5.8, full_risk + 0.65))
    ax.set_xticks(np.arange(50, 101, 10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    stem = output / "selective_prediction_risk_coverage"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    input_rows = [
        {
            "source": str(path.resolve()),
            "sha256": _sha256(path),
            "role": role,
        }
        for path, role in (
            (args.risk_csv, "frozen cross-fitted OOF risk-coverage metrics"),
            (args.bootstrap_csv, "frozen class-stratified bootstrap intervals"),
            (args.summary_json, "frozen policy-selection audit"),
        )
    ]
    _write_csv(output / "input_manifest.csv", input_rows)
    _write_csv(
        output / "internal_to_publication_names.csv",
        [
            {
                "internal_name": SELECTED_INTERNAL_PROFILE,
                "publication_name": PUBLIC_POLICY_NAME,
                "scope": "training-OOF selective prediction",
            }
        ],
    )
    caption = (
        "Selective risk as a function of prediction coverage on the 6,387 training out-of-fold "
        "predictions. Samples are accepted in ascending order of uncertainty from the frozen "
        "cross-fitted selective policy; the underlying toxicity probabilities and binary decision "
        "threshold remain unchanged. The amber point marks the prespecified 90% coverage operating "
        f"point (accepted error, {risk_90:.2f}%; 95% class-stratified bootstrap CI, "
        f"{lower_90:.2f}-{upper_90:.2f}%; 5,000 replicates). The dashed line denotes the "
        f"full-coverage error ({full_risk:.2f}%). This analysis quantifies selective reliability on "
        "training OOF predictions and is not an external-validation result."
    )
    (output / "caption.md").write_text(caption + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Selective prediction risk-coverage figure\n\n"
        "This package is generated from the frozen training-OOF selective-prediction analysis. "
        "It contains editable SVG/PDF artwork, a 600-dpi PNG, publication-facing source tables, "
        "input provenance, naming translation, checksums, and the complete caption. Test1 and "
        "Test2 labels were not used. Regenerate with "
        "`python paper_figures/scripts/plot_selective_prediction_risk_coverage.py`.\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"Wrote {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
