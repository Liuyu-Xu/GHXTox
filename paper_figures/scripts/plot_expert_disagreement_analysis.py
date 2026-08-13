#!/usr/bin/env python
"""Plot the frozen Test1/Test2 1D--3D expert disagreement analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


SPLITS = ("Test1", "Test2")
GROUPS = (
    "both_correct",
    "one_d_only_correct",
    "three_d_only_correct",
    "both_wrong",
)
GROUP_PUBLIC = {
    "both_correct": "Both experts correct",
    "one_d_only_correct": "1D expert only correct",
    "three_d_only_correct": "3D expert only correct",
    "both_wrong": "Both experts wrong",
}
GROUP_AXIS = {
    "both_correct": "Both\ncorrect",
    "one_d_only_correct": "1D-only\ncorrect",
    "three_d_only_correct": "3D-only\ncorrect",
    "both_wrong": "Both\nwrong",
}
ATTRIBUTES = (
    "peptide_length",
    "cysteine_count",
    "mean_plddt",
    "chemical_site_coverage",
    "conformer_available",
)
ATTRIBUTE_PUBLIC = {
    "peptide_length": "Peptide length",
    "cysteine_count": "Cys count",
    "mean_plddt": "Mean pLDDT",
    "chemical_site_coverage": "Chemical-site coverage",
    "conformer_available": "Conformer availability",
}


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _root()
    result = root / "runs" / "expert_disagreement_analysis_20260810_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", type=Path, default=result / "sample_assignments.csv")
    parser.add_argument("--counts-csv", type=Path, default=result / "group_counts.csv")
    parser.add_argument("--summaries-csv", type=Path, default=result / "group_attribute_summaries.csv")
    parser.add_argument("--rescue-csv", type=Path, default=result / "rescue_rates.csv")
    parser.add_argument("--contrasts-csv", type=Path, default=result / "primary_contrasts.csv")
    parser.add_argument("--summary-json", type=Path, default=result / "summary.json")
    parser.add_argument(
        "--protocol-json",
        type=Path,
        default=root / "configs" / "expert_disagreement_analysis_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "paper_figures" / "output" / "expert_disagreement_analysis",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Replace only the named files in an existing generated figure package.",
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
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _one(
    rows: list[dict[str, str]],
    *,
    split: str,
    group: str | None = None,
    attribute: str | None = None,
    label_stratum: str | None = None,
    endpoint: str | None = None,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("split_name") == split
        and (group is None or row.get("outcome_group") == group)
        and (attribute is None or row.get("attribute") == attribute)
        and (label_stratum is None or row.get("label_stratum") == label_stratum)
        and (endpoint is None or row.get("endpoint") == endpoint)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for split={split}, group={group}, attribute={attribute}, "
            f"label={label_stratum}, endpoint={endpoint}; found {len(matches)}"
        )
    return matches[0]


def _validate(
    samples: list[dict[str, str]],
    counts: list[dict[str, str]],
    summaries: list[dict[str, str]],
    rescue: list[dict[str, str]],
    contrasts: list[dict[str, str]],
    summary: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    if len(samples) != 1708 or len(counts) != 8 or len(summaries) != 40:
        raise ValueError("Unexpected sample/group table sizes")
    if len(rescue) != 12 or len(contrasts) != 10:
        raise ValueError("Unexpected rescue/contrast table sizes")
    if protocol.get("protocol_status") != "frozen_before_outcome_group_computation":
        raise ValueError("The disagreement protocol is not frozen")
    if float(protocol["expert_operating_points"]["one_d_threshold"]) != 0.392:
        raise ValueError("Unexpected 1D threshold")
    if float(protocol["expert_operating_points"]["three_d_threshold"]) != 0.677819:
        raise ValueError("Unexpected 3D threshold")
    if summary.get("decision") != "descriptive_analysis_complete_no_model_change":
        raise ValueError("Unexpected analysis decision")
    if any(int(row["canonical_prediction_uses_conformer"]) != 0 for row in samples):
        raise ValueError("A canonical prediction is marked as using conformer metadata")
    if any((int(row["conformer_available"]) == 1) != (int(row["peptide_length"]) <= 30) for row in samples):
        raise ValueError("Conformer availability is not the frozen length-defined metadata")
    if any(float(row["bh_q_value"]) <= 0.05 for row in contrasts):
        raise ValueError("The figure contract assumes no BH-significant primary contrast")
    expected_counts = {
        ("Test1", "both_correct"): 1032,
        ("Test1", "one_d_only_correct"): 29,
        ("Test1", "three_d_only_correct"): 24,
        ("Test1", "both_wrong"): 41,
        ("Test2", "both_correct"): 529,
        ("Test2", "one_d_only_correct"): 13,
        ("Test2", "three_d_only_correct"): 15,
        ("Test2", "both_wrong"): 25,
    }
    for (split, group), expected in expected_counts.items():
        if int(_one(counts, split=split, group=group)["sample_count"]) != expected:
            raise ValueError(f"Unexpected group count for {split}/{group}")


def _public_composition_rows(counts: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in SPLITS:
        for group in GROUPS:
            row = _one(counts, split=split, group=group)
            output.append(
                {
                    "split_name": split,
                    "outcome_group": GROUP_PUBLIC[group],
                    "sample_count": row["sample_count"],
                    "split_fraction": row["group_fraction"],
                    "toxic_count": row["toxic_count"],
                    "non_toxic_count": row["non_toxic_count"],
                    "toxic_fraction_within_group": row["toxic_fraction"],
                    "one_d_threshold": 0.392,
                    "three_d_threshold": 0.677819,
                }
            )
    return output


def _public_rescue_rows(rescue: list[dict[str, str]]) -> list[dict[str, Any]]:
    endpoint_public = {
        "three_d_rescue_among_one_d_errors": "3D rescues 1D errors",
        "one_d_rescue_among_three_d_errors": "1D rescues 3D errors",
    }
    label_public = {"all": "All", "non_toxic": "Non-toxic", "toxic": "Toxic"}
    output: list[dict[str, Any]] = []
    for row in rescue:
        output.append(
            {
                "split_name": row["split_name"],
                "label_stratum": label_public[row["label_stratum"]],
                "rescue_direction": endpoint_public[row["endpoint"]],
                "numerator": row["numerator"],
                "denominator": row["denominator"],
                "rate": row["rate"],
                "ci95_lower": row["wilson_ci_lower"],
                "ci95_upper": row["wilson_ci_upper"],
                "uncertainty": "95% Wilson interval",
                "one_d_threshold": 0.392,
                "three_d_threshold": 0.677819,
            }
        )
    return output


def _public_effect_rows(contrasts: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in contrasts:
        output.append(
            {
                "split_name": row["split_name"],
                "attribute": ATTRIBUTE_PUBLIC[row["attribute"]],
                "contrast": "3D expert only correct minus 1D expert only correct",
                "effect_type": (
                    "Availability proportion difference"
                    if row["attribute"] == "conformer_available"
                    else "Cliff's delta"
                ),
                "effect": row["primary_effect"],
                "ci95_lower": row["primary_effect_ci_lower"],
                "ci95_upper": row["primary_effect_ci_upper"],
                "three_d_only_n": row["three_d_only_n"],
                "one_d_only_n": row["one_d_only_n"],
                "permutation_p_value": row["permutation_p_value"],
                "bh_q_value": row["bh_q_value"],
                "bootstrap_replicates": row["bootstrap_replicates"],
                "permutation_replicates": row["permutation_replicates"],
                "uncertainty": "95% label-stratified bootstrap interval",
            }
        )
    return output


def _write_exact_table(path: Path, source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    text = text.replace("Frozen 1D--3D expert disagreement results", "Exact 1D--3D expert disagreement results")
    path.write_text(text, encoding="utf-8")


def _canonicalize_svg_clip_ids(path: Path) -> None:
    """Replace Matplotlib's process-specific clip IDs by ordered stable IDs."""
    text = path.read_text(encoding="utf-8")
    identifiers = re.findall(r'<clipPath id="([^"]+)">', text)
    for index, identifier in enumerate(identifiers):
        replacement = f"clip_path_{index:03d}"
        text = text.replace(f'url(#{identifier})', f'url(#{replacement})')
        text = text.replace(f'id="{identifier}"', f'id="{replacement}"')
    path.write_text(text, encoding="utf-8")


def _draw(
    output: Path,
    counts: list[dict[str, str]],
    rescue: list[dict[str, str]],
    contrasts: list[dict[str, str]],
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "ghxtox-expert-disagreement-v1",
        }
    )
    orange = "#E87522"
    blue = "#2878B5"
    accent = "#E5A11A"
    navy = "#174A7E"
    dark = "#3F4650"
    gray = "#697586"
    light_grid = "#D9DDE3"
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 3.55),
        gridspec_kw={"width_ratios": [1.10, 1.35, 1.08]},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.035, h_pad=0.03, wspace=0.07, hspace=0.03)

    # A: all four outcome groups, with true-class direction encoded by cell color.
    toxic_fraction = np.zeros((2, 4), dtype=np.float64)
    sample_count = np.zeros((2, 4), dtype=int)
    toxic_count = np.zeros((2, 4), dtype=int)
    for split_index, split in enumerate(SPLITS):
        for group_index, group in enumerate(GROUPS):
            row = _one(counts, split=split, group=group)
            sample_count[split_index, group_index] = int(row["sample_count"])
            toxic_count[split_index, group_index] = int(row["toxic_count"])
            toxic_fraction[split_index, group_index] = 100.0 * float(row["toxic_fraction"])
    class_map = LinearSegmentedColormap.from_list(
        "non_toxic_to_toxic", [(0.0, blue), (0.5, "#F7F7F7"), (1.0, orange)]
    )
    image = axes[0].imshow(toxic_fraction, cmap=class_map, vmin=0.0, vmax=100.0, aspect="auto")
    axes[0].set_xticks(np.arange(4), [GROUP_AXIS[group] for group in GROUPS], rotation=0)
    axes[0].set_yticks(np.arange(2), ("test set 1", "test set 2"))
    axes[0].tick_params(axis="x", labelsize=6.8, pad=3, length=0)
    axes[0].tick_params(axis="y", length=0)
    for row_index in range(2):
        for column_index in range(4):
            fraction = toxic_fraction[row_index, column_index]
            color = "white" if fraction < 22.0 or fraction > 78.0 else dark
            axes[0].text(
                column_index,
                row_index,
                f"n={sample_count[row_index, column_index]}\nTox\n{fraction:.0f}%",
                ha="center",
                va="center",
                fontsize=6.9,
                color=color,
                fontweight="semibold" if column_index == 2 else "normal",
            )
    colorbar = fig.colorbar(image, ax=axes[0], orientation="horizontal", fraction=0.075, pad=0.18)
    colorbar.set_label("Toxic fraction within group (%)", fontsize=7.5)
    colorbar.ax.tick_params(labelsize=7.0, width=0.7, length=2.5)
    colorbar.set_ticks([0, 50, 100])
    for spine in axes[0].spines.values():
        spine.set_linewidth(0.8)

    # B: reciprocal rescue rates by true class and benchmark.
    label_internal = ("all", "non_toxic", "toxic")
    label_axis = ("All", "Non-toxic", "Toxic")
    x_positions = np.asarray([0.0, 1.0, 2.0, 3.5, 4.5, 5.5])
    directions = (
        ("three_d_rescue_among_one_d_errors", "3D rescues 1D errors", blue, "o", -0.10),
        ("one_d_rescue_among_three_d_errors", "1D rescues 3D errors", orange, "s", 0.10),
    )
    for endpoint, public, color, marker, offset in directions:
        estimates: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for split in SPLITS:
            for label in label_internal:
                row = _one(rescue, split=split, label_stratum=label, endpoint=endpoint)
                estimate = 100.0 * float(row["rate"])
                estimates.append(estimate)
                lower.append(estimate - 100.0 * float(row["wilson_ci_lower"]))
                upper.append(100.0 * float(row["wilson_ci_upper"]) - estimate)
        axes[1].errorbar(
            x_positions + offset,
            estimates,
            yerr=np.asarray([lower, upper]),
            fmt=marker,
            linestyle="none",
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.1,
            markersize=4.8,
            elinewidth=0.9,
            capsize=2.3,
            capthick=0.9,
            label=public,
            zorder=3,
        )
    axes[1].axvline(2.75, color=light_grid, linewidth=0.8, zorder=0)
    axes[1].set_xticks(x_positions, [*label_axis, *label_axis], rotation=24, ha="right")
    axes[1].set_ylabel("Conditional rescue rate (%)")
    axes[1].set_ylim(0, 80)
    axes[1].set_yticks([0, 20, 40, 60, 80])
    axes[1].grid(axis="y", color=light_grid, linewidth=0.45, zorder=0)
    axes[1].text(1.0, 77.0, "test set 1", ha="center", va="top", fontsize=8.0, fontweight="bold")
    axes[1].text(4.5, 77.0, "test set 2", ha="center", va="top", fontsize=8.0, fontweight="bold")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), frameon=False, ncol=1)

    # C: primary attribute effects, one scale because binary Cliff's delta equals risk difference.
    y = np.arange(len(ATTRIBUTES), dtype=np.float64)
    split_style = {
        "Test1": (navy, "o", -0.12),
        "Test2": (orange, "D", 0.12),
    }
    for split in SPLITS:
        color, marker, offset = split_style[split]
        estimates: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for attribute in ATTRIBUTES:
            row = _one(contrasts, split=split, attribute=attribute)
            estimate = float(row["primary_effect"])
            estimates.append(estimate)
            lower.append(estimate - float(row["primary_effect_ci_lower"]))
            upper.append(float(row["primary_effect_ci_upper"]) - estimate)
        axes[2].errorbar(
            estimates,
            y + offset,
            xerr=np.asarray([lower, upper]),
            fmt=marker,
            linestyle="none",
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.1,
            markersize=4.8,
            elinewidth=0.9,
            capsize=2.3,
            capthick=0.9,
            label="test set 1" if split == "Test1" else "test set 2",
            zorder=3,
        )
    axes[2].axvline(0.0, color=gray, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    axes[2].set_yticks(y, [ATTRIBUTE_PUBLIC[attribute] for attribute in ATTRIBUTES])
    axes[2].invert_yaxis()
    axes[2].set_xlim(-1.0, 1.0)
    axes[2].set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axes[2].set_xlabel("Effect: 3D-only minus 1D-only")
    axes[2].grid(axis="x", color=light_grid, linewidth=0.45, zorder=0)
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False, ncol=2)

    for index, ax in enumerate(axes):
        if index > 0:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax.text(
            -0.18 if index != 2 else -0.38,
            1.07,
            chr(ord("A") + index),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    stem = output / "expert_disagreement_analysis"
    metadata = {
        "Title": "Frozen 1D-3D expert disagreement analysis",
        "Author": "GHXTox study",
        "Subject": "Complementary sequence and structural expert errors",
        "Keywords": "peptide toxicity, expert disagreement, structural rescue, sequence rescue",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(stem.with_suffix(".pdf"), metadata=metadata)
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
    _canonicalize_svg_clip_ids(stem.with_suffix(".svg"))


def main() -> None:
    args = parse_args()
    input_paths = (
        args.sample_csv,
        args.counts_csv,
        args.summaries_csv,
        args.rescue_csv,
        args.contrasts_csv,
        args.summary_json,
        args.protocol_json,
    )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(f"Refusing to overwrite non-empty figure package: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_csv(args.sample_csv)
    counts = _read_csv(args.counts_csv)
    summaries = _read_csv(args.summaries_csv)
    rescue = _read_csv(args.rescue_csv)
    contrasts = _read_csv(args.contrasts_csv)
    summary = _read_json(args.summary_json)
    protocol = _read_json(args.protocol_json)
    _validate(samples, counts, summaries, rescue, contrasts, summary, protocol)

    _write_csv(source_dir / "sample_assignments.csv", samples)
    _write_csv(source_dir / "outcome_group_composition.csv", _public_composition_rows(counts))
    _write_csv(source_dir / "reciprocal_rescue_rates.csv", _public_rescue_rows(rescue))
    _write_csv(source_dir / "attribute_effects.csv", _public_effect_rows(contrasts))
    _write_exact_table(output / "exact_results_table.md", args.counts_csv.parent / "results_table.md")
    _draw(output, counts, rescue, contrasts)

    roles = (
        "identity-bearing sample assignments and structural attributes",
        "four-group counts and true-label composition",
        "complete four-group attribute summaries",
        "reciprocal expert rescue rates with Wilson intervals",
        "3D-only versus 1D-only effects and corrected inference",
        "frozen result summary and no-model-change decision",
        "pre-outcome disagreement protocol",
    )
    _write_csv(
        output / "input_manifest.csv",
        [
            {
                "source": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "role": role,
            }
            for path, role in zip(input_paths, roles)
        ],
    )
    _write_csv(
        output / "internal_to_publication_names.csv",
        [
            {
                "internal_name": internal,
                "publication_name": public,
                "scope": "frozen 1D--3D expert disagreement",
            }
            for internal, public in GROUP_PUBLIC.items()
        ],
    )
    _write_csv(
        output / "panel_manifest.csv",
        [
            {
                "figure": "Supplementary expert disagreement analysis",
                "panel": "A",
                "data_file": "source_data/outcome_group_composition.csv",
                "script": "plot_expert_disagreement_analysis.py",
                "model": "Frozen task-adaptive sequence and confidence-aware structural experts",
                "sample_count": "Test1 n=1,126; Test2 n=582",
                "uncertainty": "none; exact counts and within-group toxic fraction",
                "conclusion": "Discordant decisions are sparse, and 3D-only corrections are predominantly non-toxic.",
            },
            {
                "figure": "Supplementary expert disagreement analysis",
                "panel": "B",
                "data_file": "source_data/reciprocal_rescue_rates.csv",
                "script": "plot_expert_disagreement_analysis.py",
                "model": "Frozen task-adaptive sequence and confidence-aware structural experts",
                "sample_count": "conditional error denominators shown in source table",
                "uncertainty": "95% Wilson interval",
                "conclusion": "The structural expert rescues approximately half of non-toxic sequence-expert errors but few toxic errors.",
            },
            {
                "figure": "Supplementary expert disagreement analysis",
                "panel": "C",
                "data_file": "source_data/attribute_effects.csv",
                "script": "plot_expert_disagreement_analysis.py",
                "model": "Frozen task-adaptive sequence and confidence-aware structural experts",
                "sample_count": "Test1: 3D-only n=24, 1D-only n=29; Test2: 3D-only n=15, 1D-only n=13",
                "uncertainty": "95% label-stratified bootstrap interval; 5,000 replicates",
                "conclusion": "Shorter length is the only directionally repeated tendency; no primary attribute remains significant after BH correction.",
            },
        ],
    )

    caption = (
        "The frozen structural expert provides sparse complementary corrections that are concentrated in non-toxic, shorter peptides rather than in uniformly high-confidence or chemically dense structures. "
        "(A) At the training-derived standalone thresholds (1D, 0.392; 3D, 0.677819), both experts are correct for 1,032/1,126 Test1 and 529/582 Test2 peptides. The structural expert alone is correct for 24 and 15 peptides, of which 19 (79.2%) and 14 (93.3%) are non-toxic; cell color denotes the toxic fraction within each outcome group. "
        "(B) Among errors made by the 1D expert, the 3D expert rescues 51.4% of non-toxic versus 17.9% of toxic Test1 peptides and 48.3% versus 9.1% on Test2. Reciprocal 1D rescue rates are shown for context. Error bars are 95% Wilson intervals. "
        "(C) The 3D-only-correct group is descriptively shorter than the 1D-only-correct group in both benchmarks (Cliff's delta -0.286 for Test1 and -0.523 for Test2), but none of the ten split-by-attribute primary comparisons remains significant after Benjamini--Hochberg correction. Horizontal intervals are 95% label-stratified bootstrap intervals from 5,000 replicates; p-values use 10,000 true-label-stratified permutations. For conformer availability, the plotted effect is a proportion difference, numerically equivalent to Cliff's delta for a binary variable. "
        "pLDDT is structure-prediction confidence rather than experimental structural accuracy. Chemical-site coverage is the fraction of residues with at least one model-derived site. Conformer availability is exactly determined by length in these benchmarks (available at <=30 residues) and was not used by the canonical single-structure predictions; its association is therefore not evidence of an ensemble-induced correction. Test1 and Test2 are established, non-independent project benchmarks and are analyzed separately."
    )
    (output / "caption.md").write_text(caption + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Frozen 1D--3D expert disagreement figure\n\n"
        "Scientific question: which peptide strata are rescued by the structural expert versus the sequence expert, and where do both experts fail?\n\n"
        "The figure uses the canonical single-structure Test1/Test2 probabilities. Standalone thresholds are fixed at 0.392 (1D) and 0.677819 (3D). No model, probability, calibration, fusion weight, or threshold is changed.\n\n"
        "Panel A shows exact group counts and within-group toxic fractions. Panel B shows reciprocal conditional rescue rates with 95% Wilson intervals. Panel C shows the 3D-only-correct minus 1D-only-correct effect for five prespecified attributes; intervals use 5,000 label-stratified bootstrap replicates, p-values use 10,000 true-label-stratified permutations, and BH correction covers all ten primary tests.\n\n"
        "Artwork is 178 mm wide and supplied as editable SVG, vector PDF, and 600-dpi PNG. Arial is requested with Liberation Sans/DejaVu Sans fallbacks. The manuscript palette uses orange for 1D evidence, blue for 3D evidence, dark blue for fusion, and neutral gray for reference elements.\n\n"
        "Regenerate with `python paper_figures/scripts/plot_expert_disagreement_analysis.py`. Exact values are in `exact_results_table.md`; publication-facing plot data are in `source_data/`. The identity-bearing sample table is included for audit.\n\n"
        "Interpretation is associative, not causal. Conformer availability is an exact length-defined metadata field and is not consumed by the canonical prediction.\n",
        encoding="utf-8",
    )

    checksum_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'expert_disagreement_analysis.pdf'}")


if __name__ == "__main__":
    main()
