"""Build the manuscript figures from frozen GHXTox result tables.

This script deliberately reads only compact, versioned result artifacts.  It does
not rerun either expert and therefore cannot silently mix old and current model
checkpoints.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "paper_figures/data/current_interpretability/frozen_results"
OUT = ROOT / "paper_figures/output/manuscript_figures"
PRED = ROOT / "runs/default_multidimensional_fusion_single_structure"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Blue-orange publication palette.  Orange consistently denotes the 1D
# sequence branch; blue denotes structural evidence; dark blue denotes fusion.
ORANGE = "#E87522"
ORANGE_LIGHT = "#FBE3D2"
BLUE = "#2878B5"
BLUE_LIGHT = "#D9EAF7"
NAVY = "#174A7E"
FUSION = NAVY
GRAY = "#6B7280"
BASELINE = "#A8B0BA"
REFERENCE = "#667085"
LIGHT = "#F4F7FA"
GRID = "#DDE3EA"

with (ROOT / "configs/default_multidimensional_fusion.json").open(encoding="utf-8") as stream:
    FUSION_CONFIG = json.load(stream)
WEIGHT_1D = float(FUSION_CONFIG["fusion"]["weight_1d"])
WEIGHT_3D = float(FUSION_CONFIG["fusion"]["weight_3d"])
THRESHOLD = float(FUSION_CONFIG["fusion"]["threshold"])


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def box(ax, xy, wh, title, lines, color=BLUE, *, fontsize=8, title_fontsize=None):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="white", edgecolor=color, linewidth=1.5,
    )
    ax.add_patch(patch)
    if title_fontsize is None:
        title_fontsize = fontsize + 0.5
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top",
            weight="bold", color=color, fontsize=title_fontsize)
    ax.text(x + w / 2, y + h / 2 - 0.015, "\n".join(lines), ha="center",
            va="center", fontsize=fontsize, linespacing=1.3)
    return patch


def arrow(ax, start, end, *, color=GRAY, text=None, rad=0.0):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                          linewidth=1.2, color=color,
                          connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(arr)
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.018,
                text, ha="center", va="bottom", fontsize=7, color=color)


def overview_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.02, 0.37), (0.13, 0.26), "Peptide", ["amino-acid", "sequence"], GRAY)
    box(ax, (0.21, 0.59), (0.28, 0.31), "Task-adaptive 1D expert",
        ["ProtT5 + LoRA", "bidirectional Mamba", "multi-seed calibration"], ORANGE, fontsize=8)
    box(ax, (0.21, 0.12), (0.28, 0.34), "Sequence-enhanced 3D expert",
        ["predicted structure + pLDDT", "residue geometry + chemical sites",
         "confidence-gated interaction"], BLUE, fontsize=8, title_fontsize=7.2)
    box(ax, (0.57, 0.36), (0.18, 0.28), "Probability fusion",
        ["OOF-frozen weights", "and decision threshold"], FUSION, fontsize=8)
    box(ax, (0.82, 0.39), (0.15, 0.22), "Prediction",
        ["toxic", "or non-toxic"], NAVY, fontsize=9)

    arrow(ax, (0.15, 0.56), (0.21, 0.74))
    arrow(ax, (0.15, 0.45), (0.21, 0.30))
    arrow(ax, (0.49, 0.74), (0.57, 0.57), text=r"$p_{1D}$")
    arrow(ax, (0.49, 0.30), (0.57, 0.44), text=r"$p_{3D}$")
    arrow(ax, (0.75, 0.51), (0.82, 0.51))
    save(fig, "fig1_overview")


def expert_detail_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.985, "pLDDT-aware multi-scale structural expert", ha="center",
            va="top", fontsize=12, weight="bold")

    box(ax, (0.03, 0.70), (0.18, 0.19), "Input", ["ESMFold coordinates", "residue pLDDT"], GRAY, fontsize=7)
    box(ax, (0.03, 0.42), (0.18, 0.19), "Semantics", ["ESM2 embeddings", "sequence-aligned"], BLUE, fontsize=7)
    box(ax, (0.03, 0.14), (0.18, 0.19), "Sites", ["type + position", "orientation + mask"], ORANGE, fontsize=7)

    box(ax, (0.29, 0.66), (0.23, 0.25), "Multi-scale geometry",
        ["pLDDT-gated nodes", "three spatial edge scales", "equivariant EGNN"], BLUE, fontsize=7)
    box(ax, (0.29, 0.36), (0.23, 0.21), "Sequence context",
        ["ESM2 projection", "residue context"], BLUE, fontsize=7)
    box(ax, (0.29, 0.08), (0.23, 0.20), "Site graph",
        ["functional-group nodes", "site-residue edges"], ORANGE, fontsize=7)

    box(ax, (0.61, 0.48), (0.20, 0.29), "Interaction fusion",
        ["cross-attention", "confidence gates", "global + local pooling"], NAVY, fontsize=7)
    box(ax, (0.86, 0.51), (0.11, 0.23), "3D logit", ["base logits", "to p3D"], BLUE, fontsize=7)

    for y in (0.795, 0.515, 0.235):
        arrow(ax, (0.21, y), (0.29, min(0.80, max(0.18, y))))
    arrow(ax, (0.52, 0.78), (0.61, 0.67))
    arrow(ax, (0.52, 0.46), (0.61, 0.61))
    arrow(ax, (0.52, 0.18), (0.61, 0.53))
    arrow(ax, (0.81, 0.62), (0.86, 0.62))

    ax.text(0.79, 0.20,
            "Optional ablation residual\nPepFlow / AF2 conformer descriptors\n"
            "trained on frozen backbone; disabled in\ncanonical single-structure output",
            ha="center", va="center", fontsize=7.5, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=LIGHT, edgecolor=GRAY,
                      linewidth=0.8, linestyle="--"))
    arrow(ax, (0.79, 0.31), (0.72, 0.48), color=GRAY, text="ablation only")
    save(fig, "fig2_expert_detail")


def discrimination_calibration_figure() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.15))
    series = (
        ("p_1d", "Task-adaptive 1D expert", ORANGE, "--", "s", 1.7),
        ("p_3d", "3D structural expert", BLUE, "-.", "^", 1.7),
        ("toxicity_probability", "Probability fusion", NAVY, "-", "o", 2.2),
    )
    for row, split in enumerate(("test1", "test2")):
        data = pd.read_csv(PRED / f"{split}_predictions.csv")
        y = data["label"].to_numpy()
        display_split = "test set 1" if row == 0 else "test set 2"

        ax = axes[row, 0]
        ax.plot([0, 1], [0, 1], "--", color=REFERENCE, lw=0.9)
        for column, label, color, linestyle, _marker, linewidth in series:
            p = data[column].to_numpy()
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=color, ls=linestyle, lw=linewidth,
                    label=label, zorder=3 if column == "toxicity_probability" else 2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
        ax.set_title(f"{display_split} ROC", color=NAVY, weight="semibold")

        ax = axes[row, 1]
        prevalence = float(y.mean())
        ax.axhline(prevalence, ls="--", color=REFERENCE, lw=0.9)
        for column, label, color, linestyle, _marker, linewidth in series:
            p = data[column].to_numpy()
            precision, recall, _ = precision_recall_curve(y, p)
            ax.plot(recall, precision, color=color, ls=linestyle, lw=linewidth,
                    label=label, zorder=3 if column == "toxicity_probability" else 2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
        ax.set_title(f"{display_split} precision-recall", color=NAVY,
                     weight="semibold")

        ax = axes[row, 2]
        ax.plot([0, 1], [0, 1], "--", color=REFERENCE, lw=0.9)
        for column, label, color, linestyle, marker, linewidth in series:
            p = data[column].to_numpy()
            frac, mean = calibration_curve(y, p, n_bins=10, strategy="uniform")
            ax.plot(mean, frac, color=color, ls=linestyle, marker=marker,
                    ms=4.0, lw=linewidth, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=1.0, label=label,
                    zorder=3 if column == "toxicity_probability" else 2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed fraction")
        ax.set_title(f"{display_split} reliability", color=NAVY,
                     weight="semibold")
        for a in axes[row]:
            a.grid(color=GRID, alpha=0.65, lw=0.5)
    for i, ax in enumerate(axes.flat):
        ax.text(-0.18, 1.08, chr(65 + i), transform=ax.transAxes,
                weight="bold", fontsize=12, color=NAVY)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.005), handlelength=2.8, columnspacing=1.5)
    fig.tight_layout(w_pad=1.2, h_pad=1.4, rect=(0, 0, 1, 0.95))
    save(fig, "fig3_discrimination_calibration")


def ablation_figure() -> None:
    metrics = ["BACC", "MCC", "AUROC", "AUPRC"]
    source = pd.read_csv(ROOT / "paper_figures/data/manuscript/fig5_module_ablation.csv")
    panels = []
    for panel in ("A", "B", "C"):
        frame = source[source.panel == panel]
        variants = (
            frame[["variant", "display_order"]]
            .drop_duplicates()
            .sort_values("display_order")["variant"]
            .tolist()
        )
        values = [
            frame[frame.variant == variant].set_index("metric").loc[metrics, "value"].tolist()
            for variant in variants
        ]
        title = {"A": "test set 1", "B": "test set 2", "C": "Training OOF"}[panel]
        panels.append((title, values[0], values[1], *variants))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), sharey=True)
    x = np.arange(len(metrics))
    for i, (title, a, b, la, lb) in enumerate(panels):
        ax = axes[i]
        ax.bar(x - 0.18, a, 0.36, color=BASELINE, label="Comparator")
        ax.bar(x + 0.18, b, 0.36, color=BLUE, label="Current")
        ax.set_xticks(x, metrics, rotation=28)
        ax.set_ylim(0.55, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, alpha=0.7, lw=0.5)
        ax.text(-0.12, 1.06, chr(65 + i), transform=ax.transAxes, weight="bold", fontsize=12)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.01), columnspacing=1.4)
    fig.tight_layout(w_pad=0.7, rect=(0, 0.09, 1, 1))
    save(fig, "fig5_module_ablation")


def selective_prediction_figure() -> None:
    """Combine frozen OOF and benchmark-transfer tables into unique panels A--E."""
    oof = pd.read_csv(
        ROOT / "paper_figures/output/selective_prediction_formal_current_default_v2/"
        "source_data/five_coverage_metrics.csv"
    )
    transfer = pd.read_csv(
        ROOT / "paper_figures/output/"
        "selective_prediction_frozen_threshold_transfer_current_default_v2/"
        "source_data/benchmark_transfer_metrics.csv"
    )
    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.92], hspace=0.48, wspace=0.9)
    axes = [
        fig.add_subplot(grid[0, :3]),
        fig.add_subplot(grid[0, 3:]),
        fig.add_subplot(grid[1, :2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[1, 4:]),
    ]

    oof_styles = {
        "Frozen uncertainty ranking": (FUSION, "o", "Frozen policy"),
        "Random rejection": (BASELINE, "s", "Random rejection"),
    }
    for ax, metric, scale, ylabel in (
        (axes[0], "Accepted error", 100.0, "Accepted error (%)"),
        (axes[1], "MCC", 1.0, "MCC"),
    ):
        for method, (color, marker, public) in oof_styles.items():
            frame = oof[(oof.method == method) & (oof.metric == metric)].copy()
            frame = frame.sort_values("coverage_realized")
            x = 100 * frame.coverage_realized.to_numpy()
            y = scale * frame.estimate.to_numpy()
            lo = scale * frame.ci95_lower.to_numpy()
            hi = scale * frame.ci95_upper.to_numpy()
            ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
            ax.plot(x, y, color=color, marker=marker, ms=3.8, lw=1.6,
                    markerfacecolor="white", label=public)
        ax.axvline(90, color=REFERENCE, ls=(0, (3, 2)), lw=1.0)
        ax.set(xlabel="Coverage (%)", ylabel=ylabel)
        ax.grid(color=GRID, alpha=0.7, lw=0.5)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")

    splits = ["Test1", "Test2"]
    split_labels = ["test set 1", "test set 2"]
    coverage = []
    for split in splits:
        row = transfer[(transfer.split_name == split) &
                       (transfer.method == "Frozen-threshold accepted set") &
                       (transfer.metric == "Coverage")].iloc[0]
        coverage.append(100 * float(row.estimate))
    bars = axes[2].bar(split_labels, coverage, color=FUSION, width=0.58)
    axes[2].axhline(90, color=REFERENCE, ls=(0, (3, 2)), lw=1.0)
    axes[2].set(ylabel="Realized coverage (%)", ylim=(0, 105))
    for bar, value in zip(bars, coverage):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value - 4,
                     f"{value:.1f}%", ha="center", va="top", color="white",
                     fontsize=8, weight="bold")

    transfer_styles = {
        "Full binary predictor": (NAVY, "D", "Full predictor"),
        "Frozen-threshold accepted set": (FUSION, "o", "Frozen policy"),
        "Equal-count random rejection": (BASELINE, "s", "Random rejection"),
    }
    for ax, metric, scale, ylabel, ylim in (
        (axes[3], "Accepted error", 100.0, "Accepted error (%)", (0, 8)),
        (axes[4], "MCC", 1.0, "MCC", (0.5, 1.0)),
    ):
        for offset, (method, (color, marker, public)) in zip(
            (-0.18, 0.0, 0.18), transfer_styles.items()
        ):
            frame = transfer[(transfer.method == method) & (transfer.metric == metric)]
            frame = frame.set_index("split_name").loc[splits]
            y = scale * frame.estimate.to_numpy()
            lo = y - scale * frame.ci95_lower.to_numpy()
            hi = scale * frame.ci95_upper.to_numpy() - y
            ax.errorbar(np.arange(2) + offset, y, yerr=np.asarray([lo, hi]),
                        fmt=marker, color=color, markerfacecolor="white",
                        ms=4.5, lw=1.0, capsize=2.2, label=public)
        ax.set(xticks=np.arange(2), xticklabels=split_labels, ylabel=ylabel, ylim=ylim)
        ax.grid(axis="y", color=GRID, alpha=0.7, lw=0.5)
    handles, labels = axes[3].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), ncol=3)

    for i, ax in enumerate(axes):
        ax.text(-0.13, 1.14, chr(65 + i), transform=ax.transAxes,
                weight="bold", fontsize=12, va="top")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.17,
                        hspace=0.56, wspace=1.15)
    save(fig, "selective_prediction_combined")


def occlusion_figure(split: str) -> None:
    rows = pd.read_csv(FROZEN / "sequence_residue_occlusion.csv")
    cohort = pd.read_csv(FROZEN / "representative_cohort.csv")
    rows = rows[rows.split_name.str.lower() == split.lower()].copy()
    cohort = cohort[cohort.split_name.str.lower() == split.lower()].copy()
    ids = cohort.sort_values(["expert_outcome_group", "label", "selection_rank_within_group"])["sample_id"].tolist()
    ordered = [rows[rows.sample_id == sid].sort_values("residue_position_1based") for sid in ids]
    max_len = max(len(x) for x in ordered)
    arr = np.full((len(ordered), max_len), np.nan)
    letters = np.full((len(ordered), max_len), "", dtype=object)
    labels = []
    for i, frame in enumerate(ordered):
        arr[i, : len(frame)] = frame["signed_l1_normalized_attribution"].to_numpy()
        letters[i, : len(frame)] = frame["residue"].astype(str).to_numpy()
        first = frame.iloc[0]
        labels.append(f"{first.sample_id} | {first.expert_outcome_group} | y={int(first.label)}")
    vmax = np.nanmax(np.abs(arr))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "blue_white_orange", [BLUE, "#FFFFFF", ORANGE]
    )
    im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="none")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if letters[i, j]:
                color = "white" if abs(arr[i, j]) > 0.55 * vmax else "black"
                ax.text(j, i, letters[i, j], ha="center", va="center", fontsize=5.2,
                        family="monospace", color=color)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=6.5)
    ax.set_xlabel("Residue position (amino-acid letters shown in cells)")
    split_label = "test set 1" if split.lower() == "test1" else "test set 2"
    ax.set_title(f"{split_label} 1D residue occlusion: frozen representative cohort")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Signed within-peptide L1-normalized attribution")
    fig.tight_layout()
    save(fig, f"{split.lower()}_residue_occlusion")


def structural_sensitivity_figure() -> None:
    samples = pd.read_csv(FROZEN / "plddt_gate_samples.csv")
    residues = pd.read_csv(FROZEN / "plddt_gate_residues.csv")
    cats = pd.read_csv(FROZEN / "structure_category_deletion.csv")
    sites = pd.read_csv(FROZEN / "structure_site_deletion.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))

    ax = axes[0, 0]
    for split, color in (("Test1", BLUE), ("Test2", ORANGE)):
        d = samples[samples.split_name == split]
        split_label = "test set 1" if split == "Test1" else "test set 2"
        ax.scatter(d.mean_plddt, d.global_gate, s=6, alpha=0.28, color=color, label=split_label)
    rho = samples[["mean_plddt", "global_gate"]].corr(method="spearman").iloc[0, 1]
    ax.set(xlabel="Mean predicted pLDDT", ylabel="Global structural gate", title=f"Confidence use ($\\rho$={rho:.2f})")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    vals = samples["within_peptide_spearman_plddt_node_gate"].dropna().to_numpy()
    ax.hist(vals, bins=30, color=BLUE, alpha=0.82)
    ax.axvline(np.median(vals), color="black", ls="--", lw=1, label=f"median={np.median(vals):.2f}")
    ax.set(xlabel="Within-peptide Spearman $\\rho$", ylabel="Peptide count", title="Residue-level association")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    order = cats.groupby("site_category")["delta_probability_3d_baseline_minus_deleted"].median().abs().sort_values().index
    vals_by = [cats.loc[cats.site_category == k, "delta_probability_3d_baseline_minus_deleted"].to_numpy() for k in order]
    ax.boxplot(vals_by, vert=False, tick_labels=order, showfliers=False,
               boxprops=dict(color=NAVY), medianprops=dict(color=ORANGE), whiskerprops=dict(color=NAVY), capprops=dict(color=NAVY))
    rng = np.random.default_rng(7)
    for y, vals0 in enumerate(vals_by, 1):
        take = vals0 if len(vals0) <= 80 else rng.choice(vals0, 80, replace=False)
        ax.scatter(take, y + rng.normal(0, 0.045, len(take)), s=5, alpha=0.25, color=BLUE)
    ax.axvline(0, color=REFERENCE, lw=0.8)
    ax.set(xlabel=r"$\Delta p_{3D}$ (baseline - deleted)", title="Chemical-category deletion")

    ax = axes[1, 1]
    top = sites.assign(abs_delta=lambda x: x.delta_probability_3d_baseline_minus_deleted.abs()).nlargest(11, "abs_delta").copy()
    top["display"] = top.apply(
        lambda r: f"{'test set 1' if r.split_name == 'Test1' else 'test set 2'} "
        f"{r.residue}{int(r.residue_position_1based)} | "
        + str(r.site_types).replace("hydrophobic", "hydroph.").replace("conformational", "conf."),
        axis=1,
    )
    top = top.sort_values("delta_probability_3d_baseline_minus_deleted")
    colors = np.where(top.delta_probability_3d_baseline_minus_deleted >= 0, ORANGE, BLUE)
    ax.barh(top.display, top.delta_probability_3d_baseline_minus_deleted, color=colors)
    ax.axvline(0, color=REFERENCE, lw=0.8)
    ax.tick_params(axis="y", labelsize=6.2)
    ax.set(xlabel=r"$\Delta p_{3D}$ (baseline - deleted)", title="Largest site sensitivities")

    for i, ax in enumerate(axes.flat):
        ax.grid(color=GRID, alpha=0.65, lw=0.5)
        ax.text(-0.16, 1.08, chr(65 + i), transform=ax.transAxes,
                weight="bold", fontsize=12, va="top")
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    save(fig, "structural_sensitivity")


def disagreement_cases_figure() -> None:
    cohort = pd.read_csv(FROZEN / "representative_cohort.csv")
    chosen = cohort[cohort.sample_id.isin(["sequence_23_23", "sequence_34_34"])].copy()
    chosen = chosen.set_index("sample_id").loc[["sequence_23_23", "sequence_34_34"]].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for idx, row in chosen.iterrows():
        ax = axes[idx]
        vals = [row.p_1d, row.p_3d, row.toxicity_probability]
        bars = ax.bar([0, 1, 2], vals, color=[ORANGE, BLUE, FUSION], width=0.65)
        ax.axhline(THRESHOLD, color=REFERENCE, ls="--", lw=1.2, label="Decision threshold")
        ax.set_xticks([0, 1, 2], ["1D expert", "3D expert", "Fusion"])
        ax.set_ylim(0, 1.08)
        ax.set_title("3D evidence rescues 1D" if idx == 0 else "3D evidence overrides 1D")
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=7.5)
        ax.text(-0.12, 1.08, chr(65 + idx), transform=ax.transAxes,
                weight="bold", fontsize=12, va="top")
        ax.grid(axis="y", color=GRID, alpha=0.7, lw=0.5)
    axes[0].set_ylabel("Toxicity probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.10, 1, 1))
    save(fig, "expert_disagreement_cases")


def main() -> None:
    discrimination_calibration_figure()
    ablation_figure()
    selective_prediction_figure()
    occlusion_figure("Test1")
    occlusion_figure("Test2")
    structural_sensitivity_figure()
    disagreement_cases_figure()
    print(f"Built manuscript figures in {OUT}")


if __name__ == "__main__":
    main()
