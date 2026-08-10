"""Build the manuscript figures from frozen GHXTox result tables.

This script deliberately reads only compact, versioned result artifacts.  It does
not rerun either expert and therefore cannot silently mix old and current model
checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
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
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

BLUE = "#2368a2"
ORANGE = "#e07a2d"
GREEN = "#2a9d6f"
PURPLE = "#7451a6"
GRAY = "#6b7280"
LIGHT = "#eef3f8"
THRESHOLD = 0.4702225653


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def box(ax, xy, wh, title, lines, color=BLUE, *, fontsize=8):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="white", edgecolor=color, linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top",
            weight="bold", color=color, fontsize=fontsize + 0.5)
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
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.985, "GHXTox current multidimensional inference path", ha="center",
            va="top", fontsize=12, weight="bold")

    box(ax, (0.03, 0.38), (0.13, 0.28), "Peptide", ["amino-acid", "sequence"], GRAY)
    box(ax, (0.22, 0.62), (0.27, 0.29), "1D sequence expert",
        ["ProtT5 residue embeddings", "LoRA task adaptation", "2-layer BiMamba",
         "3-seed + OOF Platt"], BLUE, fontsize=7)
    box(ax, (0.22, 0.16), (0.27, 0.34), "3D structural expert",
        ["single ESMFold test structure", "ESM2 semantics + pLDDT",
         "multi-scale EGNN", "chemical-site cross-attention"], GREEN, fontsize=7)
    box(ax, (0.57, 0.36), (0.18, 0.30), "Frozen fusion",
        ["0.72 × p1D", "+ 0.28 × p3D", "threshold", "0.470223"], PURPLE, fontsize=8)
    box(ax, (0.82, 0.39), (0.15, 0.24), "Prediction",
        ["toxic", "or non-toxic"], ORANGE, fontsize=9)

    arrow(ax, (0.16, 0.57), (0.22, 0.77))
    arrow(ax, (0.16, 0.47), (0.22, 0.34))
    arrow(ax, (0.49, 0.75), (0.57, 0.58), text="p1D")
    arrow(ax, (0.49, 0.33), (0.57, 0.44), text="p3D")
    arrow(ax, (0.75, 0.51), (0.82, 0.51))

    ax.text(0.355, 0.08,
            "Training-only structural augmentation: multiple conformers regularize the 3D expert;\n"
            "the public prediction interface consumes one canonical structure and base_logits.",
            ha="center", va="center", fontsize=7.5, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=LIGHT, edgecolor="none"))
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
        ["pLDDT-gated nodes", "three spatial edge scales", "equivariant EGNN"], GREEN, fontsize=7)
    box(ax, (0.29, 0.36), (0.23, 0.21), "Sequence context",
        ["ESM2 projection", "residue context"], BLUE, fontsize=7)
    box(ax, (0.29, 0.08), (0.23, 0.20), "Site graph",
        ["functional-group nodes", "site–residue edges"], ORANGE, fontsize=7)

    box(ax, (0.61, 0.48), (0.20, 0.29), "Interaction fusion",
        ["cross-attention", "confidence gates", "global + local pooling"], PURPLE, fontsize=7)
    box(ax, (0.86, 0.51), (0.11, 0.23), "3D logit", ["base_logits", "→ p3D"], GREEN, fontsize=7)

    for y in (0.795, 0.515, 0.235):
        arrow(ax, (0.21, y), (0.29, min(0.80, max(0.18, y))))
    arrow(ax, (0.52, 0.78), (0.61, 0.67))
    arrow(ax, (0.52, 0.46), (0.61, 0.61))
    arrow(ax, (0.52, 0.18), (0.61, 0.53))
    arrow(ax, (0.81, 0.62), (0.86, 0.62))

    ax.text(0.79, 0.20,
            "Optional training residual\nPepFlow / AF2 conformer descriptors\n"
            "regularize representation; excluded from\npublic single-structure output",
            ha="center", va="center", fontsize=7.5, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=LIGHT, edgecolor=GRAY,
                      linewidth=0.8, linestyle="--"))
    arrow(ax, (0.79, 0.31), (0.72, 0.48), color=GRAY, text="training only")
    save(fig, "fig2_expert_detail")


def discrimination_calibration_figure() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    for row, split in enumerate(("test1", "test2")):
        data = pd.read_csv(PRED / f"{split}_predictions.csv")
        y = data["label"].to_numpy()
        p = data["toxicity_probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="uniform")

        ax = axes[row, 0]
        ax.plot(fpr, tpr, color=BLUE, lw=1.8)
        ax.plot([0, 1], [0, 1], "--", color="#a0a0a0", lw=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
        ax.set_title(f"{'Test1' if row == 0 else 'Test2'} ROC")

        ax = axes[row, 1]
        ax.plot(recall, precision, color=ORANGE, lw=1.8)
        ax.axhline(float(y.mean()), ls="--", color="#a0a0a0", lw=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
        ax.set_title(f"{'Test1' if row == 0 else 'Test2'} precision–recall")

        ax = axes[row, 2]
        ax.plot([0, 1], [0, 1], "--", color="#a0a0a0", lw=0.8)
        ax.plot(mean, frac, "o-", color=GREEN, ms=3.5, lw=1.4)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed fraction")
        ax.set_title(f"{'Test1' if row == 0 else 'Test2'} reliability")
        for a in axes[row]:
            a.grid(alpha=0.18, lw=0.5)
    for i, ax in enumerate(axes.flat):
        ax.text(-0.18, 1.08, chr(65 + i), transform=ax.transAxes, weight="bold", fontsize=10)
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    save(fig, "fig3_discrimination_calibration")


def ablation_figure() -> None:
    metrics = ["BACC", "MCC", "AUROC", "AUPRC"]
    panels = [
        ("Test1 historical benchmark", [0.9179, 0.8458, 0.9699, 0.9426], [0.9208, 0.8464, 0.9711, 0.9474], "Ca 3D-v1", "pLDDT multi-scale 3D-v2"),
        ("Test2 historical benchmark", [0.8481, 0.6320, 0.9316, 0.6969], [0.8453, 0.6128, 0.9295, 0.7241], "Ca 3D-v1", "pLDDT multi-scale 3D-v2"),
        ("Paired training OOF", [0.9153, 0.8312, 0.9687, 0.9388], [0.9171, 0.8342, 0.9692, 0.9401], "without chemical residual", "with chemical residual"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    x = np.arange(len(metrics))
    for i, (title, a, b, la, lb) in enumerate(panels):
        ax = axes[i]
        ax.bar(x - 0.18, a, 0.36, color="#a9c7df", label=la)
        ax.bar(x + 0.18, b, 0.36, color=BLUE, label=lb)
        ax.set_xticks(x, metrics, rotation=28)
        ax.set_ylim(0.55, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2, lw=0.5)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.31), fontsize=6.4)
        ax.text(-0.12, 1.06, chr(65 + i), transform=ax.transAxes, weight="bold", fontsize=10)
    axes[0].set_ylabel("Metric value")
    fig.tight_layout(w_pad=0.7, rect=(0, 0.12, 1, 1))
    save(fig, "fig5_module_ablation")


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
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="none")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if letters[i, j]:
                color = "white" if abs(arr[i, j]) > 0.55 * vmax else "black"
                ax.text(j, i, letters[i, j], ha="center", va="center", fontsize=5.2,
                        family="monospace", color=color)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=6.5)
    ax.set_xlabel("Residue position (amino-acid letters shown in cells)")
    ax.set_title(f"{split.capitalize()} 1D residue occlusion: frozen representative cohort")
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
        ax.scatter(d.mean_plddt, d.global_gate, s=6, alpha=0.28, color=color, label=split)
    rho = samples[["mean_plddt", "global_gate"]].corr(method="spearman").iloc[0, 1]
    ax.set(xlabel="Mean predicted pLDDT", ylabel="Global learned gate", title=f"A  Sample-level confidence use (Spearman ρ={rho:.2f})")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    vals = samples["within_peptide_spearman_plddt_node_gate"].dropna().to_numpy()
    ax.hist(vals, bins=30, color=GREEN, alpha=0.82)
    ax.axvline(np.median(vals), color="black", ls="--", lw=1, label=f"median={np.median(vals):.2f}")
    ax.set(xlabel="Within-peptide Spearman ρ", ylabel="Peptide count", title="B  pLDDT–node-gate association")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    order = cats.groupby("site_category")["delta_probability_3d_baseline_minus_deleted"].median().abs().sort_values().index
    vals_by = [cats.loc[cats.site_category == k, "delta_probability_3d_baseline_minus_deleted"].to_numpy() for k in order]
    ax.boxplot(vals_by, vert=False, tick_labels=order, showfliers=False,
               boxprops=dict(color=PURPLE), medianprops=dict(color="black"), whiskerprops=dict(color=PURPLE), capprops=dict(color=PURPLE))
    rng = np.random.default_rng(7)
    for y, vals0 in enumerate(vals_by, 1):
        take = vals0 if len(vals0) <= 80 else rng.choice(vals0, 80, replace=False)
        ax.scatter(take, y + rng.normal(0, 0.045, len(take)), s=5, alpha=0.25, color=PURPLE)
    ax.axvline(0, color="#777", lw=0.8)
    ax.set(xlabel="Δp3D (baseline − category-deleted)", title="C  Chemical-category deletion")

    ax = axes[1, 1]
    top = sites.assign(abs_delta=lambda x: x.delta_probability_3d_baseline_minus_deleted.abs()).nlargest(11, "abs_delta").copy()
    top["display"] = top.apply(lambda r: f"{r.split_name}:{r.residue}{int(r.residue_position_1based)} {r.site_types}", axis=1)
    top = top.sort_values("delta_probability_3d_baseline_minus_deleted")
    colors = np.where(top.delta_probability_3d_baseline_minus_deleted >= 0, ORANGE, BLUE)
    ax.barh(top.display, top.delta_probability_3d_baseline_minus_deleted, color=colors)
    ax.axvline(0, color="#777", lw=0.8)
    ax.tick_params(axis="y", labelsize=6.2)
    ax.set(xlabel="Δp3D (baseline − site-deleted)", title="D  Largest individual-site sensitivities")

    for ax in axes.flat:
        ax.grid(alpha=0.16, lw=0.5)
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
        bars = ax.bar([0, 1, 2], vals, color=[BLUE, GREEN, PURPLE], width=0.65)
        ax.axhline(THRESHOLD, color=ORANGE, ls="--", lw=1.2, label=f"fusion threshold={THRESHOLD:.3f}")
        ax.set_xticks([0, 1, 2], ["p1D", "p3D", "fused"])
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{'A' if idx == 0 else 'B'}  {row.sample_id}: {row.expert_outcome_group}")
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=7.5)
        ax.text(0.03, 0.96, f"true label={int(row.label)}; fused prediction={int(row.prediction)}",
                transform=ax.transAxes, va="top", fontsize=7.2)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        ax.grid(axis="y", alpha=0.18, lw=0.5)
    axes[0].set_ylabel("Toxicity probability")
    fig.tight_layout(w_pad=1.5)
    save(fig, "expert_disagreement_cases")


def main() -> None:
    overview_figure()
    expert_detail_figure()
    discrimination_calibration_figure()
    ablation_figure()
    occlusion_figure("Test1")
    occlusion_figure("Test2")
    structural_sensitivity_figure()
    disagreement_cases_figure()
    print(f"Built manuscript figures in {OUT}")


if __name__ == "__main__":
    main()
