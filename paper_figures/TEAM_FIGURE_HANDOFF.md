# GHXTox paper-figure handoff

Status date: 2026-08-08
Purpose: teammate coordination only. This document defines what may be drawn now, what may be drafted without final export, and what must remain paused until the underlying evidence is frozen.

## 1. Scope decision

### Draw now and prepare as final figure assets

| Manuscript figure | Publication-facing name | Status | Immediate output |
|---|---|---|---|
| Figure 6 | Test1 residue-level sensitivity of the sequence expert | **DRAW NOW** | `test1_residue_occlusion.{svg,pdf,png}` |
| Figure 7 | Test2 residue-level sensitivity of the sequence expert | **DRAW NOW** | `test2_residue_occlusion.{svg,pdf,png}` |
| Figure 8 | Structural confidence and chemical-site sensitivity | **DRAW NOW, RESTRICTED PANELS** | `structural_sensitivity.{svg,pdf,png}` |

The frozen interpretability package has passed its baseline checks and contains checksums. These three figures must be generated from the existing files; the model does not need to be rerun.

### Stable core may be drafted, but must not be treated as final artwork

| Manuscript figure | Draft scope | Status | Draft output |
|---|---|---|---|
| Figure 1 | Dataset, task-adaptive sequence expert, single-structure 3D core, frozen probability fusion and decision | **CORE DRAFT ONLY** | `draft_fig1_overview_core.svg` |
| Figure 2 | ESM2-enhanced residue semantics, pLDDT-aware hybrid geometric graph, chemical-site branch, cross-attention/gating and structural classifier | **CORE DRAFT ONLY** | `draft_fig2_structural_core.svg` |

Do not draw, label, or reserve a data-bearing multi-conformer panel in these drafts. Keep the source fully editable because the manuscript captions cannot be finalized until the structural checkpoint decision is resolved. Do not insert these drafts into `v1.tex` yet.

### Paused: do not draw yet

| Item | Status | Reason for pause | Unblocking requirement |
|---|---|---|---|
| Figure 3: ROC, PR and reliability | **PAUSED** | The final structural probability vector is unresolved; current ECE values are not consistently available for the retained multi-conformer outputs. | Freeze the final structural checkpoint and regenerate one canonical Test1/Test2 probability package, including reliability-bin data. |
| Figure 4: expert complementarity and decision changes | **PAUSED** | Expert outcomes, probability shifts and corrected/regressed cases depend on the final structural vector. | Freeze final `p_3D`, fused probabilities and fixed-threshold paired outcomes. |
| Figure 5: module-level ablation | **PAUSED** | Existing rows mix training OOF, historical benchmark and current benchmark protocols; no common fold-level plotting table currently supports an A--F comparison. | Produce protocol-separated frozen plot tables with fold/sample uncertainty and define the panels before drawing. |
| Figure 9: expert agreement and representative cases | **PAUSED** | Uses final 3D/fusion outcomes and cache-covered cases; part of the planned content is conformer-adjacent. | Resolve the final structural checkpoint and redesign the figure without unresolved conformer claims. |
| Test1/Test2 multi-conformer ablation tables or plots | **PAUSED** | The current pilot-trained residual was applied outside its generator training domain and shows a one-sided positive probability shift. | Recover/evaluate the source-aware frozen checkpoint or demote the result to an exploratory near-null ablation. |
| Any conformer overlay, RMSF, contact-occupancy, conformer-source, availability, residual-shift or conformer decision-transition panel | **PAUSED** | Multi-conformer evidence and source matching are unresolved. | A matched-source, training-frozen analysis with an approved reporting decision. |
| Current-versus-previous paired-bootstrap plot/Table 7 extension | **PAUSED** | The archived bootstrap corresponds to the zero-conformer vector, not the unresolved retained extension. | Generate aligned current/prior vectors and the prespecified 5,000-resample analysis after the model decision. |

No placeholder values, estimated curves, or values copied from reference papers may be used while an item is paused.

## 2. Figure 6 production brief: Test1 residue occlusion

### Scientific question

Which residue positions produce directionally reproducible changes in the frozen 1D sequence-expert probability for the eight prespecified Test1 cases?

### Frozen inputs

- `data/current_interpretability/frozen_results/sequence_residue_occlusion.csv`
- `data/current_interpretability/frozen_results/sequence_sample_summary.csv`
- `data/current_interpretability/frozen_results/representative_cohort.csv`
- `data/current_interpretability/frozen_results/summary.json`
- `data/current_interpretability/frozen_results/checksums.sha256`

Filter strictly to `split_name == "Test1"`. Preserve the cohort order defined by `selection_rank_within_group`; do not reorder cases by attribution magnitude.

### Layout

- Double-column width, 178 mm; suggested height 115--145 mm.
- Eight horizontal sequence heat strips, one case per row.
- Each cell corresponds to one residue and displays the amino-acid letter when legible at final size.
- Shared diverging scale centered at zero, using `signed_l1_normalized_attribution`.
- Positive attribution: vermillion; negative attribution: blue; zero: white/light neutral.
- Row annotations: true label, publication-facing expert-outcome group, baseline `p_1D`, strongest residue/position and mean seed sign agreement.
- Add a compact legend defining the sign: positive means that occlusion lowers the calibrated toxicity probability.

Do not add biological motif or toxic-residue claims. Do not show only the largest cases; all eight prespecified rows are required.

### Caption draft

> Residue occlusion identifies heterogeneous but directionally reproducible sequence sensitivities in the frozen 1D expert on Test1. Each row shows one prespecified peptide, and each cell reports the signed within-peptide L1-normalized change in calibrated 1D probability after zeroing the corresponding post-contextual residue vector. Cases retain the prespecified outcome-group order rather than being selected by attribution magnitude. Seed sign agreement summarizes directional consistency across the three frozen checkpoints. The values quantify model sensitivity and do not identify experimentally validated toxic residues.

## 3. Figure 7 production brief: Test2 residue occlusion

### Scientific question

Does the same frozen 1D expert show comparable residue-level sensitivity patterns in the eight prespecified Test2 cases?

### Frozen inputs

Use the same five files as Figure 6 and filter strictly to `split_name == "Test2"`.

### Layout

Match Figure 6 exactly:

- same 178-mm width and row height;
- same case-order rule;
- same diverging limits and colorbar;
- same annotations and typography;
- same sign definition.

Test2 contains exact sequence matches to a subset of Test1. Similar maps must therefore be described as a reproducibility check, not independent external validation.

### Caption draft

> Test2 residue-occlusion profiles provide a matched reproducibility view of the frozen 1D expert. The eight prespecified cases are displayed with the same ordering, normalization and signed color scale used for Test1. Positive values indicate that residue-vector occlusion lowers the calibrated 1D probability, whereas negative values indicate an increase. Because Test2 contains exact sequence matches to a subset of Test1, concordant maps are interpreted as repeated-sequence consistency rather than independent external validation.

## 4. Figure 8 production brief: structural confidence and chemical-site sensitivity

### Scientific question

How is predicted-structure confidence associated with learned gating, and how sensitive is the frozen structural expert to controlled deletion of modeled chemical sites?

### Frozen inputs

- `data/current_interpretability/frozen_results/plddt_gate_samples.csv`
- `data/current_interpretability/frozen_results/plddt_gate_residues.csv`
- `data/current_interpretability/frozen_results/plddt_gate_summary.json`
- `data/current_interpretability/frozen_results/structure_category_deletion.csv`
- `data/current_interpretability/frozen_results/structure_site_deletion.csv`
- `data/current_interpretability/frozen_results/summary.json`
- `data/current_interpretability/frozen_results/checksums.sha256`

### Required 2 x 2 layout

**(A) Peptide-level confidence gating**

- x-axis: `mean_plddt`; y-axis: `global_gate`.
- Show Test1 and Test2 with different marker shapes in a shared coordinate system.
- Annotate the dataset-specific Spearman coefficient and 95% peptide-bootstrap interval from the frozen summary.
- State `n=1,126` and `n=582`; pLDDT is predictor confidence, not experimental structure accuracy.

**(B) Residue-scale gate association**

- Distribution of `within_peptide_spearman_plddt_node_gate` by dataset.
- Use violin/box/jitter or box plus lightly jittered observations; show the mean and the frozen peptide-bootstrap interval.
- Use identical axis limits for Test1 and Test2.

**(C) Chemical-category deletion effects**

- Plot signed `delta_probability_3d_baseline_minus_deleted` for the eight modeled site categories.
- Show individual case-level points rather than only a bar mean; optionally overlay median and interquartile range.
- A second symbol or adjacent small axis may show absolute effect, but signed and absolute values must not be conflated.
- These are 16-case descriptive sensitivity results, not population effect sizes.
- If a peptide has no site in a category, treat that peptide--category cell as missing; do not replace it with a zero-effect observation.

**(D) Largest individual-site deletion effects**

- Select a fixed, stated number of sites by absolute `delta_probability_3d_baseline_minus_deleted` (recommended 10--12, including both signs).
- Horizontal signed lollipop/point plot, labeled by split, residue, 1-based position and `site_types`.
- Show `delta_fused_probability_holding_1d_fixed` as a smaller secondary marker only if the legend remains clear.
- Selection and sorting must be performed in code and documented in the figure README.

### Excluded fields and analyses

Do not use or display:

- `conformer_available`;
- `conformer_logit_delta`;
- `base_probability_3d_without_conformer_residual`;
- generator/source labels;
- cache-covered versus fallback comparisons;
- any claim about flexibility, molecular dynamics, toxicophores or causal biochemical sites.

### Caption draft

> Predicted-structure confidence is strongly associated with peptide-level gating, whereas residue-scale associations and chemical-site deletion effects are heterogeneous. (A) Mean pLDDT and the learned global structural gate for Test1 and Test2, with dataset-specific Spearman coefficients and 95% peptide-bootstrap intervals. (B) Distribution of within-peptide correlations between residue pLDDT and node gates. (C) Signed changes in structural-expert probability after category-level chemical-site deletion in the prespecified 16-peptide cohort. (D) The largest probability-increasing and probability-decreasing individual-site deletions. pLDDT is a confidence estimate from structure prediction, and deletion effects quantify frozen-model sensitivity rather than experimental structure accuracy or validated toxicophores.

## 5. Stable architecture drafts

### Figure 1 core draft

Draw left to right:

1. peptide sequence and predicted single structure;
2. task-adaptive sequence expert: ProtT5, LoRA adaptation, bidirectional Mamba, pooling, three-seed average and OOF Platt calibration;
3. confidence-aware structural expert: ESM2-enhanced residue semantics, pLDDT-aware hybrid geometric graph, full-atom chemical-site branch and structural classifier;
4. frozen probability fusion, `0.72 p_1D + 0.28 p_3D`;
5. fixed decision threshold `0.4702225653`.

Do not add an independent 2D expert. Chemical-site modeling belongs inside the structural expert and uses the structural green palette.

### Figure 2 core draft

Recommended panels:

- (A) residue-semantic inputs and two-layer contextual encoder;
- (B) sequential plus spatial-neighborhood graph and three pLDDT-aware EGNN layers;
- (C) full-atom chemical-site construction, typed interaction messages and zero-initialized chemical residual;
- (D) confidence-modulated sequence--structure interaction, global features, pooling and structural output.

Do not include conformer statistics, conformer graphs, ensemble aggregation or availability masks. Store only editable draft source until the manuscript model description is reconciled.

Architecture provenance:

- `../configs/lora_mamba_1d.json`
- `../runs/3d_v2_default/config.json`
- `../configs/default_multidimensional_fusion.json` (fusion constants only)
- `../src/ghxtox/lora_mamba_1d.py`
- `../src/ghxtox/models/dual_modal.py`
- `../src/ghxtox/models/chemical_sites.py`
- `../src/ghxtox/multidimensional_default.py`

## 6. Tables

The exact-value tables already present in `v1.tex` remain LaTeX tables. Do not redraw them as figures. In particular:

- do not draw the two multi-conformer ablation tables;
- do not convert the point-estimate/paired-bootstrap table into a forest plot yet;
- do not merge training OOF rows with Test1/Test2 rows in one ranked bar chart;
- do not mark significance until the pairing, resample count and interval are frozen.

## 7. Common production rules

- Default final width: 178 mm for the three active figures; architecture drafts also use 178 mm.
- Font: Arial/Helvetica/Source Sans 3; final text at least 7.5 pt; panel labels 10--11 pt bold.
- Sequence blue `#0072B2`; structure green `#009E73`; fusion purple `#CC79A7`; toxic vermillion `#D55E00`; non-toxic blue `#0072B2`; neutral/fallback gray `#9AA0A6`.
- Use marker shape or line style in addition to color.
- No gradients, shadows, pseudo-3D bars, rainbow/jet maps or rasterized text.
- Final active figures require SVG, vector PDF and 600-dpi PNG; keep an editable source.
- Each final figure directory must contain `input_manifest.csv`, `panel_manifest.csv`, `caption.md`, `README.md` and `checksums.sha256`.
- Artwork and captions use publication-facing names only. Internal paths/checkpoints may appear in manifests and README files.
- Never join sample-level tables by row order; use `split_name` plus `source_index` and verify labels.

## 8. Recommended teammate order

1. Figure 6 and Figure 7 together, so their scales and typography are identical.
2. Restricted Figure 8.
3. Editable stable-core drafts for Figures 1 and 2.
4. Stop and return the editable files for scientific review. Do not begin paused figures without an updated status file.

This status decision overrides broader generic figure plans elsewhere in the repository for the present handoff. An item moves out of `PAUSED` only after `FIGURE_STATUS.csv` and this document are revised together.
