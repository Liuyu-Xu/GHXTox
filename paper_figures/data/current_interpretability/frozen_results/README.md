# Frozen GHXTox interpretability experiment

Status: **full**. Only a `full` run that passes all baseline checks is admissible as
manuscript evidence. A `smoke` run is implementation QA only.

Model: `GHXTox-Multidimensional-Fusion`. The deployed fusion is
`0.72 × p_1D + 0.28 × p_3D`, with frozen threshold `0.470222565313`.
The standalone outcome strata use the training-derived thresholds `0.392`
for the 1D expert and `0.677819` for the 3D expert.

## Representative cohort

Cases are selected before attribution. Within each dataset and expert-outcome group,
one toxic and one non-toxic peptide nearest the frozen fusion threshold are selected.
If only one class is available, the two peptides nearest the threshold are selected.
Ties are resolved by source index. Attribution values never enter case selection.

## 1D perturbation

Post-contextual fixed-representation occlusion: one residue-aligned task-adapted ProtT5 hidden vector is set to zero while all other hidden vectors, sequence length, BiMamba parameters, pooling, calibration, and thresholds are held fixed.

Per-seed effects are baseline logit minus occluded logit. For every state, the three
raw seed probabilities are averaged first and the frozen OOF Platt transform is then
applied. Display attribution uses signed within-peptide L1 normalization. These values quantify
frozen-model sensitivity and are not residue-level biological causal effects.

## 3D perturbation

The selected chemical-site mask entry and its type, coordinate, orientation, and orientation-mask tensors are set to zero; all non-target sequence, geometry, pLDDT, other chemical sites, conformer descriptors, weights, and thresholds are held fixed.

The complete frozen 3D forward pass, including the conformer residual when available,
is recomputed. The 1D expert is held fixed, so the induced fused-probability change is
exactly `0.28 × Δp_3D`. Predicted structures and chemical-site deletions are model
probes, not experimental structures or biochemical interventions.

## pLDDT--gate diagnostic

The analysis reports the sample-level association between mean pLDDT and the global
gate, plus within-peptide Spearman correlations between residue pLDDT and node gates.
Uncertainty is summarized with 2000 peptide-level bootstrap replicates
using seed 2026. This tests model use of confidence, not structure accuracy.
