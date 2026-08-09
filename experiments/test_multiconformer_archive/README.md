# Test-set multi-conformer archive

This directory is intentionally outside the canonical model package and public
prediction path. It preserves only the scripts and compact evidence needed to
audit the retired Test1/Test2 multi-conformer experiment.

The team decision dated 2026-08-09 is authoritative:

- multi-conformer information may remain in training-side checkpoints and
  ablation evidence;
- Test1, Test2, and new public predictions always use the single-structure
  `base_logits` from the retained full2795-trained checkpoint;
- `ghxtox-predict` does not accept conformer manifests or cache roots;
- test conformer caches, task chunks, server packages, and intermediate logs
  are not canonical model assets.

Machine-readable decision: `decision.json`.

Compact retained evidence:

- `results/source_aware_multiconformer_frozen_test_20260808_r4_corrected/summary.json`
- `results/source_independent_conformer_deployment_decision_20260808_v1/decision.json`
- `results/raw_view_spatial_embedding_five_fold_comparison_20260809_v1/decision.json`
- `results/original_lineage_source_independent_deployment_decision_20260809_v1/decision.json`
- `results/original_lineage_fusion_aware_residual_screen_decision_20260809_v1/decision.json`

Scripts under this directory are research-only and are not installed as console
entry points. They must not be imported by `ghxtox.multidimensional_default`.
