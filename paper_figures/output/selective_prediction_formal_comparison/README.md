# Formal selective-prediction comparison

Scientific question: does the frozen uncertainty ranking identify difficult predictions better than rejecting the same number of samples at random?

The package uses 6,387 training out-of-fold predictions. The uncertainty policy is fixed to the previously selected `margin_only` profile. Its only input is `decision_closeness = 1 - abs(p_GHXTox - tau) / max(tau, 1 - tau)`, with `tau = 0.47022256531284556`; the stored class-balanced logistic output is used only as a ranking score. No Test1/Test2 label, policy refit, feature reselection, model retraining, fusion change, or threshold change is involved.

Random rejection comprises 5,000 uniform nested rankings (seed 20260820), with the exact accepted count used by the selective policy at each target coverage. Intervals and paired contrasts use 5,000 class-stratified bootstrap replicates (seed 20260821); each replicate compares the two methods on the same resampled cohort, and the random method receives one nested random ranking. The primary contrast is random-minus-selective accepted error at 90% coverage. No p-values or significance marks are reported.

Artwork is 178 mm wide and is provided as editable SVG, vector PDF, and 600-dpi PNG. Frozen uncertainty ranking is shown as a blue solid line with circular markers; random rejection is a neutral-gray dashed line with square markers; amber marks the prespecified 90% operating point. Exact values are in `five_coverage_table.md` and `source_data/five_coverage_metrics.csv`. Regenerate with `python paper_figures/scripts/plot_selective_prediction_formal_comparison.py`.
