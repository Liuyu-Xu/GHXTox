# Frozen-threshold selective-prediction transfer figure

Scientific question: does the already selected training-OOF abstention cutoff retain utility when applied unchanged to Test1 and Test2?

The figure uses the canonical single-structure GHXTox predictions and the immutable `margin_only` policy. The sole feature is normalized decision closeness. The uncertainty cutoff is 0.473617658741599, with `score <= cutoff` accepted. Test-specific quantiles, target coverages, feature selection, recalibration, and threshold changes are prohibited. The realized coverage is therefore an outcome. Labels are evaluation-only.

Intervals use 5,000 label-stratified fixed-mask paired bootstrap replicates (seed 20260825). The equal-count random control uses 5,000 fixed-cohort masks (seed 20260824), and one equal-count random mask inside each bootstrap replicate. No p-values or significance marks are used.

Artwork is 178 mm wide and supplied as editable SVG, vector PDF, and 600-dpi PNG. Exact values are in `exact_results_table.md` and `source_data/benchmark_transfer_metrics.csv`. Regenerate with `python paper_figures/scripts/plot_selective_prediction_test_transfer.py`.
