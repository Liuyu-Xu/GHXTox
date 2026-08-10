# Frozen 1D--3D expert disagreement figure

Scientific question: which peptide strata are rescued by the structural expert versus the sequence expert, and where do both experts fail?

The figure uses the canonical single-structure Test1/Test2 probabilities. Standalone thresholds are fixed at 0.392 (1D) and 0.677819 (3D). No model, probability, calibration, fusion weight, or threshold is changed.

Panel A shows exact group counts and within-group toxic fractions. Panel B shows reciprocal conditional rescue rates with 95% Wilson intervals. Panel C shows the 3D-only-correct minus 1D-only-correct effect for five prespecified attributes; intervals use 5,000 label-stratified bootstrap replicates, p-values use 10,000 true-label-stratified permutations, and BH correction covers all ten primary tests.

Artwork is 178 mm wide and supplied as editable SVG, vector PDF, and 600-dpi PNG. Arial is requested with Liberation Sans/DejaVu Sans fallbacks. The palette follows the project convention: sequence blue, structure green, toxic vermillion, and neutral dark gray.

Regenerate with `python paper_figures/scripts/plot_expert_disagreement_analysis.py`. Exact values are in `exact_results_table.md`; publication-facing plot data are in `source_data/`. The identity-bearing sample table is included for audit.

Interpretation is associative, not causal. Conformer availability is an exact length-defined metadata field and is not consumed by the canonical prediction.
