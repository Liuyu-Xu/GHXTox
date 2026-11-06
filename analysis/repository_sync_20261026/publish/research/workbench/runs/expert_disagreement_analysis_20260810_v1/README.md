# Frozen 1D--3D expert disagreement analysis

This package classifies every canonical Test1/Test2 sample into four mutually exclusive groups using the pre-existing standalone thresholds: 1D = `0.392` and 3D = `0.677819`. No checkpoint, probability, calibration, fusion rule, or threshold is changed.

## Outputs

- `sample_assignments.csv`: identity-bearing probabilities, decisions, correctness groups, and all five requested attributes.
- `group_counts.csv`: group size and true-label composition.
- `group_attribute_summaries.csv`: four-group descriptive distributions and confidence intervals.
- `rescue_rates.csv`: reciprocal expert rescue rates overall and by true label.
- `primary_contrasts.csv`: 3D-only-correct versus 1D-only-correct effects, bootstrap intervals, label-stratified permutation p-values, and BH q-values.
- `prespecified_bins.csv`: frozen descriptive length/Cys/pLDDT/site-coverage views, including availability within length bins.
- `conformer_source_summary.csv`: observational generator-availability composition.
- `results_table.md`, `summary.json`, `input_manifest.csv`, and `checksums.sha256`: human- and machine-readable provenance.

## Statistical contract

Continuous group summaries use medians/IQRs and label-stratified bootstrap intervals. The primary continuous effect is Cliff's delta; conformer availability uses a proportion difference. Tests shuffle the two discordant group labels within true-label strata. BH correction is applied once across the ten split-by-attribute primary tests.

## Interpretation boundary

These are post hoc descriptive associations on established benchmarks. pLDDT is prediction confidence, chemical sites are model-derived candidates, and conformer availability is confounded by generator/length assignment. The canonical public predictions use single-structure 3D base logits, so availability does not mean the prediction consumed a conformer ensemble. Test1 and Test2 must remain separate.
