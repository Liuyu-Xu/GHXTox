# Frozen-threshold Test1/Test2 selective-prediction transfer

This package applies the already selected training-OOF `margin_only` policy to the canonical single-structure Test1 and Test2 prediction files. The policy uses only normalized closeness of the frozen GHXTox fusion probability to the fixed binary threshold. Its uncertainty-score cutoff is `0.473617658741599`; samples at or below this value are accepted, and samples above it are labeled `Uncertain`.

The score transform, feature, binary threshold, abstention cutoff, and acceptance inequality were frozen before Test outcomes were inspected. Test-specific quantiles and target coverages are not used. `label_blind_decisions.csv` was written and hashed before labels were read for metrics. Test labels enter only the post-selection metric, random-control, and bootstrap stages.

Random rejection uses 5,000 uniform masks with exactly the realized accepted count for each split (seed 20260824). Paired uncertainty uses 5,000 label-stratified bootstrap replicates (seed 20260825); the original frozen acceptance mask is indexed onto resampled observations and is never recalculated from Test quantiles. No p-values or significance marks are reported.

## Results

- Test1: realized coverage 88.54% (997/1126 accepted); accepted error 2.71% versus full 5.24% and equal-count random mean 5.24%; decision `strong_support`.
- Test2: realized coverage 86.94% (506/582 accepted); accepted error 3.16% versus full 5.33% and equal-count random mean 5.33%; decision `strong_support`.

Overall decision: `strong_support_on_both_benchmarks`.

These are frozen-policy transfer results on established Test1/Test2 benchmark sets. They are stronger than test-specific abstention tuning, but they are not evidence from a newly collected independent dataset and do not authorize any change to the selected policy.
