# Formal selective-prediction experiment

This package completes the frozen GHXTox selective-prediction experiment on 6,387 training out-of-fold predictions. The selected policy remains `margin_only`; no feature was reselected, no model/fusion/threshold was changed, and no Test1 or Test2 label was loaded.

The uncertainty rank is a monotone function of closeness to the frozen binary threshold. It is an error-ranking score, not a calibrated error probability. Exact results are reported at 100%, 95%, 90%, 85%, and 80% coverage.

Random rejection uses 5,000 uniform nested random rankings with exactly the same accepted count as selective prediction at every coverage. Paired uncertainty uses 5,000 true-label-stratified bootstrap replicates; each replicate evaluates both the frozen uncertainty ranking and an equal-coverage uniform random ranking on the same resampled observations.

Primary control result at 90% coverage: random-minus-selective accepted error = 0.021914, 95% paired-bootstrap CI [0.018089, 0.025748]. Decision: `supports_difficult_sample_identification_on_training_oof`.

This is a training-OOF reliability analysis, not external validation. The original binary prediction remains available for every abstained sample.
