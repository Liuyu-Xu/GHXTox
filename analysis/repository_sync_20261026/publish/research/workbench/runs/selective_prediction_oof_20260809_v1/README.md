# Selective prediction OOF analysis

This directory contains the frozen, training-OOF-only selective-prediction analysis for GHXTox. No Test1 or Test2 labels were loaded. The canonical fused probability, binary threshold, and binary prediction are retained unchanged.

Selected opt-in policy: `margin_only` at nominal 90% coverage. Larger uncertainty scores are less reliable; samples above the frozen policy threshold are reported as `Uncertain`. The class-balanced logistic score is used for ranking and is not a calibrated per-sample error probability.

The `conformer_extended` profile is diagnostic only because conformer-variability fields are not available in canonical single-structure public inference.
