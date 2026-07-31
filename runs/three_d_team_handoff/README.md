# Retained 3D OOF Team Handoff

This directory is the frozen train-only 3D input contract for the teammates
building the final 1D/2D/3D fusion.

## Files

- `oof_3d_predictions.csv`: 6,387 aligned standalone 3D OOF probabilities plus
  per-row quality and conformer-availability fields.
- `oof_3d_quality.csv`: the same quality/missingness contract without sequence
  or prediction columns.
- `handoff_contract.json`: schema, fixed threshold, metrics, SHA256 provenance,
  and mandatory usage rules.

The retained score is `toxicity_probability`; its frozen standalone decision
threshold is `0.677819`. The probability column must not be recalibrated inside
this package.

## Required join and missingness behavior

Join streams only by `source_index`, then assert exact `fold` and `label`
agreement. `base_structure_available=1` for all rows. The capability-aware
multi-conformer branch is available only when `conformer_available=1`; when it
is zero, ensemble fields are blank and `missing_conformer_mask=1`. Blank fields
must not be converted to physical zeros without a separate missing indicator.

## Leakage boundary

Do not train a fusion/meta-model on this complete OOF table and report its
in-sample result. A valid fusion claim must use the repository's nested OOF
protocol so that every upstream prediction and fusion parameter used for one
outer fold excludes that fold. A plain split of this already-generated OOF
table is not automatically nested because predictions for other folds may come
from upstream models trained with the outer evaluation fold.

Test1/Test2 must not be used to fit fusion weights, calibration, thresholds, or
missingness behavior.
