# ProtTox-TSK deployment policy v1

This artifact is an optional post-head policy for the frozen `ProtTox-TSK`
MLP checkpoint. It does not modify the Frozen ProtT5 encoder, TriStat pooling,
physicochemical features, or MLP weights.

## Provenance

- Training feature cache: `runs/prottox_tsk_v1/features.pt` (6,387 records)
- Policy input: complete five-fold training OOF predictions only
- Bound checkpoint: `runs/prottox_tsk_v1/mlp_fivefold_20epochs_v1/final_model.pt`
- Calibration: cross-fitted Platt scaling for OOF policy selection; a separate
  full-training-OOF Platt transform is stored only for deployment application.
- Test1/Test2 labels were not used to fit calibration or choose a threshold.

## Frozen operating modes

| Mode | OOF objective | Threshold | Intended trade-off |
|---|---|---:|---|
| `screening` | maximize F2 | 0.199311 | prioritize sensitivity |
| `balanced` | maximize MCC | 0.486109 | default balanced research mode |
| `strict` | maximize F0.5 | 0.628709 | prioritize precision |

## Test1 audit transfer (not used for selection)

| Mode | ACC | Sensitivity | Precision | MCC | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---|
| screening | 0.8810 | 0.9125 | 0.7337 | 0.7368 | 292/700/106/28 |
| balanced | 0.9281 | 0.8281 | 0.9107 | 0.8200 | 265/780/26/55 |
| strict | 0.9183 | 0.7625 | 0.9385 | 0.7948 | 244/790/16/76 |

The calibrated Test1 Brier score is 0.06010 for all three modes. This is not
an improvement over the raw Test1 Brier result (0.05989), so calibration remains
an optional, provenance-bound policy rather than a promoted claim of better
probability calibration. Test2 is not evaluated for policy selection or
promotion because it is a labeled subset of Test1.
