# ProtTox-LocalMotif v1 OOF decision

## Scope

This is a paired training-OOF comparison over the audited 6,387-record cache and its fixed five group folds. Test1, Test2, website/API code, threshold tuning and all 3D components are excluded.

## Models

The retained Frozen ProtT5 → TriStat + PhysChem MLP is the baseline. Both LocalMotif variants use a separate, SHA-bound float16 cache of the same frozen residue embeddings and a zero-initialized gated logit residual.

- `nonlocal_projection_control`: residue projection, masked global pooling and a parameter-matched MLP.
- `multiscale_cnn_attention`: residue projection, 3/5/7-residue convolutions and masked attention pooling.

The local branches have 144,623 and 144,673 parameters respectively: a difference of 50 (0.035%), within the frozen 0.2% control tolerance.

## Candidate versus retained MLP

| Seed | MCC delta | AUPRC delta | Brier delta |
| --- | ---: | ---: | ---: |
| 20260905 | +0.008521 | +0.011261 | +0.004988 |
| 20260906 | -0.007859 | +0.009771 | +0.008132 |
| Mean | +0.000331 | +0.010516 | +0.006560 |

## Decision

`retain_as_ablation`.

The candidate fails the frozen requirement that MCC must not decline in either seed and fails the required mean MCC gain of +0.005. Its higher AUPRC does not override that instability, and Brier is worse in both seeds. No held-out test evaluation was run; the retained TriStat-MLP stays the default.

Machine-readable provenance and calculations are in `promotion_report.json` in this directory.
