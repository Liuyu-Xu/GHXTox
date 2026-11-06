# Quick → Full routing OOF study

## Scope

This study uses only the 6,387 identity-aligned training OOF rows. It does not retrain a toxicity predictor and it does not load Test1 or Test2 labels.

`Quick` is the three-seed 1D mean transformed by the frozen web Quick Platt calibration and classified at `0.3920986716`. `Full` is the existing canonical `p_fusion` classified at `0.4702225653`.

## Frozen margin baseline

The web Quick policy’s existing reject band (`±0.2605202561`) recommends Full for 639 / 6,387 records (10.00%). In that group, Full corrects 97 Quick errors and introduces 48 regressions: +49 net corrections. Outside the band, Quick and Full make identical binary decisions on all 5,748 records.

This supports the current deployable behavior: return Quick directly outside the band; recommend Full inside it. Both outputs remain computational predictions and do not establish experimental toxicity.

## Adaptive router screen

`configs/quick_full_adaptive_router_v1.json` freezes a five-fold cross-fitted Ridge expected-Brier-gain router. Its input is available before ESMFold: Quick probability/margin, seed dispersion, length, net charge, hydropathy, composition, and Cys descriptors. No embedding was used because an identity-aligned frozen Quick-embedding OOF artifact is not retained. No labels, Full probability, 3D features, or realized gain can enter router inputs.

The router selected Full for 24.74% of OOF records, compared with 10.00% for the margin baseline. It reached the same +49 net corrections but had worse cost-adjusted utility (`0.0440569` vs `0.0437671`). Its promotion decision is therefore `retain_margin_baseline`.

## Artifacts

- `runs/quick_full_margin_baseline_oof_v1/`: row-level correction ledger and provenance summary.
- `runs/quick_full_adaptive_router_oof_v1/`: cross-fitted router routes and frozen decision.

The router is retained as negative evidence, not connected to the website or Full API.
