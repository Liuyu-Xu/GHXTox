# ProtTox-CysConditional v1 OOF decision

## Scope

This is a paired, training-OOF-only comparison. It uses the audited 6,387-record Frozen ProtT5 feature cache and its fixed five group folds. Test1 and Test2 are excluded from model selection and were not accessed for this candidate.

## Frozen candidates

The retained TriStat-MLP is the baseline. `shared_residual` is the parameter-added control. `cys_conditional_residual` routes a learned, zero-initialized gated residual through one of three Cys-count conditions: `0–1`, `2–3`, or `4+`. The condition is not a label rule.

All runs use 20 epochs, learning rate 0.001, and seeds 20260905 and 20260906.

## Paired candidate versus retained MLP

| Seed | MCC delta | AUPRC delta | Brier delta | Low-Cys toxic sensitivity delta |
| --- | ---: | ---: | ---: | ---: |
| 20260905 | +0.002637 | +0.000011 | -0.000016 | +0.003817 |
| 20260906 | +0.000365 | -0.000129 | -0.000002 | +0.003817 |
| Mean | +0.001501 | -0.000059 | -0.000009 | +0.003817 |

## Frozen promotion gate

The candidate must have non-lower MCC in both seeds, mean MCC gain at least +0.005, mean AUPRC decline no worse than -0.002, and improve low-Cys toxic sensitivity by at least +0.05 in both seeds.

## Decision

`retain_as_ablation`.

Although aggregate MCC did not decline and AUPRC stayed effectively unchanged, the MCC gain is below the required +0.005 and the target low-Cys sensitivity improvement is far below +0.05. The retained MLP remains the default. No held-out test evaluation was run for this non-promoted candidate.

Machine-readable provenance and calculations: `promotion_report.json` in this directory.
