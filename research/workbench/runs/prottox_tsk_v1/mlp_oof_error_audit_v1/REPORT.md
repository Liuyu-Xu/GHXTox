# ProtTox-TSK MLP training-OOF error audit v1

## Scope

This report is an engineering diagnostic over the retained 6,387-row training
OOF prediction file only. It does not use Test1, Test2, website requests, or
3D data.

## Error counts

`TP/TN/FP/FN = 1558 / 4353 / 216 / 260` at the frozen threshold of 0.5.

## Stable within-label error patterns

False negatives compared with correctly recognized toxic peptides show lower
Cys count/fraction (standardized mean differences `-2.681/-2.378`, five-fold
sign consistency `5/5`), higher hydrophobic fraction (`+1.408`, `5/5`), and
shorter length (`-0.632`, `5/5`).

False positives compared with correctly recognized non-toxic peptides show
higher Cys count/fraction (`+1.380/+1.534`, `5/5`), lower hydrophobic fraction
(`-1.016`, `5/5`), and wider minimum Cys spacing (`+0.538`, `5/5`).

## Cys-regime error rates

| Cys count | Toxic records | FN rate among toxic | Non-toxic records | FP rate among non-toxic |
|---|---:|---:|---:|---:|
| 0 | 207 | 0.7391 | 3006 | 0.0093 |
| 1 | 55 | 0.7273 | 531 | 0.0339 |
| 2–3 | 159 | 0.3145 | 587 | 0.0954 |
| 4+ | 1397 | 0.0122 | 445 | 0.2562 |

## Engineering decision

The retained MLP encodes a Cys-count toxicity prior that is too strong for the
low-Cys toxic and high-Cys non-toxic regimes. Do **not** add a hard coded Cys
rule or tune a test threshold. If a new candidate is authorized, it should be
pre-registered as a lightweight, learned Cys-regime-conditioned residual using
the three regimes `0–1`, `2–3`, and `4+`, evaluated only under the same paired
five-fold OOF protocol.
