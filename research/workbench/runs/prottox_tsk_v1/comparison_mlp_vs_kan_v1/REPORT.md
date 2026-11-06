# ProtTox-TSK MLP vs KAN OOF audit

Decision: **retain_mlp_default_keep_kan_ablation**

KAN did not improve the primary MCC metric in every paired same-protocol run; it remains an ablation candidate despite any ranking/calibration gains.

| Metric | Mean KAN − MLP | Population SD |
|---|---:|---:|
| MCC | -0.006364 | 0.005501 |
| AUROC | 0.001111 | 0.000394 |
| AUPRC | 0.001526 | 0.001284 |
| BRIER | -0.001277 | 0.000665 |

This audit compares only the listed same-protocol training OOF runs. It is not a Test1/Test2 claim.
