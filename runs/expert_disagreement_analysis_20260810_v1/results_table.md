# Frozen 1D--3D expert disagreement results

All groups use the training-derived 1D threshold 0.392 and 3D threshold 0.677819. Values are descriptive associations on established benchmarks.

## Test1

| Group | n | Toxic n | Length median [IQR] | Cys median [IQR] | Mean pLDDT median [IQR] | Site coverage median [IQR] | Conformer available |
|---|---:|---:|---:|---:|---:|---:|---:|
| Both experts correct | 1032 | 277 | 33.000 [23.000, 41.000] | 0.000 [0.000, 4.000] | 0.737 [0.634, 0.825] | 0.929 [0.884, 0.960] | 424/1032 (41.1%) |
| 1D expert only correct | 29 | 15 | 25.000 [18.000, 40.000] | 1.000 [0.000, 4.000] | 0.701 [0.573, 0.803] | 0.933 [0.900, 0.975] | 18/29 (62.1%) |
| 3D expert only correct | 24 | 5 | 20.000 [15.000, 28.750] | 1.000 [0.000, 2.000] | 0.656 [0.587, 0.763] | 0.935 [0.905, 0.976] | 18/24 (75.0%) |
| Both experts wrong | 41 | 23 | 26.000 [20.000, 34.000] | 1.000 [0.000, 4.000] | 0.647 [0.570, 0.765] | 0.920 [0.846, 0.957] | 26/41 (63.4%) |

Direct complementarity:

- three_d_rescue_among_one_d_errors: 24/65 = 36.92% (95% Wilson CI 26.23--49.08%).
- one_d_rescue_among_three_d_errors: 29/70 = 41.43% (95% Wilson CI 30.63--53.12%).

## Test2

| Group | n | Toxic n | Length median [IQR] | Cys median [IQR] | Mean pLDDT median [IQR] | Site coverage median [IQR] | Conformer available |
|---|---:|---:|---:|---:|---:|---:|---:|
| Both experts correct | 529 | 33 | 31.000 [17.000, 41.000] | 0.000 [0.000, 1.000] | 0.726 [0.645, 0.810] | 0.931 [0.886, 0.968] | 250/529 (47.3%) |
| 1D expert only correct | 13 | 2 | 28.000 [25.000, 41.000] | 2.000 [0.000, 4.000] | 0.694 [0.548, 0.797] | 0.950 [0.900, 0.975] | 7/13 (53.8%) |
| 3D expert only correct | 15 | 1 | 18.000 [15.000, 26.500] | 0.000 [0.000, 2.000] | 0.703 [0.593, 0.772] | 0.938 [0.920, 1.000] | 12/15 (80.0%) |
| Both experts wrong | 25 | 10 | 26.000 [16.000, 36.000] | 2.000 [0.000, 4.000] | 0.610 [0.563, 0.693] | 0.913 [0.815, 0.951] | 15/25 (60.0%) |

Direct complementarity:

- three_d_rescue_among_one_d_errors: 15/40 = 37.50% (95% Wilson CI 24.22--52.97%).
- one_d_rescue_among_three_d_errors: 13/38 = 34.21% (95% Wilson CI 21.21--50.11%).

## Primary contrast: 3D-only correct minus 1D-only correct

| Split | Attribute | Effect | 95% CI | Permutation p | BH q |
|---|---|---:|---:|---:|---:|
| Test1 | Peptide length | -0.286 | [-0.549, 0.000] | 0.0978 | 0.4890 |
| Test1 | Cys count | -0.112 | [-0.374, 0.154] | 0.4630 | 0.7085 |
| Test1 | Mean pLDDT | -0.057 | [-0.348, 0.241] | 0.7232 | 0.8857 |
| Test1 | Chemical-site coverage | 0.000 | [-0.303, 0.310] | 1.0000 | 1.0000 |
| Test1 | Conformer availability | 0.129 | [-0.106, 0.358] | 0.3985 | 0.7085 |
| Test2 | Peptide length | -0.523 | [-0.831, -0.164] | 0.0156 | 0.1560 |
| Test2 | Cys count | -0.308 | [-0.631, 0.046] | 0.1540 | 0.5133 |
| Test2 | Mean pLDDT | 0.159 | [-0.303, 0.590] | 0.4960 | 0.7085 |
| Test2 | Chemical-site coverage | 0.062 | [-0.364, 0.477] | 0.7971 | 0.8857 |
| Test2 | Conformer availability | 0.262 | [-0.036, 0.559] | 0.2148 | 0.5369 |

For continuous attributes the effect is Cliff's delta; for conformer availability it is a proportion difference. Positive values indicate larger values or higher availability in the 3D-only-correct group.
