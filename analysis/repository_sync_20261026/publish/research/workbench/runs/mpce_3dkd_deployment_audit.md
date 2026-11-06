# MPCE-3DKD deployment-consistency audit

audit set：200 条训练序列（每 fold 40 条，RNG seed 20260905），只比较推理一致性，不使用 label 或 Test1/Test2。

| check | max abs diff | tolerance | pass |
| --- | ---: | ---: | --- |
| cache inference vs standalone runner（ensemble） | 1.192e-07 | 1e-05 | PASS |
| cache vs runner（seed 20260905） | 1.788e-07 | 1e-05 | PASS |
| cache vs runner（seed 20260906） | 5.960e-08 | 1e-05 | PASS |
| runner repeat determinism | 0.000e+00 | 1e-09 | PASS |

整体判定：PASS。
