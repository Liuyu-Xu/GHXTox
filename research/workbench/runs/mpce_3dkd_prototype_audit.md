# MPCE-3DKD prototype interpretability audit

prototype 共 8 个：0–3 对应 toxic affinity 侧，4–7 对应 non-toxic affinity 侧；统计基于各自 fold checkpoint 对自身 outer fold 的输出，样本按 seed 各出现一次。

## 使用/坍缩检查

- global closest 每 seed 计数：{0: 353.0, 1: 561.5, 2: 465.0, 3: 332.5, 4: 1157.0, 5: 1260.0, 6: 1252.5, 7: 1005.5}
- toxic 侧计数：{0: 1589.0, 1: 1808.5, 2: 1770.0, 3: 1219.5}
- non-toxic 侧计数：{0: 1601.5, 1: 1667.5, 2: 1620.0, 3: 1498.0}
- 分配熵 2.826 / 3.000 bits（normalized 0.942）
- collapse check：no_collapse

## global-closest prototype 的 label toxic fraction 与 PhysChem 均值

| prototype | label=1 fraction | length | net charge | hydropathy | basic frac | hydrophobic frac | aromatic frac | Cys count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.865 | 31.3 | 0.39 | -0.30 | 0.133 | 0.268 | 0.079 | 4.90 |
| 1 | 0.932 | 32.0 | 0.32 | -0.27 | 0.130 | 0.267 | 0.086 | 5.40 |
| 2 | 0.908 | 32.6 | 0.39 | -0.24 | 0.133 | 0.280 | 0.085 | 5.19 |
| 3 | 0.917 | 32.3 | 0.23 | -0.24 | 0.128 | 0.270 | 0.082 | 5.30 |
| 4 | 0.060 | 31.9 | 3.00 | -0.04 | 0.165 | 0.430 | 0.090 | 0.92 |
| 5 | 0.062 | 30.9 | 1.83 | -0.04 | 0.148 | 0.426 | 0.089 | 0.91 |
| 6 | 0.042 | 32.2 | 2.45 | -0.14 | 0.162 | 0.415 | 0.086 | 0.92 |
| 7 | 0.063 | 30.9 | 2.28 | -0.07 | 0.157 | 0.427 | 0.090 | 0.84 |

完整逐样本 assignment 见 `runs/mpce_3dkd_prototype_assignment_ledger.csv`。
