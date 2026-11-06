# 3DKD-HyperNet v1 结果

日期：2026-09-05。v1 冻结；训练 OOF 双种子；未读 Test1/Test2。

## 结果

| 模型 | MCC（20260905/20260906） | AUPRC 均值 | Brier 均值 |
| --- | ---: | ---: | ---: |
| A：v1 | 0.82631 | 0.94294 | 0.05764 |
| B：Hypernet 无 KD | 0.82750 / 0.81930 | 0.93563 | 0.06231 |
| D：static + KD | 0.82611 / 0.82380 | 0.93987 | 0.06062 |
| C：Hypernet + KD | 0.82737 / 0.82269 | 0.94065 | 0.06032 |

## 判定

C vs A：

- seed2 MCC 0.82269 < v1 0.82631（未通过）
- mean AUPRC Δ −0.00229、Brier Δ +0.00269（未通过）

C vs D：

- seed2 MCC 低于 static+KD
- mean ΔMCC ≈ 0.00008（要求 ≥0.002，未通过）

C vs B：

- seed1 MCC 略低于无 KD hypernet

结论：`retain_as_ablation`。3D KD 监督能显著提升 hypernet 的 AUPRC/Brier，但 dynamic modulation 在已有 KD 后没有稳定增量；C 仍未达到产品门槛。
