# KD-Global + Prototype Residual：amplitude-validation v2

日期：2026-09-05。v1 冻结；训练 OOF 双种子；未读 Test1/Test2。

## β 选择

- candidates {0.25, 0.5, 0.75}
- per outer fold inner validation（label-stratified 20%）
- eligible 条件：inner AUPRC ≥ baseline−0.002、inner Brier ≤ baseline+0.002；从中选最大 inner MCC，并列取更小 β
- 实际选择几乎全部为 0.25；少数 fold3 选到 0.5

## OOF 结果

| 模型 | MCC（20260905/20260906） | AUPRC 均值 | Brier 均值 |
| --- | ---: | ---: | ---: |
| A：v1 | 0.82631 | 0.94294 | 0.05764 |
| B：inner-selected flat residual | 0.82865 / 0.82831 | 0.94402 | 0.05690 |
| C：inner-selected prototype residual | 0.83039 / 0.82797 | 0.94351 | 0.05734 |

## 判定

C vs A：

- 两 seed MCC 均不低（通过）
- mean ΔMCC = +0.00287（要求 ≥0.005，未通过）
- AUPRC/Brier 通过

C vs B：

- seed2 MCC 0.82797 < B 0.82831（未通过“两 seed 均不低于”）
- mean ΔMCC = +0.00070（要求 ≥0.002，未通过）
- AUPRC/Brier 通过

结论：`stop_beta_search`。即使允许 inner-selected β，C 仍未晋级且未稳定超过 flat residual；选择结果集中在 β=0.25，说明上一轮的差距不是残差幅度不足。

按预注册规则，不再尝试 β=0.4/0.6/1.0 等后续候选。
