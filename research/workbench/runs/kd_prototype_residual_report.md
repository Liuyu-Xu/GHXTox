# KD-Global + Prototype-Residual 第一轮结果

日期：2026-09-05。v1 冻结作 global，未改；训练 OOF 双种子；未读 Test1/Test2。

## 设计

- A：frozen v1（p=(p1+p2)/2 → z_global）
- B：v1 + 同参数量 flat MLP residual，z = z_global + 0.25·tanh(Δ)
- C：v1 + multi-view interaction + 每类 4 prototype residual（真实标签监督，无 KD）
- 修正头只在 fold≠k 训练，fold k 评价

## OOF 结果

| 模型 | MCC（20260905/20260906） | AUPRC 均值 | Brier 均值 |
| --- | ---: | ---: | ---: |
| A：v1 | 0.82631 | 0.94294 | 0.05764 |
| B：v1+flat residual | 0.82750 / 0.82716 | 0.94394 | 0.05700 |
| C：v1+prototype residual | 0.82961 / 0.82910 | 0.94347 | 0.05726 |

## 判定

C vs A：

- MCC Δ = +0.00304（要求 ≥0.005，未通过，差 0.002）
- AUPRC Δ = +0.00053（通过）
- Brier Δ = −0.00038（通过）

C vs B：

- 两 seed MCC 均不低于 B，平均 Δ = +0.00202（≥0.002，通过）
- AUPRC Δ = −0.00046（通过）
- Brier Δ = +0.00026（通过）

结论：`not_promoted_vs_v1`。prototype residual 确实优于 flat residual，说明 interaction+prototype 归纳偏置有效；但相对 v1 的 MCC 增益还不够门槛，修正幅度被 β=0.25 限制。

## Decision-flip audit（C seed1）

- v1 FP → C correct：7
- v1 FN → C correct：10
- v1 correct → C wrong：9
- v1 correct 保持正确：5932

被修正样本的 v1 margin 均值约 0.035，残差接近饱和；修正主要发生在决策边界附近，错误回退也集中在边界样本。
