# PhysChem-Conditioned Hypernetwork Dynamic Predictor 第一轮

日期：2026-09-05。v1 冻结；训练 OOF 双种子；未读 Test1/Test2。

## 结果

| 模型 | MCC（20260905/20260906） | AUPRC 均值 | Brier 均值 |
| --- | ---: | ---: | ---: |
| A：v1 | 0.82631 | 0.94294 | 0.05764 |
| B：静态低秩网络 | 0.82105 / 0.81852 | 0.93493 | 0.06303 |
| C：Hypernetwork 动态预测器 | 0.82750 / 0.81930 | 0.93563 | 0.06231 |

## 判定

C vs B：

- 两 seed MCC 均不低于 B（0.82750>0.82105；0.81930>0.81852）
- mean ΔMCC = +0.00361（通过 ≥0.002）
- AUPRC Δ +0.00070、Brier Δ −0.00072（均通过）

C vs A：

- seed2 MCC 0.81930 < v1 0.82631（“两 seed 均不低”未通过）
- mean MCC、AUPRC、Brier 均低于 v1

结论：`dynamic_conditioning_supported_but_not_product`。PhysChem 条件动态调制确实优于同容量静态低秩网络，说明“样本级动态参数”假设有初步正信号；但整体仍明显低于 3DKD v1，未达产品替换门槛。
