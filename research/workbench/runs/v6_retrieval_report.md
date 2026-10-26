# V6-0 Retrieval Memory Diagnostic

日期：2026-09-05。只做诊断；未训练 retrieval 架构；未改 v1；未读 Test1/Test2。

## 方法

- outer fold k 的 memory bank 仅用 fold != k 样本；
- cosine similarity over standardized TriStat+PhysChem；
- local toxicity score = 按 1/(1+d) 加权的 kNN 标签均值（k=5,11）；
- 查询与 v1 排错方向比较。

## 结果

| k | AUROC 均值 | AUPRC 均值 | Spearman vs v1 | v1 错误方向正确率 |
| --- | ---: | ---: | ---: | ---: |
| 5 | 0.952 | 0.909 | 0.801 | 0.31–0.46 |
| 11 | 0.960 | 0.929 | 0.827 | — |

5 折 AUROC 范围：k5 0.944–0.960，稳定但不强于 v1（v1 AUROC≈0.973）。

## 判定

- local evidence 本身有较强毒性排序信息；
- 但与 v1 概率高度冗余（Spearman 0.80–0.83）；
- 在 v1 排错样本上，local 方向正确率仅 31–46%，明显低于有用阈值；

按预注册停止条件：`stop_v6_retrieval`。检索记忆没有提供稳定的补充信号；若再训练 global+retrieval 架构大概率只是让模型更复杂。
