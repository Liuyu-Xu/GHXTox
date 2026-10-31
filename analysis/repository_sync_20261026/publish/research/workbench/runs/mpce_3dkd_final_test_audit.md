# MPCE-3DKD 最终 Test1/Test2 只读审计

日期：2026-09-05。模型、ensemble、阈值与校准已在审计前冻结；本表不用于任何模型选择。

MPCE-3DKD 口径：canonical batch-size-1 ProtT5 产品 full refit、双 seed 概率平均、固定阈值 0.5、raw probability（未施加展示校准）。

| 数据集 | 模型 | MCC | AUROC | AUPRC | Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| Test1 | 论文 GHXTox | 0.8705 | 0.9811 | 0.9674 | 0.0410 |
| Test1 | MPCE-3DKD | 0.85184 | 0.97355 | 0.95231 | 0.05099 |
| Test2 | 论文 GHXTox | 0.6673 | 0.9433 | 0.8046 | 0.0436 |
| Test2 | MPCE-3DKD | 0.61562 | 0.93470 | 0.71682 | 0.04995 |

审计记录：

- Test1：`runs/mpce_3dkd_final_audit_test1_v1/metrics.json`、`predictions.csv`
- Test2：`runs/mpce_3dkd_final_audit_test2_v1/metrics.json`、`predictions.csv`
- Held-out canonical 特征缓存：`runs/mpce_3dkd_test1_canonical_single_features.pt`、`runs/mpce_3dkd_test2_canonical_single_features.pt`

注意：Test2 是 Test1 的带标签子集（历史审计 split，非独立测试）；论文 GHXTox 数值来自论文/用户给定对照表，未在本仓库重新运行。
