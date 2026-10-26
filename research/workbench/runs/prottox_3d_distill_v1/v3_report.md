# ProtTox-3DKD V3-0：Teacher Utility Predictability Diagnostic

日期：2026-09-05。只做诊断，未训练 Dual Expert，未改 v1，未读 Test1/Test2。

## Utility 定义（冻结）

- `u = (y - p_base)^2 - (y - p_teacher)^2`
- helpful：`u > 0.01`
- 每折 probe 使用 teacher_k / base_k（均未见 fold k）为全部样本生成 utility，再按 fold != k 训练、fold=k 验证
- 描述性 prevalence 使用 crossfitted 单表

## 结果（训练 OOF，5-fold probe）

helpful prevalence ≈ 6.3%。

| Feature set | AUROC(helpful) | AUPRC | Spearman | R² | sign acc | >0.60 fold count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: Representation only | 0.522 | 0.099 | −0.040 | −0.571 | 0.480 | 1/5 |
| B: + base prob/margin/entropy | 0.801 | 0.333 | −0.135 | −0.953 | 0.468 | 5/5 |

Oracle upper bound（假设知道每条样本是否 teacher 更好）：

- Brier：0.0576 → 0.0384（Δ −0.019）
- MCC：0.827 → 0.889（Δ +0.062）

说明理论上限可观，但前提是能知道 teacher 何时有用。

## 解读

- base 的自信度/边界信息能对“teacher 是否可能帮忙”的二值问题给出不错 AUROC（0.80，5/5 折 >0.60）；
- 但连续 utility 完全不可预测：Spearman 均值为负、R² 大幅为负、sign accuracy 低于 0.5；
- 说明模型能部分识别 teacher “可能有用”的区域，却不能估计修正的方向/幅度；
- representation-only 接近随机，说明帮助主要和 base 自身犹豫程度相关，而不是可由序列内容额外推断。

## 判定

冻结门槛要求同时满足 prevalence≥5%、mean AUROC≥0.65、4/5 折 AUROC>0.60、mean Spearman≥0.20。实际 mean Spearman=−0.135，门槛未通过。

`stop_v3_dual_expert`。不训练 Reliability-Gated Dual-Expert；ProtTox-3DKD v1 保持最终交付。
