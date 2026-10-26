# ProtTox-3DKD V5-0：Ranking Diagnostic

日期：2026-09-05。只做诊断；未训练 V5、未改 v1、未读 Test1/Test2。

## 口径

- 每条样本用自身 fold 的 outer-safe teacher/base（均未见过该 fold）
- AUROC/AUPRC 与 pair ordering accuracy 在 toxic–non-toxic pairs 上计算
- v1 排反的 pair = toxic p_base < non-toxic p_base
- teacher correct fraction = teacher 将 toxic p 排在 non-toxic p 之上的比例

## 结果（按 fold）

| fold | base AUROC | teacher AUROC | teacher 纠正 v1 排反比例 |
| --- | ---: | ---: | ---: |
| 0 | 0.978 | 0.995 | 94.1% |
| 1 | 0.973 | 0.989 | 84.5% |
| 2 | 0.972 | 0.987 | 87.9% |
| 3 | 0.970 | 0.994 | 90.9% |
| 4 | 0.972 | 0.958 | 45.0% |

总体：teacher AUROC 均值 0.985 vs base 0.973；teacher 平均纠正 v1 排反比例 80.5%。

## 判定

前两个条件（teacher 排序能力强、能修正 v1 错误排序）在 fold0–3 成立且很强，但 **fold4 稳定性不成立**：teacher AUROC 低于 base，且对 v1 排反 pair 的纠正率仅 45%（接近随机）。

按“跨 5 折稳定”门槛：`do_not_authorize_v5_yet`。fold4 需要先解释其 teacher 为何异常（例如该折 teacher checkpoint 本身 OOF 明显偏弱），在修复/重训 teacher 前不正式训练 V5；也不允许剔除 fold4 或事后换 teacher 来让结果“通过”。

该结果作为“ranking distillation 仍有潜在价值但当前 teacher 流不满足稳定性”的存档证据保留。
