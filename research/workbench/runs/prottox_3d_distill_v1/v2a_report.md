# ProtTox-3DKD-V2A：3D-teacher residual KD（消融记录）

日期：2026-09-05。v1（ProtTox-3DKD v1.0）保持冻结只读；本实验只做训练 OOF，未读取 Test1/Test2。

## 设计

```text
z_final = z_base(v1 fold ensemble, frozen) + g(x) * delta(x)
residual target r = z_teacher_logit - z_base_logit
L = BCE + KD(T=2, lambda=1) + Huber(residual, lambda=1)
```

每折 k：v1 base = mlp_teacher_kd seed 20260905/20260906 的 fold-k checkpoint 概率平均后取 logit；teacher = 对应 outer-safe teacher logit；残差头零初始化。

## 修正记录

第一版目标生成与评估漏用 v1 checkpoint 的 feature mean/std，直接把原始 3081 维特征输入 TriStatMLPHead，产生无效结果（已标记为 superseded 目录：`v2a_targets`、`v2a_seed*_v1`）。修正归一化后使用 `v2a_targets_corrected` 与 `v2a_seed*_corrected_v1` 重新生成并重跑。

## 结果（训练 OOF）

| 模型 | MCC（20260905/20260906） | AUPRC | Brier |
| --- | ---: | ---: | ---: |
| v1 fold-base ensemble（对照） | 0.82631 | 0.94294 | 0.05764 |
| mlp_teacher_kd（v1 实验 seed） | 0.82706 / 0.82392 | 0.94172 / 0.94173 | 0.05851 / 0.05943 |
| V2-A residual KD | 0.81886 / 0.82626 | 0.93653 / 0.93798 | 0.06224 / 0.06041 |

V2-A 相对 v1 fold-base ensemble：

- seed1：ΔMCC −0.00745、ΔAUPRC −0.00641、ΔBrier +0.00460
- seed2：ΔMCC −0.00005、ΔAUPRC −0.00496、ΔBrier +0.00277

## 判定

`retain_as_ablation`。两个 seed 均未达到“MCC 不低于对照、均值 +0.005、AUPRC 不降、Brier 不升”的冻结门槛。较合理的解释：residual target 实际是 teacher（OOF MCC≈0.78）与 v1（OOF MCC≈0.83）之间的差值；让 v1 去学一个更弱模型的修正，在多数样本上没有带来可复现收益。gate + residual 的表达能力没有弥补这一点。

## 产物

- 协议：`configs/prottox_3dkd_v2a_protocol_v1.json`
- 修正目标表：`runs/prottox_3d_distill_v1/v2a_targets_corrected/`
- 双 seed OOF：`runs/prottox_3d_distill_v1/v2a_seed20260905_corrected_v1/`、`.../v2a_seed20260906_corrected_v1/`

下一步按优先级进入 V2-B（teacher pooled representation feature KD），v1 仍不修改。
