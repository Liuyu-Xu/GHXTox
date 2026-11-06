# ProtTox-3DKD V4：Uncertainty-Gated Privileged Refinement

日期：2026-09-05。训练 OOF 双种子；未改 v1、未读 Test1/Test2。

## 冻结设计

- shared representation → z_base（label head）与 z_priv（privileged head）
- gate：g = 1 − 2·|σ(z_base) − 0.5|（detached）
- final：z = (1−g)·z_base + g·z_priv
- A：v1（已冻结 fold-base 双 seed OOF 对照）
- B：同参数量、固定 g=0.5、全局 teacher KD
- C：同参数量、动态 gate、无 teacher KD
- D：同参数量、动态 gate、teacher KD 按 g 加权（正式候选）

## OOF 结果

| 臂 | MCC（20260905/20260906） | AUPRC（两 seed 均值） | Brier（两 seed 均值） |
| --- | ---: | ---: | ---: |
| A（v1 对照） | 0.82631 | 0.94294 | 0.05764 |
| B（无 gate + KD） | 0.82261 / 0.82906 | 0.94294 | 0.05889 |
| C（gate、无 KD） | 0.83202 / 0.82941 | 0.94294 | 0.05939 |
| D（gate + weighted KD） | 0.83172 / 0.82990 | 0.94311 | 0.05941 |

D 相对 A：

- ΔMCC 均值：+0.00450（门槛要求 ≥ +0.005）
- ΔAUPRC 均值：+0.00017（通过）
- ΔBrier 均值：+0.00177（≤ +0.002，通过）

D 相对 B（+0.00497）通过“不低于”检查；D 相对 C 平均几乎无差（+0.00009），且 seed1 低于 C。

## 判定

`retain_as_ablation`。D 未满足 D_vs_A 的 MCC 门槛，也未证明相对 C 的 teacher 增益；不再推进 V4 部署。值得记录的现象：C（无 teacher 的动态 gate）相对 A 有 +0.0044 MCC，说明“按自身不确定性门控的第二头”本身有一点价值，但幅度不足以晋级，且 D 的 teacher 加权没有带来额外收益。
