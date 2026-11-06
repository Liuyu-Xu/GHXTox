# MPCE-Net B 第一轮结果（消融）

日期：2026-09-05。v1 未修改，仅训练 OOF，未读 Test1/Test2。

## 冻结设计摘要

- Mean/Max/Std 各投影 64d，PhysChem 投影 64d
- 交互：mean×max / mean×std / max×std / mean×physchem
- 每类 4 prototype（96d cosine），LogSumExp affinity
- z_final = z_global + sigmoid(alpha)·z_proto，alpha 零初始化
- loss：BCE(final) + 0.5·prototype class CE
- 无 KD、无 3D、无 attention/CNN

## OOF 结果

| 模型 | MCC（20260905/20260906） | AUPRC 均值 | Brier 均值 |
| --- | ---: | ---: | ---: |
| A0 原 TriStat-MLP | 0.81577 / 0.81977 | 0.92881 | 0.05912 |
| A1 v1（严格保底） | 0.82631 | 0.94294 | 0.05764 |
| MPCE-B | 0.83224 / 0.83350 | 0.93914 | 0.06198 |

MPCE-B 相对 A1：

- MCC Δ +0.00658（通过 +0.005 门槛，两 seed 均不低）
- AUPRC Δ −0.00380（超过 −0.002 容忍，失败）
- Brier Δ +0.00434（超过 +0.002 容忍，失败）

## 判定

`retain_as_ablation`。B 的 MCC 明显高于普通 MLP 与 v1，说明多视图交叉+多原型对分类判别有真实作用；但排序与概率校准劣于 v1，未达到冻结门槛。

按协议，不进入 C（parameter-efficient ensemble）；v1 继续保持最终交付基线。
