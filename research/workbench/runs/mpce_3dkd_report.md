# MPCE-3DKD 最终实验报告

日期：2026-09-05。只训练 OOF，未读 Test1/Test2；ProtTox-3DKD v1 与前端均未改动。

## 冻结设计摘要

- 架构完全复用 MPCE-B：Mean/Max/Std 投影 64d、PhysChem 投影 64d、四组低秩交互、每类 4 个 96d prototype、z_final = z_global + sigmoid(alpha)·z_proto。
- 训练目标：BCE(z_final) + 0.5·prototype CE + 1.0·KD(z_final, teacher；T=2.0)。
- teacher 复用 ProtTox-3DKD v1 的严格 outer-safe 单结构 3D targets；KD 不修改 v1 的任何 checkpoint。
- 5 group folds × seeds 20260905/20260906，20 epochs、lr 1e-3、batch 256、阈值 0.5。

## OOF 汇总

| 模型 | MCC（05 / 06） | AUPRC（05 / 06） | Brier（05 / 06） |
| --- | ---: | ---: | ---: |
| A0 原 TriStat-MLP | 0.81577 / 0.81977 | 0.92927 / 0.92834 | 0.05931 / 0.05893 |
| A1 ProtTox-3DKD v1（双 seed ensemble，context） | 0.82631 | 0.94294 | 0.05764 |
| B MPCE-B | 0.83224 / 0.83350 | 0.93861 / 0.93967 | 0.06177 / 0.06219 |
| **D MPCE-3DKD（本轮）** | **0.83677 / 0.83954** | **0.94457 / 0.94272** | **0.05871 / 0.05737** |

## 两 seed 均值对比

| 模型 | MCC 均值 | AUPRC 均值 | Brier 均值 |
| --- | ---: | ---: | ---: |
| A1 v1（context） | 0.82631 | 0.94294 | 0.05764 |
| B MPCE-B | 0.83287 | 0.93914 | 0.06198 |
| **D MPCE-3DKD** | **0.83816** | **0.94365** | **0.05804** |

D 相对 B：ΔMCC +0.00529，ΔAUPRC +0.00451，ΔBrier -0.00394。

## 冻结判定

MPCE-3DKD 同时保留 MPCE-B 的判别优势并把概率质量拉回：相对 MPCE-B，MCC 未损失（实际更高）、AUPRC 未下降（实际更高）、Brier 至少改善 0.001（实际明显改善）。

**最终专利架构推荐：MPCE-3DKD。**

部署仍为 sequence-only：ProtT5 → 多视图/低秩交互/多原型 → prediction，整条 3D teacher 删除。

ProtTox-3DKD v1 继续作为不可修改的部署基线；本实验不改变任何 v1 checkpoint、代码或前端。

## 每折 MCC（两 seed 均值）

| fold | v1（context） | B MPCE-B | D MPCE-3DKD | Δ(D−B) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.84193 | 0.85810 | 0.86586 | +0.00776 |
| 1 | 0.83055 | 0.83487 | 0.84494 | +0.01007 |
| 2 | 0.83314 | 0.82944 | 0.82527 | -0.00417 |
| 3 | 0.82794 | 0.81968 | 0.83374 | +0.01406 |
| 4 | 0.80044 | 0.82294 | 0.82173 | -0.00122 |

## 完整性说明

- v1 双 seed ensemble OOF CSV SHA-256：`828d50019559888d...`
- 所有 OOF CSV 均与 audited fold manifest 逐行核对 sample_id/label/fold。
- Test1/Test2 在整个实验中未读取；未做任何 seed/阈值/fold 的事后选择。

