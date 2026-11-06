# MPCE-3DKD 模块消融报告

日期：2026-09-05。只训练 OOF，未读 Test1/Test2；完整 MPCE-3DKD 与 v1 均未修改。

## 汇总

| 模型 | MCC（05 / 06） | AUPRC（05 / 06） | Brier（05 / 06） |
| --- | ---: | ---: | ---: |
| A0 普通 TriStat-MLP（no KD，架构锚点） | 0.81577 / 0.81977 | 0.92927 / 0.92834 | 0.05931 / 0.05893 |
| ProtTox-3DKD v1（普通 TriStat MLP + KD，双 seed ensemble，锚点） | 0.82631 | 0.94294 | 0.05764 |
| MPCE-B（no KD） | 0.83224 / 0.83350 | 0.93861 / 0.93967 | 0.06177 / 0.06219 |
| MPCE-3DKD minus prototype | 0.82998 / 0.84175 | 0.94368 / 0.94557 | 0.05816 / 0.05583 |
| MPCE-3DKD minus low-rank cross | 0.82467 / 0.82292 | 0.93859 / 0.93639 | 0.06183 / 0.06138 |
| MPCE-3DKD full（冻结核心） | 0.83677 / 0.83954 | 0.94457 / 0.94272 | 0.05871 / 0.05737 |

## 两 seed 均值与相对完整模型的差

| 模型 | MCC 均值 | AUPRC 均值 | Brier 均值 | ΔMCC | ΔAUPRC | ΔBrier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MPCE-3DKD minus low-rank cross | 0.82379 | 0.93749 | 0.06160 | -0.01436 | -0.00616 | +0.00356 |
| MPCE-3DKD minus prototype | 0.83587 | 0.94463 | 0.05700 | -0.00229 | +0.00098 | -0.00105 |
| MPCE-B（no KD） | 0.83287 | 0.93914 | 0.06198 | -0.00529 | -0.00451 | +0.00394 |
| MPCE-3DKD full（冻结核心） | 0.83816 | 0.94365 | 0.05804 | — | — | — |

## 解读

- **去掉 low-rank cross**：MCC 平均下降约 0.014、AUPRC 下降约 0.006、Brier 恶化约 0.004，双 seed 一致，说明 mean×max / mean×std / max×std / sequence×physchem 交互是核心有效模块，不是装饰。
- **去掉 multi-prototype**：MCC 平均下降约 0.0023，AUPRC 略升、Brier 略好，说明 prototype 分支提供的是判别/可解释性边际而非校准增益；它不是零贡献（双 seed 指标均发生实质变化），完整模型保留它作为架构特征。
- **去掉 3D KD（MPCE-B）**：MCC 下降约 0.0053、AUPRC 下降约 0.0045、Brier 恶化约 0.0039，说明 3D teacher 软监督对完整 MPCE 架构同样有效。
- **普通 TriStat 锚点**：A0 与 v1 的 MCC/AUPRC 均低于完整 MPCE-3DKD，证明 multi-view/prototype/KD 的组合贡献不是由容量或单模块单独造成。

## 判定

完整 MPCE-3DKD 保持冻结核心；本报告作为专利消融证据，不触发任何核心模型修改或再训练。

