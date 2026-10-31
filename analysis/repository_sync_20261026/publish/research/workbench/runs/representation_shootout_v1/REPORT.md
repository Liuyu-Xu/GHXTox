# PepBERT Representation Shootout（V1 冻结协议）结果

日期：2026-09-05。只训练 OOF，未读 Test1/Test2；ProtTox-3DKD v1 与现有前端均未改动。

## OOF 汇总

| 臂 | MCC（20260905 / 20260906） | AUPRC（05 / 06） | Brier（05 / 06） |
| --- | ---: | ---: | ---: |
| A0 ProtT5 TriStat-MLP (no KD) | 0.81577 / 0.81977 | 0.92927 / 0.92834 | 0.05931 / 0.05893 |
| B PepBERT-small-UniParc TriStat-MLP (no KD) | 0.72129 / 0.72198 | 0.86383 / 0.86272 | 0.08425 / 0.08393 |
| C PepBERT-large-UniParc TriStat-MLP (no KD) | 0.76487 / 0.76471 | 0.89524 / 0.89229 | 0.07296 / 0.07353 |

两 seed 均值差（相对 A0）：

| 候选 | ΔMCC | ΔAUPRC | ΔBrier | 判定 |
| --- | ---: | ---: | ---: | --- |
| B PepBERT-small-UniParc TriStat-MLP (no KD) | -0.09614 | -0.06553 | +0.02497 | 不通过 |
| C PepBERT-large-UniParc TriStat-MLP (no KD) | -0.05298 | -0.03504 | +0.01412 | 不通过 |

## 每折 MCC

| 臂/seed | fold0 | fold1 | fold2 | fold3 | fold4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A 20260905 | 0.85290 | 0.82006 | 0.81666 | 0.79907 | 0.79128 |
| A 20260906 | 0.86268 | 0.82056 | 0.82418 | 0.80158 | 0.79097 |
| B 20260905 | 0.75947 | 0.72534 | 0.71306 | 0.72065 | 0.68999 |
| B 20260906 | 0.76166 | 0.71909 | 0.70345 | 0.72276 | 0.70355 |
| C 20260905 | 0.80524 | 0.76449 | 0.75827 | 0.75163 | 0.74580 |
| C 20260906 | 0.80201 | 0.75943 | 0.75984 | 0.75372 | 0.74918 |

## 运行/规模信息

| 臂 | encoder | 参数量 | MLP 参数 | TriStat 维度 | 全量 6387 特征生成 | 峰值显存 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B | pepbert-small-uniparc | 1,861,469 | 95,251 | 489 | 3.82s | 128.0 MiB |
| C | pepbert-large-uniparc | 4,947,869 | 188,371 | 969 | 4.69s | 166.1 MiB |

## 判定

两个 PepBERT 臂的双 seed MCC、AUPRC、Brier 全部稳定差于现有 ProtT5 TriStat-MLP。

- B（PepBERT-small-UniParc）：MCC −0.097 左右，AUPRC −0.066，Brier +0.025；
- C（PepBERT-large-UniParc）：MCC −0.053，AUPRC −0.035，Brier +0.014；
- 没有任何候选满足冻结的 continue gate，且不是单折/单 seed 偶然，而是 5 折 2 seed 一致落后。

**结论：按冻结协议停止 PepBERT representation 线，不进入 AMT-PepTox 设计。**

现有 ProtTox-3DKD v1 继续作为不可修改的部署保底；后续不会把本次失败作为借口回退交付。

