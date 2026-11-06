# MPCE-3DKD low-rank interaction branch audit

方法：对冻结 fold checkpoint 的各自 outer fold，逐一将该 64d 交互分支置零并重算最终概率。

| branch | mean abs activation | MCC（zeroed 均值） | AUPRC（zeroed） | Brier（zeroed） | full−zeroed MCC | full−zeroed AUPRC | full−zeroed Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_max | 0.28597 | 0.83187 | 0.93981 | 0.05841 | +0.00629 | +0.00384 | -0.00037 |
| mean_std | 0.28109 | 0.83258 | 0.94155 | 0.05817 | +0.00557 | +0.00210 | -0.00013 |
| max_std | 0.30305 | 0.83296 | 0.93860 | 0.05796 | +0.00519 | +0.00505 | +0.00009 |
| sequence_physchem | 0.18359 | 0.83561 | 0.94296 | 0.05856 | +0.00255 | +0.00069 | -0.00052 |

完整 MPCE-3DKD 两 seed 均值：MCC 0.83816、AUPRC 0.94365、Brier 0.05804。

结论：若某一分支的置零几乎不改变指标，则该分支仍可能只提供边际信息；若四个分支均有非零影响，则交互模块不是退化空转。
