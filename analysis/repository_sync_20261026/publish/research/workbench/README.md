# GHXTox研究工作台（2026-10-26）

这里保留研究版源码、冻结协议、正负结果和分析报告。仓库根目录采用上游公开实现；这里的实验不自动改变公开模型。原工作目录未删除或覆盖。

## 从哪里读

- [全部研究总览](reports/研究分析总览_20261025.md)
- [尝试总结与后续优先级](reports/尝试总结与研究优先级_20261026.md)
- [SaProt分类及同规模融合](analysis/saprot_standalone_20261026/REPORT.md)
- [SaProt逐样本贡献分析](analysis/saprot_contribution_20261026/REPORT.md)
- [完整纠错与添错名单](analysis/saprot_contribution_20261026/SAMPLE_LIST.md)
- [历史完整研究README](HISTORICAL_README.md)
- [研究复现和资产说明](REPRODUCIBILITY.md)

## 最新发现

冻结表征上的轻量分类头：同规模4成员基线MCC0.82594，SaProt结构融合0.84006，全结构掩码融合0.83889。结构融合纠正101条、添错63条，净少38条；其中34条来自减少原负类误报。低Cys原正类净添8条错，76条膜作用注释阳性纠0添6。尚不能归因为三维/溶血机制。101条纠错中只有18条也是历史完整GHXTox的错题。

以上不是完整GHXTox的新成绩。两融合仍为研究候选，SaProt蒸馏尚未执行。下一步需同时控制总体性能与低Cys正类/膜作用候选的漏判，并保留普通自蒸馏及全掩码对照。

## 文件组织

`src/`、`configs/`、`scripts/`、`tests/`为研究版实现；`analysis/`和`runs/`为协议与保留结果；`reports/`为汇总入口。大于512KiB的文本结果用`.gz`无损保存，SHA256和原路径在`ASSET_MANIFEST.json`。权重、特征、下载原文及运行日志未打包。负结果同样保留，不能把文件名视作方法已验证成功。
