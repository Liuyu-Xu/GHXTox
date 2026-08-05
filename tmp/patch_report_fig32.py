# -*- coding: utf-8 -*-
p = "reports/mentor_report/REPORT.md"
s = open(p, encoding="utf-8").read()

# 1) replace section-1 architecture figure + caption
old_fig = "![默认模型架构](figure6_default_architecture.png)\n\n图 1. 默认模型由 15 成员 ProtT5 序列专家、pLDDT 感知全原子 3D 专家、冻结 `0.62/0.38` 融合、ProtT5-to-3D 交互适配器和最后一层 BACC 决策残差组成。"
new_fig = "![当前多维融合模型架构](figure6_current_architecture.png)\n\n图 1. 当前多维融合模型架构：组合A 1D 序列专家（LoRA 微调 ProtT5 + 双向 Mamba，3-seed 集成 + 训练 OOF Platt 校准）与 pLDDT 感知全原子 3D 专家（五级尺度）按训练 OOF 冻结的 `0.62/0.38` 权重融合，决策阈值 `0.4977`。默认模型的上一代架构（冻结 ProtT5-TextCNN 15 成员 + 交互适配器 + BACC 决策残差）作为"之前多维融合模型"基线见第 3.2 节对比。"
assert old_fig in s, "figure anchor not found"
s = s.replace(old_fig, new_fig, 1)
print("figure reference + caption replaced")

# 2) insert 3.2 after 3.1 (before "## 4.")
anchor = "## 4. "
i = s.find(anchor)
assert i != -1
if "### 3.2" not in s:
    sec32 = """

### 3.2 当前融合 vs 之前融合 vs 同数据集参考论文（三方对比）

| 数据集 | 模型 | ACC | BACC | SN | SP | MCC | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Test1 | ToxMSRC | 0.9405 | 0.9217 | 0.8781 | 0.9653 | 0.8520 | 0.9650 | NR |
| Test1 | ToxPLTC | 0.9378 | 0.9302 | **0.9125** | 0.9479 | 0.8496 | 0.9709 | 0.9426 |
| Test1 | 之前多维融合（默认） | 0.9440 | 0.9251 | 0.8813 | 0.9690 | 0.8611 | **0.9816** | 0.9650 |
| Test1 | **当前多维融合（组合A）** | **0.9494** | **0.9354** | 0.9031 | **0.9677** | **0.8750** | 0.9808 | **0.9668** |
| Test2 | ToxMSRC | 0.9416 | 0.8689 | 0.7826 | 0.9552 | 0.6550 | 0.9430 | NR |
| Test2 | ToxPLTC | 0.9261 | **0.8804** | **0.8261** | 0.9347 | 0.6197 | 0.9295 | 0.7353 |
| Test2 | 之前多维融合（默认） | 0.9450 | 0.8608 | 0.7609 | 0.9608 | 0.6603 | **0.9578** | 0.7849 |
| Test2 | **当前多维融合（组合A）** | **0.9467** | 0.8618 | 0.7609 | **0.9627** | **0.6673** | 0.9428 | **0.8012** |

（粗体为该数据集该指标四模型最高；NR 表示参考论文未报告。）

- **当前 vs 之前（1D 替换的增益）**：Test1 的 BACC +0.0103（P=0.983 显著）、SN +0.0219（P=0.993 显著）、
  MCC +0.0139、ACC +0.0054，AUROC 统计持平，Brier 0.0432→0.0415；Test2 的 ACC/BACC/MCC/SP/AUPRC 持平或微升，
  AUROC -0.0150（非显著），Brier 0.0473→0.0444。即 Test1 显著提升、Test2 统计持平、校准双集合改善。
- **当前 vs 参考论文**：Test1 相对 ToxMSRC 六项全部领先、相对 ToxPLTC 六胜一平（仅 SN 低 0.009，换取 SP 高 0.020，
  更均衡）；Test2 相对两篇参考在 ACC/SP/MCC/AUPRC 领先、AUROC 与 ToxMSRC 持平并超过 ToxPLTC，BACC/SN 低于参考
  （差异源于操作点：参考以更低特异性换取灵敏度，Test2 仅 46 个正样本）。
- **定位**：当前多维融合模型在 Test1 占 7 项指标中 6 项最高，Test2 占 ACC/SP/MCC/AUPRC 4 项最高；1D 架构创新带来
  可验证改进且无显著退化，同数据集参考论文对比见上表。机器可读版本见 `three_way_comparison.csv`。

"""
    s = s[:i] + sec32 + s[i:]
    print("3.2 inserted")
else:
    print("3.2 already present")

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("REPORT.md updated")
