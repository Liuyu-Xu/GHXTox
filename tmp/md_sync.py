# -*- coding: utf-8 -*-
"""Sync 1D changes across all project md files (UTF-8, no BOM)."""
import os

def read(p):
    return open(p, encoding="utf-8").read()

def write(p, s):
    open(p, "w", encoding="utf-8", newline="\n").write(s)

# ---------- 1) reports/lora_mamba_pilot.md : fix stale BSMOTE claim ----------
p = "reports/lora_mamba_pilot.md"
s = read(p)
old = "数据增强（BSMOTE）与多种子集成（多人投票）保留在流水线外层。"
new = ("数据增强：当前组合A 实现**未启用** BSMOTE（以 pos_weight 处理正负不平衡）；"
       "序列空间点突变增强作为受控实验另行验证（见 docs/README_lora_mamba_mutate_autodl.md）。"
       "多种子集成（多人投票）保留为 3-seed 集成。")
assert old in s, p
s = s.replace(old, new, 1)
if "最终全量结果" not in s:
    s += "\n\n> 全量 3-seed + OOF 校准最终结果与配对自助检验见 `reports/combinationA_final_results.md` 与 `reports/mentor_report/REPORT.md` 第 3.1/5.2 节。\n"
write(p, s)
print("updated", p)

# ---------- 2) docs/default_model.md : note 1D innovation line ----------
p = "docs/default_model.md"
s = read(p)
if "1D innovation" not in s and "组合A" not in s:
    add = """

## 1D innovation line (combination A, does not replace the default)

The public default remains GHXTox-3D-Chemical-Adapter-Residual with its frozen
ProtT5-BSMOTE TextCNN sequence expert.  Separately, an alternative 1D expert --
**LoRA task-adaptive ProtT5 + bidirectional Mamba (S6) sequence classifier** --
was developed and validated as an architecture-differentiation innovation
(3-seed ensemble + training-OOF Platt calibration; fusion weight 0.62/0.38).
Full results, held-out paired-bootstrap CIs, and a controlled sequence-space
point-mutation augmentation experiment are documented in:
- reports/mentor_report/REPORT.md (sections 3.1, 5.2)
- reports/combinationA_final_results.md
- docs/README_lora_mamba_mutate_autodl.md
"""
    s = s.rstrip() + "\n" + add
    write(p, s)
    print("updated", p)
else:
    print("skip (already)", p)

# ---------- 3) README.md : add combination-A section ----------
p = "README.md"
s = read(p)
if "1D 架构创新线" not in s:
    anchor = "## 阅读导航与当前结论"
    assert anchor in s, "anchor missing in README"
    section = """## 1D 架构创新线（组合A：LoRA 微调 ProtT5 + 双向 Mamba）

默认模型的 1D 序列专家仍为冻结 ProtT5-BSMOTE-TextCNN（15 成员）。为满足"1D 架构必须不同"的要求，
另行验证了一条可替换的 1D 创新线（**组合A**）：

- 蛋白质大模型模块 → **ProtT5 编码器 + LoRA 低秩任务自适应微调**（基座+嵌入冻结，可训练参数约 1310 万，r=8/α=16）。
- 分类网络模块 → **双向 Mamba（S6 状态空间）序列分类器**（纯 PyTorch，前向/反向选择扫描、掩码均值池化）。
- 数据增强 → 未启用 BSMOTE（以 pos_weight 处理不平衡）；序列空间点突变增强作为受控实验另行验证。
- 多人投票 → 3-seed 集成；新增训练 OOF Platt 概率校准；融合权重/阈值由训练 OOF 重冻结（w=0.62，阈值 0.4977）。

最终结果（全量 6387，3-seed + 校准）：OOF 融合 BACC 0.9336 / MCC 0.8764 / AUROC 0.9799；
Test1 BACC/SN 显著提升（配对自助检验 P=0.983/0.993），Test2 全部指标统计持平，Brier 双集合改善；
与同数据集参考论文（ToxMSRC/ToxPLTC）对比见导师报告第 3.1 节。

- 报告：`reports/mentor_report/REPORT.md`（3.1/5.2）、`reports/combinationA_final_results.md`
- 复现：`src/ghxtox/lora_mamba_1d.py`、`scripts/run_lora_mamba_full.sh`、`scripts/run_lora_mamba_mutate.sh`
- 受控突变增强实验：`docs/README_lora_mamba_mutate_autodl.md`

"""
    s = s.replace(anchor, section + anchor, 1)
    write(p, s)
    print("updated", p)
else:
    print("skip (already)", p)

# ---------- 4) reports/publication_analysis/REPORT.md : pointer ----------
p = "reports/publication_analysis/REPORT.md"
s = read(p)
if "组合A" not in s:
    s = s.rstrip() + """

---

## 10. 1D 架构创新线（组合A）说明

本报告统计对象为对外默认模型（ProtT5-BSMOTE-TextCNN 序列专家 + 3D 化学位点专家 + 融合 + 适配器 +
BACC 决策残差）。另有一条已验证的 1D 架构创新线 **组合A（LoRA 微调 ProtT5 + 双向 Mamba）**，
其结果（OOF/配对自助检验/校准/长度分层/参考论文对比）见 `reports/mentor_report/REPORT.md` 第 3.1、5.2 节
与 `reports/combinationA_final_results.md`；该线不改变默认模型对外推理路径。
"""
    write(p, s)
    print("updated", p)
else:
    print("skip (already)", p)

# ---------- 5) findings.md / progress.md / task_plan.md : append dated entry ----------
entry = """

---

## 2026-08-05 1D 架构创新线（组合A）定稿 + 受控突变增强实验

- 组合A（LoRA 微调 ProtT5 + 双向 Mamba）全量定稿：3-seed 集成 + 训练 OOF Platt 校准，融合 w=0.62、阈值 0.4977（OOF 冻结）。
- OOF 融合 BACC 0.9336 / MCC 0.8764 / AUROC 0.9799；Test1 BACC/SN 显著提升（P=0.983/0.993），Test2 全指标统计持平，Brier 双集合改善。
- 与同数据集参考论文（ToxMSRC/ToxPLTC）：Test1 6/7 项最高并全面超 ToxMSRC；Test2 ACC/SP/MCC/AUPRC 最高，BACC/SN 低于参考（操作点差异）。
- 长度分层：增益在短肽（11-15）层最一致；唯一负向单元 Test2 16-30（n=162，正样本仅 9），小样本噪声。
- 受控序列点突变增强（rate=0.2，1-2 点突变，折内生成 + 对验证集 ≥0.8 相似度过滤）已实现并打包
  （lora_mamba_mutate_upload.zip），在服务器按预注册判定标准验证。
"""
for p in ("findings.md", "progress.md", "task_plan.md"):
    s = read(p)
    if "2026-08-05 1D 架构创新线" not in s:
        write(p, s.rstrip() + "\n" + entry)
        print("updated", p)
    else:
        print("skip (already)", p)

print("MD SYNC DONE")
