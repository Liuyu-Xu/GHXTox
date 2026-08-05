# -*- coding: utf-8 -*-
p = "reports/mentor_report/REPORT.md"
s = open(p, encoding="utf-8").read()

old_fig = "![默认模型架构](figure6_default_architecture.png)\n\n图 1. 默认模型由 15 成员 ProtT5 序列专家、pLDDT 感知全原子 3D 专家、冻结 `0.62/0.38` 融合、ProtT5-to-3D 交互适配器和最后一层 BACC 决策残差组成。"
new_fig = open("tmp/new_fig_block.md", encoding="utf-8-sig").read().strip()
assert old_fig in s, "figure anchor not found"
s = s.replace(old_fig, new_fig, 1)
print("figure block replaced")

anchor = "## 4. "
i = s.find(anchor)
assert i != -1
if "### 3.2" not in s:
    sec32 = open("tmp/sec32.md", encoding="utf-8-sig").read().rstrip()
    s = s[:i] + sec32 + "\n\n" + s[i:]
    print("3.2 inserted")
else:
    print("3.2 present")

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("REPORT.md updated, bytes:", len(s.encode("utf-8")))
