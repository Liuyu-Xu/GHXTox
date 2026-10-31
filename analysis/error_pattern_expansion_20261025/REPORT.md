# 已有错误模式统计复用说明

用户要求先暂停新增统计，避免重复。此目录仅存重新规划的protocol.json，未执行新的分组/匹配/检验脚本；主线直接复用此前结果。

## 已完成，不再重算

原始入口：[一维/三维分歧报告](D:/dachuang2026/GHXTox/analysis/expert_disagreement_20261025/REPORT.md)。

覆盖全部6387条训练记录，不限定长度、Cys或疏水性。已有outcome_cells.csv列出原标签与一维/三维/融合组合；summary.json、verification.json已完成当时的计数核验。

|用户图中分组|已有记录对应关系|数量|
|---|---|---:|
|A：一维对、三维错、融合错|fusion_harms.csv|48|
|B：两路都错、融合错|both_wrong.csv|229（原阳性漏判136、原阴性误报93）|
|C：至少一个分支错、融合对|outcome_cells.csv中sequence_only_correct且fusion_error=False的138条，加structure_only_correct且fusion_error=False的98条|236|
|D：至少一个分支对、融合错|outcome_cells.csv中sequence_only_correct且fusion_error=True的48条，加structure_only_correct且fusion_error=True的50条|98|

A包含于D，不能把A/B/C/D直接相加。旧文件fusion_corrections.csv的98条专指纠正一维错误，比C定义窄；不能把C也写成98条。

其他已有文件：expert_disagreements.csv（334条分歧）、training_expert_atlas.csv（6387条全集）、expert_groups_by_evidence.csv（证据分层）、primary_matched_pairs.csv（52对）、primary_matched_tests.csv（42项可检验比较）。旧比较并不等于此目录protocol中重新规划的各项A/B/D专属比较，不能互相冒充。

已有52对中，有毒专家对比仅3对，不足检验；可检验的42项BH校正后无q<0.05。严格匹配仅1对。此结果表示尚无稳定新特征证据，不能证明不存在生物学机制。

## 还有什么没有完成

已有36条具体资料案例（28条错题）及24条早期代表样本队列，但不是327条错题或334条分歧全部完成实验机制审核。剩余工作是将最新的分子对象、终点和化学形式证据按sample_id并回既有分组，对未核实样本保留未知，找证据充分的共同错误类型与已有正确对照。

不重跑已完成计数、52对匹配或既有检验；不再以组成预筛限制主线。此前cohort_expansion_20261025仅是特定假设的辅助预筛，不能取代全集错误模式。

本次只查阅已有报告和表格、整理对应关系。旧OOF的非严格嵌套限制仍保留；没有重训、改标签或调用Test1/Test2预测。
