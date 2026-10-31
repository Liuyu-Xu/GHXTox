# 溶血辅助监督数据可行性

已完成一次有界数据审查，并冻结用于小头分类试验的来源标签清单。不是重构GHXTox数据集。

|阶段|序列数|
|---|---:|
|HemoPI2 cross_val来源记录|1540|
|标准20氨基酸、11–50残基|1428|
|排除保留评估相关连通群|930|
|剩余候选|498|
|每折最终辅助训练|459–481|

高/低活性两类均可用，数据量足以做低成本来源标签级探索。未达到逐条原实验/化学形式已核实的金标准；已知PMID排除不证明论文独立。

已据固定方案完成分类试验，MCC未稳定提高，继续门槛失败；不继续扩建这套辅助配置。详细方法与结果见[分类试验报告](../hemolysis_auxiliary_classifier_pilot_20261026/REPORT.md)。

prepare.py、auxiliary_source_records.json、fold_manifests.json、protected_sequence_manifest.json、joint_sequence_components.json、near_neighbor_edges.json、summary.json与verification.json保留全部来源和划分，可复核。不评分保留独立集、不修改原毒性标签。
