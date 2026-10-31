# 近期 GHXTox 工作复盘（2026-10-29）

基于 D:/dachuang2026/GHXTox/analysis 中近期项目报告、预测产物与实验记录；不表示逐行审计全部原始代码。模型与数据已经反复用于探索，结果不等于独立验证。

## 已实际执行

|工作|结论|记录|
|---|---|---|
|HemoPI2、NTxPred2|作者权重都对6387条跑过；前者溶血端点不同且负类不一致，后者监督数据重叠大且作者独立集复现有疑点；均不宜直接融合|../mechanism_teacher_audit_20261027/REPORT.md|
|溶血辅助毒性分类|冻结ProtT5小头试过真实、打乱辅助任务；真实相对基线MCC+0.00131，区间跨0，打乱对照点估计更高；不是完整GHXTox|../hemolysis_auxiliary_classifier_pilot_20261026/REPORT.md|
|普通监督对比|仅fold3单种子，序列+历史3D条件融合MCC 0.8308→0.8455，低Cys原正类漏判38→42；其他四折没有对比模型预测|../supcon_pilot_20261028/REPORT.md、../cys_method_error_comparison_20261028/REPORT.md|
|低Cys及家族加权|低Cys 1.5倍五折条件融合全体错误408→418；MCD/NDBP额外加权仅首折，目标4条仍漏判|../lowcys_weight_multifold_20261028/REPORT.md、../family_weight_no_lora_20261028/REPORT.md|
|LoRA核验与移除|三份已检查历史权重LoRA B为零；修复LoRA五折单种子未见稳定收益；默认新训练移除，历史模型本身不因移除而提高|../remove_lora_20261028/REPORT.md、../lora_multifold_screen_20261028/REPORT.md|
|三个Cys专家|冻结ProtT5小分类器、同原模型融合均完成五折；原/专家/融合MCC 0.8733/0.8182/0.8616，不能否定其他专家|../cys_three_experts_20261028/REPORT.md|
|SaProt独立、融合和受控选择|两版完整模型、掩码/近似3Di、固定与受控调权重以及门控均有记录；旧版近似3Di方案净少11但区间跨0，不能认定稳定；分数阈值拆解已完成|../saprot_selective_value_20261027/REPORT.md、../saprot_gain_decomposition_20261027/REPORT.md、../saprot_cys_reuse_20261029/REPORT.md|
|Cys概率调整|原完整模型直接固定偏移、阈值及两参数学习均做；未见总体净收益|../cys_probability_adjustment_20261028/REPORT.md、../cys_learned_group_offsets_20261028/REPORT.md|
|风险探针|只提示错误风险，序列结构逻辑回归前10%找出158/327，单看阈值距离161/327；上游非完整嵌套|../cys_error_risk_probe_20261029/REPORT.md|
|标签与分组|疑点和Cys关联、同家族正确对照、阈值距离、低高Cys四分箱及测试集统计均完成；疑点不是错标|../label_cys_joint_20261027/REPORT.md、../cys_extreme_within_family_20261028/REPORT.md、../cys_threshold_distance_20261028/REPORT.md|
|数据来源与溶血剂量|核查各数据集关系、终点及曲线条件，Test2是Test1子集；剂量面板条件缺失，辅助任务未证明增益|../dataset_comparison_20261025/REPORT.md、../hemolysis_dose_audit_20261026/REPORT.md|

## 已查过但未见完成的工作

- ToxMamba/ToxiPep/StrucToxNet的数据关系及候选资料已核查；本次查看的项目报告中没有找到它们对全部训练样本进行已核验同协议模型推理的记录，不排除其他未索引产物。不能把“查过数据集”说成“模型已跑”。
- 监督对比已有首折，仍缺其余四折和多种子；无需重复首折。该方法首折对低Cys漏判未改善，目标是检验总体分类而不是假设它解决该组。
- 轻量风险探针已有，但尚无可靠预测具体纠错对象的模型；不能因为知道事后哪个模型正确就部署门控。

## 避免重复的后续安排

1. 按现有冻结方案补监督对比的其余四折，同其余四折的对应无LoRA单种子基线及历史3D条件融合比较，报告整体及低高Cys各组。该方案不是完整三种子重训；若用户希望原历史系统，需另做匹配完整训练预算。未评估计算资源或耗时。
2. 复用现有数据重叠审计，核查 ToxMamba 可用权重、协议和训练集对GHXTox的监督重叠；存在重叠时不把已有模型全体分数当独立验证。不要重跑HemoPI2/NTxPred2。
3. 留出从未参与模型和规则选择的外部数据；Test1/Test2已反复观察且互相包含，不能作为盲测。若继续训练，按组隔离并把特征选择、校准和训练放在外层内部。
