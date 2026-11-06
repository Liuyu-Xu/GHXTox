# GHXTox 研究分析目录

[研究分析总览](../reports/研究分析总览_20261025.md)是统一入口。

|目录|内容|主要阅读文件|
|---|---|---|
|ghxtox_error_atlas_20261025|训练集错题地图、相似对照、初轮统计|REPORT.md|
|ghxtox_error_atlas_20261025/evidence_phase2|身份、片段、实验对象与文献证据回查|EVIDENCE_REPORT.md|
|dataset_comparison_20261025|12个公开项目/资源的数据关系、标签差异、候选验证样本|REPORT.md|
|test_ambiguity_association_20261025|Test1/Test2待核查标记与模型错误的两张表及关联分析|REPORT.md|
|expert_disagreement_20261025|训练内一维/三维分歧、纠正与拖错、配对及生物学线索|REPORT.md|
|_migration_20261025|复制文件哈希、历史路径对应关系和迁移验证|copy_manifest.json|

脚本均与其结果放在同一分析目录。推荐从仓库根目录使用原分析环境 `D:/anaconda/envs/pytorch_env/python.exe` 运行；数据比对所需 parasail 隔离依赖也已随原分析目录复制。

注意：dataset_comparison 中只有 alignment_to_train_v2.csv / alignment_to_known_v2.csv 是最终全局比对表；debug_invalid_stats_v1 中的初版无效，保留仅为追溯。历史来源清单中的 C 盘路径不应直接改写；当前副本的位置见迁移路径表，重新计算时脚本会按新目录生成相应产物。
