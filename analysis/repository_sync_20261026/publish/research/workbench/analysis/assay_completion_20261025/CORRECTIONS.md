# 对上一轮的更正

上一轮将训练peptide_288/P83248（Oxyopinin-2a）认定为具有直接哺乳动物溶血剂量曲线的正确阳性对照。进一步核对原研究Peptide Synthesis和Results的Circular Dichroism and Peptide Synthesis段落发现，两处明确写合成Oxki1和Oxki2b，后续简写为Oxki1/Oxki2。

因此，不能把后续Oxki2曲线直接归给2a，也不能归给2c。本轮撤回2a的直接终点对照认定，将其列为待确认背景。不是据此认定2a无毒。

原研究：[Corzo等，2002](https://doi.org/10.1074/jbc.M200511200)，印刷23628及23630–23631页。原研究全文索引保存在sources/oxy_identity_correction_search.json。原数据库可能在家族成员间重复了活性注释，本轮未证明注释生成过程，不能当作已确认数据库错误。

上轮9条短名单应改为5条具具体直接哺乳动物终点记录、4条背景。5条中原标签阳性4条均为融合FN，另1条是原标签阴性的Nc1a。这是选择性病例列表，不能据此计算机制类别的漏判率。目前没有已核实、实验背景匹配的正确阳性对照。

已同步更正build_comparison.py、endpoint_comparison.json、verification.json；旧报告保留历史正文并在开头加更正提示。原标签、训练OOF预测及主数据集未改。
