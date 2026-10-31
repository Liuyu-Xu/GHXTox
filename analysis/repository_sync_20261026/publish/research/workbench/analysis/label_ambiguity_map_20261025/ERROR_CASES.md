# 争议样本分类与错误清单

FP/FN均相对原标签；分类只是证据状态，不是重新判标签。具体标签可重叠，主分类仅便于浏览。

| 样本 | 原标签 | 1D/3D/融合 | 主分类 | 依据 |
|---|---:|---|---|---|
| peptide_201 Ectatomin A | 1 | TP/FN/FN | 分子对象问题的具体案例/线索 | 原文活性对象是完整A+B双链毒素；本轮未找到孤立A链的直接毒性测定 |
| peptide_309 Ribosome-inactivating protein pleuturegin | 1 | FN/FN/FN | 分子对象问题的具体案例/线索 | 输入28 aa是38 kDa pleuturegin 的N端测序片段；原论文的翻译抑制实验对象为完整分离蛋白，未建立孤立28 aa的同等活性。 |
| peptide_622 Turripeptide Ici9.1 | 1 | FN/FN/FN | 分子对象问题的具体案例/线索 | Turripeptide Ici9.1；Fragment，离子通道毒性为同源推断，不能当作该精确短肽直接活性证据。 |
| peptide_643 Turripeptide OL11-like | 1 | FN/FN/FN | 分子对象问题的具体案例/线索 | Turripeptide OL11-like；Fragment，离子通道毒性为同源推断，优先检查实体和实验效应。 |
| peptide_2165 Phospholipase A2 CS24a | 0 | FP/FP/FP | 分子对象问题的具体案例/线索 | PLA2 CS24a 片段；酶活性功能注释含同源推断，不能视为该短序列的实测毒性。 |
| peptide_2653 Mytimycin | 0 | FP/FP/FP | 分子对象问题的具体案例/线索 | Mytimycin 标记 Fragment；原始研究描述部分表征的约6.2 kDa抗真菌肽，而保留输入只有33 aa。抗真菌也不是宿主无毒证据。 |
| peptide_3648 U10-ctenitoxin-Pr1a | 0 | FP/FP/FP | 分子对象问题的具体案例/线索 | 条目注释对小鼠和昆虫无毒，但序列标记 Fragment，功能证据指向数据库提交。不能当作合成该34 aa片段已测得无毒。 |
| peptide_3921 Putative serine protease | 0 | FP/FP/FP | 分子对象问题的具体案例/线索 | 疑似丝氨酸蛋白酶片段；已注释真菌细胞壁组分结合功能，不能据此证明孤立短肽无毒。 |
| peptide_4096 U1-pseudomyrmecitoxin-Pt1 subunit SS3 | 0 | FP/FP/FP | 分子对象问题的具体案例/线索 | Myrmexin 异二聚体中的 SS3 亚基；功能证据涉及异二聚体/复合物的抗炎活性，不是孤立 SS3 的毒性证明。 |
| peptide_4448 C-type lectin domain-containing protein 2 | 0 | FP/FP/FP | 分子对象问题的具体案例/线索 | 蛇毒 C 型凝集素蛋白的片段；没有恢复该短序列的独立毒性终点。 |
| peptide_5875 U9-ctenitoxin-Pk1a | 0 | FP/FP/FP | 分子对象问题的具体案例/线索 | 蜘蛛毒液蛋白 PKTx22C1 的片段；来源和家族相似性不能确定其宿主毒性。 |
| peptide_5902 U14-ctenitoxin-Co1c | 0 | FP/FP/FP | 分子对象问题的具体案例/线索 | 注释称脑室注射小鼠未见毒性，但为 Fragment 且功能依据为提交记录；不能外推到任意实验条件或孤立片段。 |
| peptide_251 H1C/H1U | 1 | TP/FN/FN | 化学形式证据冲突待确认 | 原摘要支持溶血/细胞效应，但UniProt N端甲酰化与Hemolytik Free记录冲突；原表单位/条件未全部核实，未证明修饰导致漏判。 |
| peptide_424 H2C/H2U | 1 | FN/FN/FN | 化学形式证据冲突待确认 | 原摘要支持溶血/细胞效应，但UniProt N端甲酰化与Hemolytik Free记录冲突；原表单位/条件未全部核实，未证明修饰导致漏判。 |
| peptide_33 Omega-hexatoxin-Hv2a | 1 | FN/FN/FN | 实验终点或作用背景需区分 | Omega-HXTX-Hv2a：昆虫钙通道选择性证据，不能无条件解释为人类毒性；原记录还含C端修饰。 |
| peptide_322 Neuropeptide Y1-like conopeptide | 1 | FN/FN/FN | 实验终点或作用背景需区分 | Cono-NPY1，C端酰胺化；注释/原始论文支持小鼠脑室给药后的行为效应，不是溶血或某个已确证受体机制。 |
| peptide_332 Conotoxin pc16a | 1 | FN/FN/FN | 实验终点或作用背景需区分 | pc16a 原论文报告所测动物、Na/K通道及烟碱受体实验未见活性；不能据此证明普遍安全，但正标签不应被当作这些终点的实测阳性。 |
| peptide_444 Nc3b | 1 | FN/FN/FN | 实验终点或作用背景需区分 | 原标签1且融合FN；原研究哺乳动物CC50/HC50均>300μM，另有昆虫作用。不同终点不能混为通用毒性阈值。 |
| peptide_5033 Natriuretic peptides A | 0 | FP/FP/FP | 实验终点或作用背景需区分 | 蛙利钠肽相关条目，血管活性/激素功能；记录还涉及前体和切割链，不能把生理活性等同于统一毒性。 |
| peptide_94 Lycotoxin I | 1 | TP/FN/FN | 直接效应支持的漏判候选 | 合成LycotoxinI C端酰胺化；兔红细胞室温1h，200μM约55%溶血；1D TP/3D FN/融合FN；不证明修饰导致错误。 |
| peptide_249 Na1b | 1 | FN/FN/FN | 直接效应支持的漏判候选 | 原文精确序列有HEK293活性与人红细胞溶血测量；融合FN；这支持研究该难例，不证明具体致错机制。 |
| peptide_463 Nc2a | 1 | FN/FN/FN | 直接效应支持的漏判候选 | 原文有较弱HEK293效应，HC50>250μM；融合FN；具体终点阈值与原标签生成链仍未知。 |
| peptide_1434 LyeTx I | 1 | TP/FN/FN | 直接效应支持的漏判候选 | LyeTxI为游离N端、酰胺化C端；兔红细胞37°C1h，ED50=130μM；1D TP/3D FN/融合FN；无同序列修饰因果对照。 |
| peptide_160 Lycotoxin II | 1 | TP/FN/FN | 该具体分子/终点证据不足 | 本轮没有确认其单独直接红细胞剂量曲线；C端直接确认；无本条溶血曲线 |
| peptide_806 unknown | 1 | FN/FN/FN | 该具体分子/终点证据不足 | 本轮短条目精确匹配和精确网页搜索未恢复可靠身份；保持未知，不按模型意见补标签。 |
| peptide_1392 Human xenin-25 | 1 | FN/FN/FN | 该具体分子/终点证据不足 | 序列匹配 human xenin-25；内源激素身份既不证明安全，也不证明原标签有误。优先恢复原始毒性依据。 |
| peptide_1623 ConoServer G9.1 | 1 | FN/FN/FN | 该具体分子/终点证据不足 | ConoServer G9.1；由前体推定成熟序列，序列证据为核酸层面，不能当作成熟肽已实测毒性。 |
| peptide_2726 Conotoxin pc16c | 0 | FP/FP/FP | 该具体分子/终点证据不足 | pc16c 的序列和C端酰胺化有记录，未恢复对应毒性测定；没有 Toxin 关键词不等于已证实无毒。 |
| peptide_9  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_11  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_18  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_31  | 1 | FN/TP/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_38  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_40  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_44  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_50  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_57  | 1 | FN/TP/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_58  | 1 | TP/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_59  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_65  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_66  | 1 | TP/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_76  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_78  | 1 | TP/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_79  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_90  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_93  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_97  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_99  | 1 | FN/TP/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_102  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_104  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_105  | 1 | TP/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_107  | 1 | TP/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_110  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_113  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_119  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_124  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_132  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_142  | 1 | FN/TP/FN | 尚未逐条归因 | fragment_reference_match；non_disulfide_modification_annotation |
| peptide_146  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_147  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match；non_disulfide_modification_annotation |
| peptide_151  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_156  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_162  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_171  | 1 | TP/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_172  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_174  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_175  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_178  | 1 | TP/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_179  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_185  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_187  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_193  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_195  | 1 | TP/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_196  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match；non_disulfide_modification_annotation |
| peptide_203  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_206  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_217  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_226  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_228  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_234  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_243  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_247  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_252  | 1 | TP/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_255  | 1 | FN/FN/FN | 尚未逐条归因 | interchain_annotation |
| peptide_257  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_259  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_276  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_289  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_298  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_307  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_315  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_320  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_324  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_326  | 1 | FN/TP/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_328  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_333  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_336  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_347  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_353  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_358  | 1 | FN/TP/FN | 尚未逐条归因 | fragment_reference_match；non_disulfide_modification_annotation |
| peptide_365  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_366  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_370  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_373  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_383  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_396  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_397  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_398  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match；non_disulfide_modification_annotation |
| peptide_401  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_418  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_420  | 1 | FN/TP/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_422  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_426  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_431  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_447  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_448  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_451  | 1 | TP/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_455  | 1 | FN/TP/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_458  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_460  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_461  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_462  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_470  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_486  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_605  | 1 | FN/TP/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_612  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_617  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_619  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_637  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_698  | 1 | TP/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_699  | 1 | FN/TP/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_706  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_709  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_721  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_730  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_733  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_736  | 1 | FN/TP/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_738  | 1 | TP/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_742  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_747  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_750  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_753  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_763  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_768  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_785  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_792  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_800  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_801  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_802  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_815  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_816  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_818  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_826  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_827  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_837  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_841  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_846  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_849  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_858  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_865  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_869  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_871  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_872  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_876  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_878  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_879  | 1 | FN/TP/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_880  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_882  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_884  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_886  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_891  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_892  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_900  | 1 | TP/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_907  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_956  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_983  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_988  | 1 | FN/TP/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1010  | 1 | FN/FN/FN | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_1091  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1107  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1109  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1148  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1175  | 1 | TP/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_1188  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_1271  | 1 | FN/TP/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1288  | 1 | FN/TP/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1350  | 1 | FN/TP/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1461  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1474  | 1 | TP/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1508  | 1 | FN/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_1526  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1529  | 1 | TP/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1551  | 1 | TP/FN/FN | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_1558  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1580  | 1 | FN/FN/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_1598  | 1 | FN/TP/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1604  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1610  | 1 | TP/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1731  | 1 | FN/TP/FN | 尚未逐条归因 | fragment_reference_match |
| peptide_1780  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1787  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1788  | 1 | FN/FN/FN | 尚未逐条归因 | identity_unresolved |
| peptide_1850  | 0 | FP/FP/FP | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_1854  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_1870  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_1892  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_1948  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_1951  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_1973  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_1998  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_1999  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_2011  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement；non_disulfide_modification_annotation |
| peptide_2098  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2233  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_2271  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2345  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2355  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_2380  | 0 | FP/FP/FP | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_2410  | 0 | FP/FP/FP | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_2478  | 0 | FP/TN/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2627  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2629  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2640  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2697  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2699  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2701  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_2707  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement |
| peptide_2728  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_2738  | 0 | FP/TN/FP | 尚未逐条归因 | cross_source_label_disagreement |
| peptide_2792  | 0 | FP/TN/FP | 尚未逐条归因 | cross_source_label_disagreement |
| peptide_2912  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2913  | 0 | TN/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2949  | 0 | FP/TN/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_2992  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3011  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3016  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement |
| peptide_3037  | 0 | FP/FP/FP | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_3116  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3117  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3134  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3138  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3141  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3148  | 0 | FP/TN/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3183  | 0 | TN/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3207  | 0 | FP/TN/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3218  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3247  | 0 | FP/TN/FP | 尚未逐条归因 | cross_source_label_disagreement |
| peptide_3266  | 0 | FP/TN/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3270  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3283  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match；cross_source_label_disagreement |
| peptide_3306  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement |
| peptide_3308  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3319  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3326  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3330  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement；non_disulfide_modification_annotation |
| peptide_3366  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3394  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3404  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match；cross_source_label_disagreement；interchain_annotation |
| peptide_3434  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3438  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement |
| peptide_3470  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3489  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3510  | 0 | FP/FP/FP | 尚未逐条归因 | identity_unresolved |
| peptide_3529  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement |
| peptide_3548  | 0 | TN/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3602  | 0 | FP/FP/FP | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_3687  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3695  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3699  | 0 | FP/FP/FP | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_3707  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3712  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3756  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3779  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_3789  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3814  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3843  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_3907  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement；non_disulfide_modification_annotation |
| peptide_3913  | 0 | FP/TN/FP | 尚未逐条归因 | cross_source_label_disagreement |
| peptide_3914  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement；non_disulfide_modification_annotation |
| peptide_3928  | 0 | FP/TN/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_4115  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement；non_disulfide_modification_annotation |
| peptide_4186  | 0 | FP/TN/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_4289  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_4292  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match；interchain_annotation |
| peptide_4315  | 0 | TN/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_4364  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement；identity_unresolved |
| peptide_4429  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match；cross_source_label_disagreement |
| peptide_4450  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_4509  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_4537  | 0 | FP/FP/FP | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_4540  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_4543  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match；cross_source_label_disagreement |
| peptide_4654  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match；non_disulfide_modification_annotation |
| peptide_4664  | 0 | FP/TN/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_4759  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_4760  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_4834  | 0 | FP/FP/FP | 尚未逐条归因 | identity_unresolved |
| peptide_4839  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_4875  | 0 | FP/TN/FP | 尚未逐条归因 | cross_source_label_disagreement |
| peptide_4895  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_4966  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_4992  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_4993  | 0 | TN/FP/FP | 尚未逐条归因 | cross_source_label_disagreement |
| peptide_5006  | 0 | FP/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_5011  | 0 | FP/FP/FP | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_5042  | 0 | FP/FP/FP | 尚未逐条归因 | cross_source_label_disagreement；non_disulfide_modification_annotation |
| peptide_5060  | 0 | FP/TN/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_5061  | 0 | FP/TN/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_5073  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_5091  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match；cross_source_label_disagreement |
| peptide_5160  | 0 | TN/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_5206  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_5207  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_5481  | 0 | FP/TN/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_5518  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_5541  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match；non_disulfide_modification_annotation |
| peptide_5700  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match；cross_source_label_disagreement；non_disulfide_modification_annotation |
| peptide_5708  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match；cross_source_label_disagreement |
| peptide_5726  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_5767  | 0 | TN/FP/FP | 尚未逐条归因 | 本轮未逐条归因，不能当可靠标签 |
| peptide_5771  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match；cross_source_label_disagreement |
| peptide_5853  | 0 | FP/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_5946  | 0 | TN/FP/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_6111  | 0 | TN/FP/FP | 尚未逐条归因 | non_disulfide_modification_annotation |
| peptide_6264  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_6357  | 0 | FP/TN/FP | 尚未逐条归因 | fragment_reference_match |
| peptide_6360  | 0 | TN/FP/FP | 尚未逐条归因 | fragment_reference_match |
