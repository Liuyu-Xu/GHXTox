# 历史三维残基输入与已有消融核查

直接读取 D:/dachuang2026/GHXTox-ziliao/best_model(1).pt：checkpoint 内含 config、model_state、epoch、训练记录；model 配置 aa_embedding_dim=32、group_embedding_dim=12、num_egnn_layers=3、graph_mode=hybrid、plm_embedding_dim=1280、chemical_site_branch=true、site_type_dim=8、chemical_site_layers=2、max_site_slots=2、use_interaction_types=true、use_hydrophobic_sites=true、exclude_same_residue_edges=false。默认公共推理使用 base_logits、单结构；不能因配置存在声称特征一定产生有效贡献。

当前源码 dual_modal.py 将 residue_features 输入 spatial_branch，也设置氨基酸嵌入和可选化学位点分支；源码与权重历史实现需进一步逐项核验，不能只凭名称认定 Cys 位置和配对被学习或已验证。

已有 runs/chemical_site_component_ablation/summary.json 为折0筛选，含去疏水通道、方向、化学相互作用等变体；模型选中 epoch 不同且对同折筛选，该结果是既有探索，不能作为完整五折或严格对照证据。已有 reports/topology_strict_oof_v1_final_result.md 在新构造的五折 backbone 上检验软Cys拓扑，没超过简单matching对照，不应当作历史checkpoint的消融或重复实验。

未完成：提取可靠的历史分折3D表征并按组隔离训练探针；未执行针对Cys与其他残基的受控遮蔽预测。full2795权重训练涉及训练数据，不能把其训练集输出当作无泄漏折外表示。下一步需要查清五个3D上游检查点、数据缓存与训练谱系，或重新训练可审计的折外结构模型，再做探针和遮蔽；训练成本尚未估算。不可用历史测试集调试。

## 五折检查点核验补充

根据 historical_selected_base_v1/index.json，五个候选 checkpoint 均在 GHXTox-ziliao/fusion_aware_residual_screen_package_20260809_v1/repository/runs 中存在，逐文件 SHA256 均匹配索引。直接读取 torch checkpoint，五折配置 chemical_site_branch=true、plm_embedding_dim=1280，每折都包含28个 chemical_site 参数张量；但仅有配置和参数不等于分支输出有效。

chemical_site_branch.residual_projection 的 weight/bias 最大绝对值：fold0 0.159958/0.114113，fold1 0.287568/0.269673，fold2 0/0，fold3 0.118733/0.112143，fold4 0.143660/0.130321。fold2 残差投影严格为零，化学分支对该路径没有非零残差贡献；不能理解为整个结构或残基分支关闭。归档 summary 确认 fold2_prediction_identity_verified，selection_status=prediction_identical_zero_chemical_fallback。各fold选择checkpoint的epoch为11、20、8、7、9。

注意它们是历史谱系选出的候选五折，不应直接宣称就是论文原始完整模型逐折权重；上游初始化谱系未全证实，选模也可能使用该折数据。对折2化学相关结果必须单独报告，不与其他折混为同一有效分支。尚未完成内部表征导出、探针、受控残基遮蔽；需要验证编码缓存和推理流程与各权重一致，不能声称这些分析已做。

## 现有推理复现记录与表示缓存（补充）

找到 raw_view_spatial_embedding_screen_package_20260809_v1：其五折 summary 报告 held 样本分别1271、1229、1273、1291、1323，原 base 分数的最大绝对差分别1.2652e-7、8.4502e-8、9.5603e-8、8.7881e-8、8.3828e-8，全部低于既定2e-6容差。这是该归档的历史复现记录，本轮读取记录，未重新运行推理。其只对五折覆盖子集逐折分别537、551、543、593、571条生成多构象相关特征。

五折 feature_cache.pt 每个存2795条、781维；检查代码可见前三个256维块是多构象的均值减单结构base、标准差和最大最小差，不是单结构原始残基或 pooled spatial embedding；不能直接将此缓存当作所需残基探针的输入。读取脚本发现通过 model output spatial_nodes 按mask做 mean+max 可以获取256维原始单结构池化表示，需在原始1.18GB处理数据及五个checkpoint上重新前向并匹配fold和结构，尚未执行。训练源谱系不全与fold2零化学残差限制仍然成立。

## 五折空间表示提取和 Cys 探针

本轮重新加载五个checkpoint，在各自留出折运行单结构前向，提取spatial_nodes的mask mean/max 256维，保存spatial_fold{0..4}.npz及extract_spatial.py。相对历史selected_base OOF概率最大差依次0.00001693、0.00001943、0.00001264、0.00019550、0.000000655；所有fold超先前2e-6容差者不得称严格复现，需查模型代码、数值环境或批次差异。没改变模型权重。

探索性探针以≥6 Cys为目标，先在每个外折留出样本内部、按序列组隔离70/30分训练探针和评价；输入长度对照 vs 长度+256维空间表示，固定逻辑回归。各折AUROC分别为长度0.647/0.595/0.569/0.582/0.622；加入空间表示0.995/0.991/0.998/0.997/0.992。意味着同一模型的空间表示可以解码高Cys信息，不证明毒性判断使用它、也不证明学习了Cys之间关联。输入本来包括残基信息，表现高并不意外；上游权重谱系不全且折内单次拆分，不视为独立验证。

另一项把其他四折模型的表示合并训练、预测当前折的探针，五折AUROC为0.912/0.659/0.979/0.196/0.500，高度不稳（spatial_probe_results.json），可能是不同模型表示坐标不一致或其他分布差异，不能解释为这些折缺少Cys信息。五个独立模型的隐藏向量不能默认轴对齐。尚未分析 Cys 位置、间隔或受控输入遮蔽；此次没有可靠证据证明预测依赖这些因素。

## Cys位置探针与化学分支遮蔽（续）

在每个fold的留出样本中，仅选择Cys>=2，按序列group划分探针训练/测试（单次70/30，seed42）。Ridge alpha100：对照使用长度和Cys数量；实验再增加256维空间池化表示。目标是首Cys相对位置与首尾Cys相对跨度，并非二硫键。首位置MAE五折对照0.1183/0.1216/0.1218/0.1417/0.1136，加入空间表示0.0973/0.1110/0.0904/0.1019/0.1216；跨度对照0.1160/0.1243/0.1166/0.1314/0.1256，加入空间表示0.1029/0.1217/0.0855/0.1108/0.1275。二者均4折降低1折升高；单次拆分及上游谱系不全，探针可读出不说明预测使用，不是键连或机制证据。实现probe_positions.py及position_probe_results.json。

对冻结模型进行推理时整体化学分支残差置零，与同一程序未遮蔽前向比较，阈值为三维历史阈值0.677819。五折分类翻转6、20、0、14、2，合计42；相对标签纠正24、新增错误18。fold2残差原本为零，翻转0符合预期。遮蔽全部化学信息，不是Cys特异消融，不是重训消融或完整融合评价；人为遮蔽可能造成分布偏移，不证明因果机制。实现chemical_off.py、chemical_off_results.json。

前向与历史值微差最大fold3约0.000196；已有历史归档的复现用不同代码/环境和批次契约，本轮未确定原因，因此不能声称历史逐数精确复现。比较遮蔽采用本轮相同程序的基线减少数值路径混杂，但仍需核查具体差异来源。

## Cys 化学位点受控遮蔽

本轮对3173条有Cys化学位点且有足够非Cys化学位点的留出折样本运行同一固定三维模型；每条分别遮蔽所有有位点Cys残基的chemical_site_mask，以及随机遮蔽相同残基数量的非Cys位点，后者按sample_id哈希固定重复3次取绝对概率变化平均。保留其它所有输入（Cys身份、残基特征、PLM、结构坐标仍在），所以仅测试化学位点通道依赖，不是抹去Cys或重训消融。匹配残基数，不匹配位点槽数、残基类别或位置；不代表真实突变或因果证据。未使用测试集。

各折 mean absolute probability change：Cys遮蔽 0.00726、0.00621、0、0.00334、0.00116；随机非Cys 0.00156、0.00220、0、0.00384、0.00155。fold0/1 Cys较大，fold3/4较小，fold2均为0；方向跨折不一致，尚不能说模型稳定更依赖Cys化学位点。浮点概率接近饱和导致部分变化显示0。样本选择和类别背景有偏，未做显著性或家族控制。详细逐样本与分层结果存 cys_site_mask_predictions.csv / cys_site_mask_summary.json；实现 cys_site_mask.py。

## 重点群体与匹配位点数/位置的遮蔽

从每折带Cys化学位点的样本中选择存在一一对应非Cys对照者：控制每个被遮蔽残基的有效位点槽数相同、相对序列位置差≤0.2，按位置差最小进行一对一匹配；3149条符合。比较每条分别遮蔽所有有化学位点的Cys残基和其匹配对照，不改变序列、结构坐标、PLM，随机对照和本轮匹配对照不可等同。此为事后设计的探索实验，仅做一个最优位置匹配，不同残基的化学种类、空间环境不能一致。

低Cys原有毒仅55条符合，实际均为含1个Cys的样本；0 Cys不能参与该实验，不能推广至全部262条。三维概率绝对变化均值 Cys/对照0.00377/0.00267；高Cys原无毒239条 0.004603/0.004578。前者fold0/1/3偏Cys、fold4偏对照、fold2均零；后者fold1偏Cys，fold0/3/4均值偏对照，fold2均零。全体3149条0.00358/0.00120，但按折统计fold0/1/3偏Cys、fold4偏对照、fold2均零。均值受概率饱和和少数大变化影响，未按家族控制或计算区间，不代表对毒性预测有净收益。

遮蔽位点数受mask而非原始Cys数量限定；序列邻近不等于三维邻近；分组依据原标签，不能推理时调用标签。来自既有模型分折候选且上游来源未完整证明独立；仅支持敏感性探索，非突变或新模型验证。见 cys_mask_matched.py、cys_matched_mask_predictions.csv、cys_matched_mask_summary.json、cys_matched_mask_coverage.json。

## 三维概率加局部/空间特征：轻量增量探索

复用历史训练OOF三维概率、标签与已有局部/空间代理，按既有序列组五折，训练四折/预测一折的正则逻辑回归。A仅三维分数logit；B=A+长度+Cys数量；C=B+局部正电/疏水、疏水接触、正电疏水接触、空间分离、非局部接触、回转半径及平均pLDDT。缺失值中位数、标准化均仅训练折拟合；C=0.1、概率阈值0.5为固定试验，不用测试集。分数输入与历史三维内部特征不完全独立，不能声称全是新增信息；历史上游分折模型、校准与选择非完整嵌套，训练折OOF预测可能依赖评估折训练信息，故不是严格无泄漏评估。未重训三维主干，也不是完整模型融合。

|方法|MCC|AUROC|AUPRC|总错|低Cys正漏判|高Cys负误报|
|---|---:|---:|---:|---:|---:|---:|
|A|0.8350|0.9570|0.9233|426|161|58|
|B|0.8285|0.9583|0.9276|441|191|75|
|C|0.8293|0.9580|0.9261|439|187|74|

C对B纠正17、新错15，低Cys正纠6新错2、高Cys负纠4新错3；fold总错误B→C分别67→67、108→105、84→89、86→85、96→93。按fold内序列组2000次配对bootstrap，MCC差C−B 95%区间[-0.0034,0.0050]，仅条件于当前预测，不包含训练随机性或上游泄漏。两目标组都是单标签，只比较错误数，不能计算各自AUROC。A使用单独校准和固定0.5，与原模型3D阈值不同，不能当作原有阈值或完整GHXTox指标。事后多次探索，未做多重比较调整。

结果不支持简单增加这些代理能稳定提升，不能排除其他非线性残基级交互或充分重训有效。代码 feature_increment_probe.py；逐样本 feature_increment_predictions.csv；结果及区间 feature_increment_results.json 和 feature_increment_bootstrap.json。
