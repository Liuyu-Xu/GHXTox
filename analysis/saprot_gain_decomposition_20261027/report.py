from pathlib import Path
import json
O=Path(__file__).resolve().parent;R=O.parents[1];rd=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
f=rd(O/'factorial_results.json');cs=rd(O/'factorial_comparisons.json');null=rd(O/'conditional_shuffle_diagnostic.json');bio=rd(O/'whole_class_biology.json');coverage=rd(O/'reused_control_coverage.json');verify=rd(O/'verification.json')
primary=f['old_3Di'];names={'A_G_tG':'维持原对照：原分数＋原开发阈值','B_G_tF':'只换成融合方案选出的阈值','C_mix_tG':'只加入SaProt分数，保留对照阈值','D_mix_tF':'SaProt分数＋融合方案阈值'}
lines=['# 净少错11条：SaProt分数与阈值的交叉拆解','',
'结论：这11条不是纯靠阈值，也不是只加入SaProt就能得到。相同保存预测上，只换阈值净少错4条，只加SaProt且保持对照阈值反而多错1条，两者配合净少错11条。SaProt分数与具体样本对齐有描述性线索，但整体增量区间跨零、排序改善很小，尚不能归因于已发现的生物机制。','',
'本轮复用所有冻结参数与预测，没有新模型拟合、权重搜索、神经训练、编码或Test1/Test2使用。以旧版近似3Di作为事后跟进主对象，同时保留其余三组合，不能当新独立实验。','',
'## 1. 一个必须说明的前提','',
'上一轮“不融合”对照也已经在开发部分选择了阈值，不存在只给融合方案调阈值却不给基线的情况。现在只是把以前在开发部分选定的两套阈值交叉套用到两套分数，作解释诊断；没有看本轮评价结果再挑阈值。只换阈值的配置不是现在新选出的部署方案。','',
'## 2. 全6387条上的四格结果','',
'| 旧版近似三维组合的诊断操作 | MCC | 错误总数 | 相对原对照净少错 | FN/FP |',
'|---|---:|---:|---:|---|']
base=primary['metrics']['A_G_tG'];baseerr=base['FN']+base['FP']
for k,m in primary['metrics'].items():
    err=m['FN']+m['FP'];lines.append(f"| {names[k]} | {m['MCC']:.5f} | {err} | {baseerr-err:+d} | {m['FN']}/{m['FP']} |")
lines+=['','也可以沿两条路径读：','',
'- 先换阈值：少错4条，再加入SaProt分数又少错7条，合计11条。',
'- 先加SaProt：多错1条，再换适合融合分数的阈值少错12条，合计仍是11条。','',
'因此分数变化和阈值变化存在配合，不能说11条全是SaProt的新知识或全是调阈值。两路径平均的代数分摊为分数3、阈值8，仅是计账方式，不意味着能把11条因果划成“3条结构知识、8条阈值”。MCC分摊同样不是生物因果解释。','',
'## 3. 全部31条纠错与20条添错','',
'对最终确实翻转的51条，检查单独改变阈值或分数是否也能产生同一翻转；四类互斥。','',
'| 翻转方式 | 最终纠错31条 | 最终添错20条 |','|---|---:|---:|',
'| 只换阈值可翻转，只加分数不行 | 14 | 7 |',
'| 只加SaProt分数可翻转，只换阈值不行 | 10 | 6 |',
'| 两种单独操作都可以翻转 | 4 | 4 |',
'| 单独操作均不行，需要两者一起 | 3 | 3 |','',
'上述只是最终改变样本的解释；全部正确背景和仍错样本也参与四格比较，不能只看成功的31条。31条纠错中21条为原无毒标签、10条为原有毒标签；20条添错为11条无毒、9条有毒。净收益主要是减少10条误报，漏判仅少1条，不是大幅补强毒肽识别。','',
'每条样本还保存精确决策距离恒等式：新距离−旧距离 = w×(SaProt概率−GHXTox概率) − (新阈值−旧阈值)。标签仅用于事后区分纠错/添错，不进入该距离或推理路由。','',
'## 4. 排名是否变好了','',
'| 不依赖单个阈值的指标 | 原GHXTox校准分数 | 混合分数 |','|---|---:|---:|']
rank=primary['rank_only'];lines+=[f"| AUPRC | {rank['G_AUPRC']:.6f} | {rank['mixture_AUPRC']:.6f} |",f"| AUROC | {rank['G_AUROC']:.6f} | {rank['mixture_AUROC']:.6f} |",'',
'这里直接用同一保存校准概率及其混合分数，不做逐阈值中心化，AUPRC与上一轮中心化决策分数表口径不同，差异不是重新训练所得。当前AUPRC仅约+0.00021，AUROC略降，不能说总体毒性排序大幅增强。','',
'## 5. SaProt是否只是一个可随便替换的分数','',
'预设200次描述性对照：在同一原折×评价半内，按GHXTox概率分10个等人数箱，仅在箱内打乱SaProt概率；原GHXTox分数、已选权重和阈值全部保持。不用标签构造箱或排列，不重新拟合。','',
'| 组合 | 真实对应MCC | 打乱对应后平均MCC | 打乱MCC的95%范围 |','|---|---:|---:|---|']
for p,d in null.items():
    lo,hi=d['conditional_shuffle_MCC_95_range'];lines.append(f"| {p} | {d['actual_MCC']:.5f} | {d['conditional_shuffle_MCC_mean']:.5f} | [{lo:.5f}, {hi:.5f}] |")
lines+=['','旧版近似三维真实对应0.87126，打乱均值0.86270；200次打乱均未达到真实值。它支持“具体分数对应关系有作用”的描述，不能把200次当独立验证或报告为正式p值。粗分箱仍会破坏GHXTox箱内细排序及家族相关，原序列群交换性也未保证，所以不能仅凭此证明新增独立信息，更不能证明是三维生物机制。','',
'## 6. 31/20整类生物学与相近背景','',
'复用全部原注释、特征、来源及同标签正确背景，未重新查单个案例或放宽配对标准。','',
'| 原标签分层 | 纠错数 | 添错数 | 纠错明确二硫键 | 添错明确二硫键 | 纠错/添错缺完整缓存 |','|---|---:|---:|---:|---:|---|']
for lab in ['0','1']:
    a,b=bio[lab]['corrected'],bio[lab]['harmed'];lines.append(f"| {lab} | {a['n']} | {b['n']} | {a['explicit_disulfide']} | {b['explicit_disulfide']} | {a['full_feature_record_unavailable']}/{b['full_feature_record_unavailable']} |")
lines+=['','合计明确二硫键：纠错8/31、添错3/20。但纠错有完整缓存14条、添错只有5条；在有缓存者中分别8/14、3/5，不能把全样本计数差解释为二硫键机制。缺失记录全部保留unknown，不是没有二硫键或修饰。','',
'家族与名称线索分散，明确curated家族条目在各标签/结果层内大多各1条，没有足量同家族纠错类别。原有膜作用/通道候选注释也不是统一实验终点：有毒纠错10条中膜作用肯定候选3、通道4；有毒添错9条中通道2。既有否定警告已排除，仍不能由少数注释推出溶血或靶点机制。长度、Cys、pLDDT和空间电荷/疏水等整类中位数、缺失数和全部正确背景见whole_class_biology.json，属于描述而非因果控制。','',
f"既有相近背景匹配复用后只有{coverage['unique_case_control_pairs']}对，覆盖{coverage['covered_corrected_cases']}/31条纠错、{coverage['covered_harmed_cases']}/20条添错；3对都对应同一条纠错，含1对明确家族及2对名称代理，不能代表完整类别或当3个独立成功案例。无足够同背景对照支持共同机制。",'',
'## 7. 其余组合全部保留','',
'| 组合 | 原对照错误 | 只换阈值错误 | 只加分数错误 | 两者一起错误 | 最终净少错 |','|---|---:|---:|---:|---:|---:|']
for p,d in f.items():
    mm=d['metrics'];err=[mm[k]['FN']+mm[k]['FP'] for k in ['A_G_tG','B_G_tF','C_mix_tG','D_mix_tF']];lines.append(f"| {p} | {err[0]} | {err[1]} | {err[2]} | {err[3]} | {err[0]-err[3]:+d} |")
lines+=['','没有因为主候选正向就隐藏其他配置。新版近似三维主要交换漏判与误报，净少错为0；两种全掩码候选净添错。','',
'## 8. 不确定性与当前决定','',
'旧版近似三维总ΔMCC约+0.00415，本次2000次配对群bootstrap的95%区间约[-0.00122,0.00949]；在相同融合阈值下加入SaProt相对只换阈值约+0.00259，95%区间[-0.00209,0.00723]，都跨零。20个预设对照的99.75%区间保存在factorial_comparisons.json。本次随机种子与前轮不同，因此描述区间略变，点预测、31/20名单和参数完全未变。','',
'本轮是已知正向候选后的解释性跟进，不是新确认；区间条件于历史预测/校准/选择，不含重训及整个研究过程的不确定性。旧全OOF校准、新序列OOF复用、候选多次开发等限制不因新的交叉替换消失。','',
'当前不部署、不蒸馏、不据这11条设计新生物模块。保留旧版近似三维受控调比例作为弱正向候选；若继续，优先在冻结方案后核查/准备定义一致、训练不重叠的独立验证面板，再决定是否投入新训练。不要继续在同一批评价题上扫描权重、阈值或挑少数机制案例。独立验证不是重构训练集，也不要求自行做生物实验，但当前没有新验证结果。','',
'## 文件与复算','',
'protocol.json、input_manifest.json记录本轮预设和源哈希；factorial_results.json/factorial_comparisons.json保存全部四格及区间；old3Di_corrected31_harmed20.json、SAMPLE_LIST.md为完整原31/20及来源；whole_class_biology.json、reused_matched_controls.json为整类/对照缓存。',
f"独立复算{verify['four_factor_decisions_replayed']}个判定、{verify['sample_margins_replayed']}条距离恒等式、{verify['changed_subtypes_replayed']}个翻转分类，最大恒等误差约{verify['max_margin_identity_error']:.2g}。原31/20名单、全部冻结参数、指标、已存区间分位数及打乱汇总复算通过；新模型拟合次数0。"]
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
cases=rd(O/'old3Di_corrected31_harmed20.json');sl=['# 旧版近似三维候选：原31纠错/20添错完整名单','', '使用原冻结参数，只作诊断；所有标签为原数据标签。','', '| ID | 角色 | 原标签 | 折 | 归因类别 | 名称 | 来源 |','|---|---|---:|---:|---|---|---|']
cn={'threshold_alone_only':'只换阈值可翻转','score_alone_only':'只加分数可翻转','either_alone_sufficient':'两者单独均可','both_changes_required':'需要配合'}
for x in cases:
    sources='; '.join(x['source_urls']);sl.append(f"| {x['sample_id']} | {x['role']} | {x['label']} | {x['fold']} | {cn[x['subtype']]} | {(x['name'] or '未知').replace('|','/')} | {sources} |")
(O/'SAMPLE_LIST.md').write_text('\n'.join(sl)+'\n',encoding='utf-8')
summary=['# 净少错11条：通俗结果','', '**需要SaProt分数与合适阈值配合，但还没证实稳定收益或新生物机制。**','', '| 做法 | 错误数 | 比原对照少错多少 |','|---|---:|---:|','| 原对照 | 343 | — |','| 只换阈值 | 339 | 4 |','| 只加SaProt，阈值不变 | 344 | -1 |','| SaProt与新阈值一起 | 332 | 11 |','',
'原对照也已经在开发部分选过阈值，四格都是解释诊断，没有新调参。','',
'- 这11条主要是少10条误报、少1条漏判。',
'- 打乱SaProt分数与样本的对应后表现下降，但这个对照不能证明新生物机制。',
'- 全部31条纠错、20条添错已检查；缓存缺失不均、同背景对照覆盖仅1条纠错，尚无可靠共同机制。',
'- 当前保留候选，不部署/蒸馏；若继续，优先冻结后做独立验证。','', '[完整报告](REPORT.md)、[全部31/20名单](SAMPLE_LIST.md)。']
(O/'SUMMARY.md').write_text('\n'.join(summary)+'\n',encoding='utf-8');print('REPORT, SUMMARY and SAMPLE_LIST written')
