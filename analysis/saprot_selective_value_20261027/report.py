from pathlib import Path
import json
from collections import Counter
O=Path(__file__).resolve().parent;R=O.parents[1];rd=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
a=rd(O/'source_fourway_atlas.json');m=rd(O/'metrics.json');c=rd(O/'comparisons.json');g=rd(O/'gate_prediction_metrics.json');match=rd(O/'matched_control_summary.json');ver=rd(O/'verification.json');selections=rd(O/'selection_records.json');dec=rd(O/'decisions.json')
label={'old_masked':'旧版＋全掩码SaProt','old_3Di':'旧版＋近似三维SaProt','new_masked':'新版＋全掩码SaProt','new_3Di':'新版＋近似三维SaProt'}
lines=['# SaProt独立互补、受控调权重与样本选择验证','',
'结论：SaProt确实能单独答对部分完整GHXTox错题，但反向错误更多。当前小型选择器尚不能可靠把这种事后互补转为分类收益；增加序列/空间特征未稳定优于简单分数选择器或受控调权重。旧版＋近似三维SaProt受控调权重出现小幅正向线索（净少错11条），区间跨零，尚不足以确认提升或开展蒸馏。','',
'本轮只复用全部6387条保存OOF、缓存注释/特征及既有匹配，不训练神经网络、不编码新结构、不查新文献、不使用Test1/Test2；默认模型和标签均未改。所有“对/错”均相对原数据标签，不自动等于生物学真值。','',
'## 1. 两个模型各自对错：原判定标准','',
'GHXTox使用各自原始固定阈值（新版按行原折阈值），SaProt独立专家使用0.5。下表不是加权融合后的纠错表，也不是下文新校准/阈值选择下的四类表。','',
'| 模型组合 | G错、S对 | G对、S错 | 两者都错 | 两者都对 |',
'|---|---:|---:|---:|---:|']
for p,v in a.items():
    cl=v['classes'];lines.append(f"| {label[p]} | {cl['G_wrong_S_right']['n']} | {cl['G_right_S_wrong']['n']} | {cl['both_wrong']['n']} | {cl['both_right']['n']} |")
lines+=['','G=完整GHXTox，S=独立SaProt。每行四组互斥且合计6387。事后知道某条该信谁，不能据此在新样本上路由；四类标签不作为推理输入。两个模型都对也不能证明它们依赖相同信息。','',
'例如全掩码SaProt可单独答对旧版84/327、新版103/357条错题，约26%/29%；原错题的大部分它也答错。不能据这84/103条推算部署会净少错同样数量，因为还必须防止269/258条反向错误。','',
'## 2. 教师种子稳定性','',
'| 组合 | 潜在帮助组总数 | 两个SaProt小头种子都答对 | 反向错误组总数 | 两种子都答错 |',
'|---|---:|---:|---:|---:|']
for p,v in a.items():
    h,b=v['classes']['G_wrong_S_right'],v['classes']['G_right_S_wrong'];lines.append(f"| {label[p]} | {h['n']} | {h['both_SaProt_seeds_correct']} | {b['n']} | {b['both_SaProt_seeds_wrong']} |")
lines+=['','这里固定GHXTox，仅检查两个SaProt小头种子；不是两套独立编码器重复，也不是同一条肽跨折重复。每条肽只有一份原外层OOF，能够检查跨折的是规则/群体的表现。','',
'## 3. 不看评价答案，实际能否选对专家','',
'同一原始神经留出折内部，按原固定序列组两半划分：一半开发，一半评价，交换后覆盖全部6387条。开发半再按序列组三折产生校准器未见的分数，用来选比例和阈值；最终标量校准器仅拟合开发半，应用到评价半。','',
'- 不融合对照：允许同样的校准和开发阈值选择。',
'- 受控调比例：SaProt占比0、10%、…、100%；每个比例只在开发内层分数上选MCC阈值，再锁定应用到评价半。平分时优先小SaProt权重，0允许完全不用SaProt。',
'- 分数选择器：根据两个概率、两边距离各自阈值的距离和分歧，估计在意见不同时SaProt是否正确。',
'- 特征选择器：再加入长度、Cys比例、电荷密度、疏水性、芳香/Pro比例、pLDDT、结构token保留比例、空间疏水接触和电荷—疏水分布等缓存特征。',
'- 选择器均为预设C=0.1小逻辑回归，只以开发半内的分歧样本拟合；一致样本不被乱标为“SaProt无帮助”。评价时仅在专家分歧且估计概率>0.5时改用SaProt；不用真实评价标签选专家。','',
'### 同一新评价流程下的结果','',
'| 组合 | 不融合MCC | 受控调比例MCC | 仅分数选择MCC | 加序列/空间特征选择MCC |',
'|---|---:|---:|---:|---:|']
for p in a:
    lines.append(f"| {label[p]} | {m[p+'_baseline']['MCC']:.5f} | {m[p+'_weight_selected']['MCC']:.5f} | {m[p+'_score_gate']['MCC']:.5f} | {m[p+'_feature_gate']['MCC']:.5f} |")
lines+=['','这张表的不融合成绩与原默认OOF不完全相同，因为它额外执行了开发/评价分离的校准和阈值选择；应横向看同一行的增量，不与历史0.8733或此前固定融合的0.8694混比。所有原默认不变。两版原始内部训练/模型选择协议也不同，不能靠本表两行裸分排名架构。','',
'### 受控调权重与特征选择的纠错/添错','',
'| 组合及方法 | 纠错 | 添错 | ΔMCC | 95%区间 | 五折净少错 |',
'|---|---:|---:|---:|---|---|']
for x in c:
    if not x['baseline'].endswith('_baseline') or x['candidate'].endswith('_score_gate'):continue
    lo,hi=x['delta_MCC_95CI'];lines.append(f"| {x['candidate']} | {x['corrected']} | {x['harmed']} | {x['delta_MCC']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {'/'.join(map(str,x['fold_net_fewer_errors']))} |")
lines+=['','旧版近似三维调权重：纠31/添20，净少11条；五折4/1/1/5/0，无一折净添错，但ΔMCC约0.00415，95%区间[-0.00116,0.00975]跨零。这个候选的MCC0.87126也没有超过历史旧默认0.87330，但两者参数选择流程不同，不能用此差直接断言SaProt好坏。正向线索保留，不称已证实提升。','',
'新版近似三维调比例净纠错/添错43/43，漏判220→193、误报153→180，主要是两类错误交换；其AUPRC决策分数0.94872→0.95378有正向变化，仍须区分排名与分类。新版特征选择器纠19/添16，仅净少3条，五折0/8/-1/-3/-1，收益集中且区间跨零。没有特征选择器达到预设继续门槛，不能据少数成功派工声称生物机制已学会。','',
'概率经各自开发阈值中心化为决策分数，统一以0.5判定；路由输出也如此。表中AUPRC/AUROC反映这个完整决策流程的排序，分数不保证是校准的毒性概率，不能用作风险概率解释。','',
'## 4. 提前识别“该信SaProt”的能力','',
'| 组合 | 新评价流程中的分歧数 | SaProt正确占比 | 分数选择器AUROC | 特征选择器AUROC | 特征选择器实际选SaProt：对/错 |',
'|---|---:|---:|---:|---:|---|']
for p,v in g.items():
    f=v['feature_gate'];lines.append(f"| {label[p]} | {v['evaluation_disagreements']} | {v['helpful_prevalence']:.1%} | {v['score_gate']['AUROC_teacher_helpful']:.3f} | {f['AUROC_teacher_helpful']:.3f} | {f['selected_helpful']}/{f['selected_harmful']} |")
lines+=['','这里判的是“哪个专家正确”，不是直接判毒性。新版近似三维分数选择器AUROC约0.651，说明存在一点可预测线索，但其实际选SaProt8对/10错。增加结构等特征后AUROC约0.628、实际选19对/16错，尚未稳定转成分类收益。不能把AUROC高于0.5当成部署成功，也不能看这些结果后再到同一评价样本上调路由阈值。','',
'## 5. 整类生物学与同背景对照','',
'全部四类均保留缓存家族、机制候选句、二硫键/修饰注释覆盖、来源和特征，没有只挑一两条。按原标签分别比较潜在帮助/反向错误的特征及五折方向；门控特征系数同时控制模型分数等输入。','',
'- 部分疏水性、空间接触或电荷—疏水分布系数在单个配置中方向较一致，但配置之间方向可变，且完整特征选择器未提高留出收益，不能据系数讲共同机制。',
'- 本轮只复用旧的相近背景配对，不重复匹配、放宽标准或扩文献。原过滤结果50条记录包含重复匹配来源；去重后只有17对全球唯一case/control。不同配置可能复用同一对，不能当新增独立证据。',
'- 相应对照没有明确curated家族支持，部分仅名称类别代理或序列相似；覆盖有限，也未建立实验终点/条件一致，不能证明家族内机制。',
'- 明确二硫键/修饰注释分别单列；未缓存或未报告保留unknown，不当“没有”。膜作用候选排除已标否定警告后也仅是注释候选，不当统一溶血真值。片段与标签冲突仍是待核查标记，不自动视为错标。','',
'详细记录：annotation_class_coverage.json、class_feature_summary.json、gate_coefficient_summary.json和matched_control_summary.json。当前未形成可据以重写网络的已验证整类机制。','',
'## 6. 证据边界与决定','',
'所有神经预测沿用历史开发数据。旧模型原有全OOF校准/选择，新模型复用原序列OOF与内层选择的历史限制，不能靠新元模型的分组留出消除；因此本轮是条件于历史预测的回顾性研究，不是新严格嵌套或独立外部验证。没有新增同规模完整GHXTox集成对照，不能排除模型数量效应。','',
'2000次配对原序列组bootstrap，20个预设MCC对照同时报告95%与99.75%校正区间，全部保留。区间条件于现有模型、校准与选择，不含重新拟合的完整不确定性；较早研究和本次方法选择的全部研究自由度也未被20比较校正覆盖。','',
'当前决定：不部署本次融合/门控，不蒸馏，不根据真实错题路由或改标签。保留旧版近似三维受控调比例为弱正向候选；不继续在相同评价样本上扫门控结构、阈值或生物特征直到涨分。若继续SaProt，应先解决更独立的统一协议验证/更强互补证据，再决定是否值得新训练，而不是据本轮小幅点估计启动复杂网络。','',
'## 7. 完成与核验','',
f"全部320次标量校准和80次选择器拟合记录已保存。独立复算{ver['scalar_predictions_replayed']}条标量预测、{ver['gate_predictions_replayed']}条选择概率，最大差{ver['max_probability_difference']}；40次比例选择的440组比例/阈值候选用暴力矩阵法独立核对，四类25548个位置、指标与已存区间分位数通过核验。未重复训练神经网络。",'',
'protocol.json、input_manifest.json、source_fourway_atlas.json、all_sample_records.json、evaluation_predictions.npz、scalar_models.json、selection_records.json、gate_models.json、metrics.json、comparisons.json、verification.json保留全部过程；SAMPLE_LIST.md列出四类中全部潜在帮助与反向错误名单。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
rows=rd(O/'all_sample_records.json');byid={x['sample_id']:x for x in rows};sample=['# 两专家独立判断：全部潜在帮助与反向错误','', '相对于原标签和原操作点；不是固定融合后的纠错。完整四类及全部样本保存在source_fourway_atlas.json和all_sample_records.json。']
for pair,v in a.items():
    for cls,title in [('G_wrong_S_right','GHXTox错、SaProt对'),('G_right_S_wrong','GHXTox对、SaProt错')]:
        kk=v['classes'][cls]['sample_ids'];sample+=['',f'## {label[pair]}：{title}（{len(kk)}条）','', '| ID | 原标签 | 折 | 名称 | 序列 |','|---|---:|---:|---|---|']
        for k in kk:
            x=byid[k];name=(x['name'] or '未知').replace('|','/');sample.append(f"| {k} | {x['label']} | {x['fold']} | {name} | {x['sequence']} |")
(O/'SAMPLE_LIST.md').write_text('\n'.join(sample)+'\n',encoding='utf-8')
summary=['# 本轮结果：通俗版','', '**找到了一部分可以互补的错题，但目前还不能稳定地提前选对专家。**','', '| 做了什么 | 结果 |','|---|---|',
'| 两模型各自对错分类 | 已覆盖6387条，区分潜在帮助、反向错误、都错、都对 |',
'| 受控调比例 | 旧版近似三维组合有净少错11条的线索，但区间跨零，未确认提升 |',
'| 按样本选择专家 | 新版近似三维净少错3条，其他特征选择多数添错，未稳定胜过调比例 |',
'| 加生物物理特征 | 暂未让选择更可靠，不能据此宣称发现新机制 |',
'| 是否蒸馏或换默认 | 暂不；保留全部结果，不重复扫同一批评价题 |','', '[完整报告](REPORT.md)、[完整名单](SAMPLE_LIST.md)。']
(O/'SUMMARY.md').write_text('\n'.join(summary)+'\n',encoding='utf-8');print('REPORT, SUMMARY and SAMPLE_LIST written')
