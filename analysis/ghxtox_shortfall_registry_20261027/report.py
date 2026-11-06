from pathlib import Path
import json
O=Path(__file__).resolve().parent;R=O.parents[1];rd=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
regs=rd(O/'category_registry.json');reg={x['key']:x for x in regs};version=rd(O/'version_manifest.json');rows=rd(O/'all_sample_shortfall_index.json');byid={x['sample_id']:x for x in rows};verify=rd(O/'verification.json')
selected=['physical_0','physical_1','physical_2','physical_3']
def rate(r,v):
    s=r['stats'][v];return f"{s['wrong']}/{s['n']}（{s['error_rate']:.1%}）" if s['n'] else '无样本'
short=['# GHXTox短板清单：通俗版','',
'**先补哪类题，已经可以明确；为什么错、用什么补，还需要候选按这些类别证明。** 本轮复用已有记录，仅补旧/新预测流联表，没有重训或重做配对。','',
'## 优先研究的两类','',
'| 类别 | 总数 | 旧OOF错题 | 新OOF错题 | 两份预测都错 | 两份都对的比较池 |','|---|---:|---:|---:|---:|---:|']
for key in selected[:2]:
    r=reg[key];j=r['joint_modes'];short.append(f"| {r['title']} | {r['population']} | {r['stats']['old_archived']['wrong']} | {r['stats']['new_noTransformer']['wrong']} | {j['both_prediction_streams_wrong']} | {j['both_right']} |")
short+=['','两份预测共257条共同错题，其中215条属于上面两类。全部6387条仍保留，这不是按Cys筛掉其他样本。共同错只是描述性交集，不等于多种子或独立验证稳定。','',
'## 应该怎么研究','',
'| 优先级 | 任务 | 候选方案必须证明什么 |','|---|---|---|',
'| 1 | 低Cys共同漏判151条：挂接膜作用、通道/受体、凝血等已有线索，保留213条正确池 | 能识别这类非典型毒肽，而非只认Cys；少漏判且不明显增加误报 |',
'| 2 | 富Cys共同误报64条：先分原负类证据，再研究骨架/家族捷径，保留357条正确池 | 不把富Cys当有毒；又不损害真正原正类 |',
'| 3 | 在前两类内部选择有证据的具体机制子类 | 多样本、同背景支持，不能只凭注释或一个案例 |',
'| 4 | 候选在目标类和全体背景上稳定纠错后 | 再决定自适应融合、门控或蒸馏，候选不限SaProt |','',
'正确池不是严格配对的实验对照；富Cys原无毒类当前没有满足指定状态的直接测量缓存，不代表自然界没有实验。低Cys也不是单一溶血机制。','',
'[完整报告](REPORT.md)、[37类逐样本清单](category_registry.json)、[候选能力要求](CANDIDATE_BRIEF.md)、[重点错题与正确池](SAMPLE_POOLS.md)。']
(O/'SHORTFALLS.md').write_text('\n'.join(short)+'\n',encoding='utf-8')

lines=['# 完整GHXTox：按版本、证据和正确对照组织的短板清单','',
'本轮完成统一清单，而非重复试模型。保留全部6387条，复用旧327错题、已有类别/语义、注释与配对，补上新版357条错误在相同类别中的表现和交集；不重新配对、重新跑旧特征检验或训练。','',
'## 1. 两份预测流到底是什么','',
'| 预测流 | 身份 | 判定点 | FN/FP | 错题数 |','|---|---|---|---:|',
'| old_archived | archived operating-point旧OOF，来自all_training_linked；不是任意full2795部署流 | 0.47022256531284556 | 192/135 | 327 |',
'| new_noTransformer | 交接包S3＋去额外Transformer结构专家OOF，内部协议不同 | 各行原先冻结阈值 | 209/148 | 357 |','',
'两份流ID、序列、标签、原外层折和序列组已逐一对齐。它们共同错257条、旧错新对70条、旧对新错100条、都对5960条。由于历史校准、模型选择、内层预算不同，不能用这些裸分或迁移计数证明架构优劣；这是一份寻找短板的描述性交集，不是多个完整模型种子重复或新的独立评价。','',
'## 2. 完整物理背景分层','',
'| 类别 | 全部条数 | 旧错误/总数 | 新错误/总数 | 共同错 | 共同对 |','|---|---:|---|---|---:|---:|']
for k in selected:
    r=reg[k];j=r['joint_modes'];lines.append(f"| {r['title']} | {r['population']} | {rate(r,'old_archived')} | {rate(r,'new_noTransformer')} | {j['both_prediction_streams_wrong']} | {j['both_right']} |")
lines+=['','这四类互斥且覆盖全体。旧统计421/175、445/76等只做一致性核对，新增的是新版按类/共同错误关联。主短板分别是低Cys原正类漏判和富Cys原负类误报；215/257条共同错误属于这两类。但Cys只是分类线索，不是实际生物机制，也不是新的纳入筛选门槛。','',
'## 3. 把机制线索挂回短板，而非把它们当真值','',
'下列都是原正类的已有注释/语义子集；类别重叠不可相加，序列组不等于生物家族。','',
'| 已有候选线索 | 总数 | 旧漏判 | 新漏判 | 两份共同漏判 | 两份共同正确 | 在低Cys151共同漏判中的成员数 |','|---|---:|---:|---:|---:|---:|---:|']
lowshared=set(reg['physical_0']['joint_error_ids']);union=set()
for key in ['membrane_or_lysis_label1','ion_channel_label1','receptor_or_signaling_label1','hemostasis_or_proteolysis_label1','antimicrobial_label1','membrane_semantic_affirmative_annotation_not_gold']:
    r=reg[key];members=set(r['stats']['old_archived']['wrong_ids']+r['stats']['old_archived']['right_pool_ids']);overlap=lowshared&members
    if key!='membrane_semantic_affirmative_annotation_not_gold':union|=overlap
    lines.append(f"| {r['title']} | {r['population']} | {r['stats']['old_archived']['wrong']} | {r['stats']['new_noTransformer']['wrong']} | {r['joint_modes']['both_prediction_streams_wrong']} | {r['joint_modes']['both_right']} | {len(overlap)} |")
lines+=['',f"低Cys151条共同漏判中，上述前五个宽功能类别联合覆盖{len(union)}条，剩余{len(lowshared-union)}条没有这五类已缓存命中；后者是注释未知/未命中，不能断言没有相应机制。",'',
'膜类别76条包含弱/否定/间接/条件性效应，仍不是统一溶血实验集合。本表“肯定文字38条”是已有语义子层，不同于此前去片段/冲突后的另一个38条集合，不能混用ID或错误数；肯定文字本身也不保证指定完整分子的直接测量。抗微生物活性不能自动等同宿主毒性，通道/受体作用也不自动等同细胞死亡。','',
'## 4. 正确比较池、旧配对和证据缺口','',
'| 主短板 | 两份都正确的同类池 | 旧流错题中有既有同标签对照者 | 其中类内对照覆盖 | 明确家族配对 |','|---|---:|---:|---:|---:|']
for key in selected[:2]:
    r=reg[key];bc=r['existing_background_control_coverage']['old_archived'];cc=r['existing_control_coverage']['old_archived'];lines.append(f"| {r['title']} | {r['joint_modes']['both_right']} | {bc['unique_error_cases_covered']} | {cc['unique_cases_with_any_old_control']} | {bc['curated_family_edges']} |")
lines+=['','对照池只是同标签/同类正确背景，不是严格匹配或同终点实验对照。既有151组成对照和27家族/名称代理对照保留原规则、跨折状态和来源；没有新的宽松匹配。类内覆盖和较宽背景覆盖分开保存，名称代理不算明确家族；同家族正确对照允许跨Cys类别用于检查捷径，不会因Cys不同直接丢弃。','',
'| 主短板 | 全类片段标记 | 全类跨源标签冲突 | 有指定直接测量状态的缓存 | 共同错题中片段标记/标签冲突 |','|---|---:|---:|---:|---|']
for key in selected[:2]:
    r=reg[key];e=r['evidence'];ss=[byid[k] for k in r['joint_error_ids']];frag=sum(x['fragment_flag_not_confirmed_label_error'] for x in ss);conf=sum(x['cross_source_conflict_not_confirmed_label_error'] for x in ss)
    lines.append(f"| {r['title']} | {e['fragment_flags']} | {e['label_conflicts']} | {e['has_cached_direct_measurement']} | {frag}/{conf} |")
lines+=['','直接测量状态只按当前all_training_linked中有结果且明确“有直接测量记录”的旁表字段计数；其他历史人工证据表口径可能不同，不把这些计数当作完整文献覆盖或严格分子/终点一致认证。原负类缺缓存不等于没有实验，也不等于实测安全。片段、相反标签、未命中均不是确定错标，不删样本、不改标签或统一加权。','',
'## 5. 证据等级与已失败方案','',
'清单用两个层次理解：A表示明确的历史统计短板，不表示致错机制已证实；B表示有指定终点/身份支持的候选；C表示注释或标签/对象疑点；D表示某解释受到反例或其具体候选未过门槛。没有任何一项在本轮升级为已确认的新生物机制。','',
'| 已有方向 | 当前结论 | 后续不能怎样重复 |','|---|---|---|']
for d in rd(O/'failed_attempt_ledger.json'):lines.append(f"| {d['direction']} | {d['result']}（{d['scope']}） | {d['do_not_repeat']} |")
lines+=['','独立MLP＋3D-KD/MPCE等旁线的正证据仍保留，但不属于本表两份完整GHXTox预测流，也未在本轮证明能解决151/64这两类短板，不能搬用分数当本轮成功。','',
'## 6. 候选方案按能力匹配，不按名字匹配','',
'优先级和验收条件见CANDIDATE_BRIEF.md、candidate_requirements.json。先核查低Cys共同漏判151条的多机制背景，要求候选能减少这类漏判并保住正确/负类；随后研究富Cys原负类64条共同误报的证据层次及骨架/家族辨别。膜作用候选属于其中一个子方向，不等同全部低Cys。','',
'完整数据和比较池始终保留，不把151/64变成只训练难例的新数据集。候选需要展示同类纠错、添错、全体FN/FP、来源/近亲隔离和选择/评价分离；门控、融合或蒸馏在候选有可靠增量之后才设计。推理时不能使用“这条原来错了”、原标签、数据库来源ID或新样本未知的机制注释。','',
'## 7. 本轮交付与下一具体任务','',
'本轮交付37个分层/证据类别及其完整分母、两版错题/正确池、257共同错误、旧配对覆盖、证据等级、失败方案及候选能力要求。没有启动候选搜索或新模型试验。','',
'下一项如继续，先基于151/64清单制作候选可复现性与终点匹配表：优先查已有模型/公开权重是否真正覆盖相关宿主效应或同骨架负类，不先重训、不预设SaProt。若只支持抗菌或曲线回归、数据训练重叠不明、无法保持同背景正确池，则不进入融合/蒸馏。候选选择和确认阶段仍需更独立评价；历史OOF/测试的反复开发限制保留。','',
'## 文件与核验','',
'SHORTFALLS.md为通俗入口；category_registry.json含全部37类和ID池；all_sample_shortfall_index.json保留6387条；shared_error_samples.json保留257交集；SAMPLE_POOLS.md列出151/64与213/357正确池；existing_control_edges.json保留关联配对及是否跨类别/跨折；candidate_requirements.json、failed_attempt_ledger.json记录能力与失败边界。',
f"独立核查{verify['all_sample_id_sequence_label_fold_group_checks']}条ID/序列/标签/折/群、{verify['categories_replayed']}类的完整错误/正确池和交集，以及{verify['existing_control_edges_rechecked']}个跨类别重复引用的旧对照边。847是分类引用次数，不是847个新配对；未做新匹配、显著性检验、训练或Test1/Test2分析。类别成员继承旧索引快照，后续细注释覆盖另存，不因新缓存偷偷改变旧类别。"]
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

brief=['# 候选解决方案：先通过短板匹配，再进入集成','', '当前只定义能力要求，没有挑选新模型、下载权重或训练。候选不限SaProt。']
for d in rd(O/'candidate_requirements.json')['priorities']:
    brief+=['',f"## 优先级{d['priority']}：{d['target']}",'',d['capability'],'']
    if d.get('inputs_allowed'):brief.append('允许输入：'+d['inputs_allowed'])
    brief+=['', '最低要求：']+[f'- {x}' for x in d['minimum_evidence']]+['','已知限制：'+d['known_limits']]
brief+=['','## 统一记录表','', '| 候选/资产 | 对应哪类短板 | 真实训练终点 | 输入能否复现 | 训练来源/近亲重叠 | 同类正确池与全体对照 | 是否进入小试 |','|---|---|---|---|---|---|---|',
'| 待核查，不预定模型 | 151低Cys共漏/64富Cys共误或其有证据子类 | 必须区分宿主毒性、溶血、抗菌、受体作用等 | 序列/结构/化学形式是否齐全 | 可追溯后再判断 | 纠错与添错一起报告 | 本轮尚未开始 |','',
'先查已有本地结果和公开可复现资产，不重复同类失败小头。只有目标类别和全体背景均有稳定净收益、且不是靠标签/来源捷径，才另行设计受限融合、门控或蒸馏。']
(O/'CANDIDATE_BRIEF.md').write_text('\n'.join(brief)+'\n',encoding='utf-8')

sample=['# 主短板的共同错题与正确比较池','', '共同错仅指两份不同协议预测流的交集；正确池是同标签/同类背景，不是实测金标准或严格配对。原数据与全部6387条仍保留。']
for key in selected[:2]:
    r=reg[key]
    for field,title in [('joint_error_ids','两份都错'),('both_right_control_pool_ids','两份都正确的比较池')]:
        kk=r[field];sample+=['',f"## {r['title']}：{title}（{len(kk)}条）",'', '| ID | 原标签 | 折 | 名称 | 片段标记 | 跨源标签疑点 |','|---|---:|---:|---|---|---|']
        for k in kk:
            x=byid[k];sample.append(f"| {k} | {x['label']} | {x['fold']} | {(x['name'] or '未知').replace('|','/')} | {x['fragment_flag_not_confirmed_label_error']} | {x['cross_source_conflict_not_confirmed_label_error']} |")
(O/'SAMPLE_POOLS.md').write_text('\n'.join(sample)+'\n',encoding='utf-8')
print('SHORTFALLS, REPORT, CANDIDATE_BRIEF and SAMPLE_POOLS written')
