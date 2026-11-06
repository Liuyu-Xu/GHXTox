from pathlib import Path
import json
O=Path(__file__).resolve().parent;R=O.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
m=read(O/'metrics.json');cs=read(O/'comparisons.json');sub=read(O/'subgroup_metrics.json');check=read(O/'verification.json')
lines=['# 两版完整GHXTox与SaProt：固定融合试验','',
'结论：已完成全部6387条保存OOF上的固定融合。20%和50%这两种预设SaProt占比未显示稳定的MCC净收益，不能把轻量基线上的收益直接转入完整GHXTox。这个结果只否定当前固定方案的继续门槛，不证明SaProt整体无用，也未排除受控权重选择的可能性。','',
'本轮无新神经训练、无新编码、未读取Test1/Test2；运行模型默认未改，尚未蒸馏。','',
'## 比较方式','',
'两版分别按同样规则做概率校准，再加入独立SaProt专家；主试验80%完整GHXTox+20%SaProt，次试验各50%。SaProt分别使用全结构token掩码和近似3Di，不扫描其他权重。校准采用原外层留出折内部按序列组固定两半交叉拟合：一半拟合、另一半评价，再交换；所有神经专家都未以该原折样本做梯度训练。相同外层分组不保证内部训练协议相同。','',
'采用StandardScaler与C=1逻辑回归，仅一个模型分数作输入；阈值固定0.5。新旧完整模型各自原始阈值先作中心平移，标准化后不改变该单变量校准的表达能力。校准和权重方案均在查看融合结果前写入protocol.json。','',
'## 先区分原始默认与同等校准基线','',
'| 版本 | 原始MCC | 本次同等校准MCC | 原始FN/FP | 校准FN/FP |','|---|---:|---:|---|---|']
for v,label in [('old','旧完整GHXTox'),('new','去额外Transformer新版')]:
    a,b=m[v+'_raw'],m[v+'_cal'];lines.append(f"| {label} | {a['MCC']:.5f} | {b['MCC']:.5f} | {a['FN']}/{a['FP']} | {b['FN']}/{b['FP']} |")
lines+=['','旧版历史OOF的校准/阈值已使用全OOF标签；新版是不同内层角色和逐折冻结参数。不能用0.87330与0.86159直接排名架构；主要看同版同等校准基线前后的变化。旧版额外校准本身MCC下降，因此不能把融合相对原始分数的全部变化归因SaProt。所有参考分数均保留。','',
'## 全部固定融合结果','',
'纠错和添错均相对于本版同等校准基线；主试验占比20%，50%为预设对照。','',
'| 版本 | SaProt输入 | SaProt占比 | MCC | ΔMCC | AUPRC | FN/FP | 纠错/添错 | 五折净少错 |',
'|---|---|---:|---:|---:|---:|---|---|---|']
for c in cs:
    a=m[c['candidate']];lines.append(f"| {c['version']} | {c['mode']} | {c['weight_SaProt']:.0%} | {a['MCC']:.5f} | {c['delta_MCC']:+.5f} | {a['AUPRC']:.5f} | {a['FN']}/{a['FP']} | {c['corrected']}/{c['harmed']} | {'/'.join(map(str,c['fold_net_fewer_errors']))} |")
lines+=['','old=历史完整模型，new=去额外Transformer模型，masked=全结构token掩码，3Di=近似结构token。结构token近似性限制沿用旧SaProt输入审计，不能当原生全原子Foldseek。','',
'主试验中，旧版masked净多错2条、3Di多错3条；新版masked多错5条，3Di仅少错1条。新版3Di的排序AUPRC改善，但FN由212增至218、FP由141降至134，存在漏判/误报交换；不能以净少1条概括为毒性识别整体提高。50%占比也没有形成MCC净收益。','',
'## 不确定性与稳定性','',
'| 比较 | ΔMCC 95%区间 | 八比较校正99.375%区间 | 两个独立小头种子的20%融合ΔMCC |',
'|---|---|---|---|']
for c in cs:
    a,b=c['delta_MCC_95CI'],c['delta_MCC_99_375CI'];lines.append(f"| {c['candidate']} | [{a[0]:+.5f}, {a[1]:+.5f}] | [{b[0]:+.5f}, {b[1]:+.5f}] | {' / '.join(f'{q:+.5f}' for q in c['single_teacher_seed_w20_delta_MCC'])} |")
lines+=['','2000次配对原序列组bootstrap，条件于保存预测和拟合校准器，不含重训或后续模型选择的不确定性。主方案均未过继续门槛；新版3Di20%的区间跨零，两个教师种子一负一正。旧版3Di50%在当前条件区间下明显退化。没有融合具有校正后的正MCC证据。种子诊断共享编码器与数据，不能当独立生物学重复。','',
'## 既有盲区','',
'| 配置 | 低Cys原正类漏判（421条） | 膜作用注释原正类漏判（76条） |',
'|---|---:|---:|']
for a in ['old_cal','old_masked_w20','old_3Di_w20','new_cal','new_masked_w20','new_3Di_w20']:
    d=sub[a];lines.append(f"| {a} | {d['lowCys_positive']['FN']} | {d['membrane76_annotation_only']['FN']} |")
lines+=['','低Cys正类在四个20%候选中均未改善。76条是旧缓存注释集合，含语义/终点不一致风险，只作诊断，不代表统一溶血机制金标准。没有显示整类盲区的可靠补强，不能根据零星纠错命名新的生物模块。','',
'## 来源审计与限制','',
'- 全部ID、序列、标签、外层fold、group与既有SaProt一一对齐；20个SaProt checkpoint的train/test ID与群隔离检查通过。',
'- 新结构使用独立train/validation/calibration/test内层角色；SaProt固定20轮使用其余四个外层折。训练样本量和模型选择预算不同，不能称两者为完全同预算方法比较。',
'- 旧完整OOF有全局标签校准/选择和多轮开发使用。新版复用旧序列OOF，其内层calibration行对应的原序列预测器可能训练过当前外层折；原序列模型训练成员缺少全部历史checkpoint可追溯证据，不能声称整套新融合严格嵌套独立。新加折内校准不能清除历史开发影响。',
'- 没有同等新增完整GHXTox成员或六序列种子的保存控制，不能排除增加集成成员的作用；轻量四成员对照不能冒充本实验的同规模控制。',
'- 本轮是冻结历史预测上的开发诊断，不是新的外部验证，未修改默认、标签或推理路由。','',
'## 后续：若要调比例，应怎样做','',
'用户提出其他融合比例是否可能有效：有可能，当前两个固定比例不足以排除。暂不把本轮负结果解释成所有权重必然无效。下一候选是另行冻结的受控权重选择，而非在全部OOF上扫描并报告最高分。','',
'建议沿用原留出折内分组评价分区，权重与校准/阈值只能用相应开发部分确定，外面的评价部分不参与；将SaProt权重0明确纳入候选，并让不融合基线获得同样校准及阈值选择机会。若再切内层校准/选择分区，需预先确定规则以控制小样本不稳定。两版、两类SaProt和全部结果保留，不能看评价分数再换网格。Test1/Test2仍不参与调权重。此受控权重选择尚未执行，不能现在承诺一定提高。','',
'当前不进入蒸馏，不部署本轮融合。若受控选择仍无稳定增益，再暂停直接融合；不要为了分数反复尝试比例。','',
'## 产物与核验','',
'protocol.json、input_manifest.json、source_audit.json、calibration_models.json记录协议和来源；metrics.json、comparisons.json、fold_metrics.json、subgroup_metrics.json保存全部结果；all_sample_predictions.json/crossfit_predictions.npz包含全部6387条背景及八候选角色，SAMPLE_LIST.md列出20%方案全部纠错/添错。',
f"独立复算{check['scalar_calibrator_predictions_replayed']}条校准预测、{check['all_sample_role_checks']}个角色；最大概率差{check['max_probability_difference']}。八种固定混合、全部指标、已存bootstrap分位数及输入哈希核验通过。没有重复神经训练。"]
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
rows=read(O/'all_sample_predictions.json');sl=['# 20%SaProt固定融合：全部纠错与添错','', '相对各自同等校准基线；开发后诊断名单，不能用于推理时按真实标签选择专家。']
for c in cs:
    if c['weight_SaProt']!=.2:continue
    arm=c['candidate'];sl+=['',f'## {arm}：纠错{c["corrected"]}，添错{c["harmed"]}','', '| ID | 原标签 | 折 | 结果 | 序列 |','|---|---:|---:|---|---|']
    for row in rows:
        if row[arm+'_role'] in ['corrected','harmed']:
            sl.append(f"| {row['sample_id']} | {row['label']} | {row['fold']} | {row[arm+'_role']} | {row['sequence']} |")
(O/'SAMPLE_LIST.md').write_text('\n'.join(sl)+'\n',encoding='utf-8')
print('REPORT and SAMPLE_LIST written')
