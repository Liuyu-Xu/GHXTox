"""Assemble the interpretable report from verified outputs; no input model execution."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_atlas import OUT, report, save

d=pd.read_csv(OUT/'all_training_atlas.csv',float_precision='round_trip')
pairs=pd.read_csv(OUT/'matched_pairs.csv')
tests=pd.read_csv(OUT/'matched_feature_tests.csv')
s=json.loads((OUT/'audit_summary.json').read_text(encoding='utf-8'))
risk=pd.read_csv(OUT/'cys_stratum_risks.csv')
secondary=json.loads((OUT/'sensitivity_summary.json').read_text(encoding='utf-8'))
evidence=json.loads((OUT/'evidence_findings.json').read_text(encoding='utf-8'))
report(d,s,pairs,tests)
original=(OUT/'REPORT.md').read_text(encoding='utf-8')
lead='''
## 本轮最值得关注的发现

模型对“富 Cys 的毒肽”表现很好，却明显更容易漏掉低 Cys 毒肽；对无毒肽，富 Cys 又更容易被误报。这个双向现象比“某个几何特征有显著性”更值得优先追查。它支持家族/来源简化线索这一候选解释，但不能证明 Cys 是致毒原因，也不能证明所有错误来自同一种机制。

|真实标签|Cys 数|样本数|错误数|错误率|序列组 bootstrap 95% 区间|
|---|---|---:|---:|---:|---|
'''
for _,r in risk.iterrows():
    label='有毒（漏检）' if r.label else '无毒（误报）'
    cys='≥4' if r.cys_ge4 else '<4'
    lead+=f'|{label}|{cys}|{int(r.n)}|{int(r.errors)}|{100*r.error_rate:.2f}%|[{100*r.group_bootstrap_ci_low:.2f}%, {100*r.group_bootstrap_ci_high:.2f}%]|\n'
lead+='''
![不同 Cys 分组的错分率](cys_error_rates.png)

在全部错误中，低 Cys 毒肽占漏检的 175/192；富 Cys 无毒肽占误报的 76/135。分母是训练 OOF 中对应真实标签分组，不能将这些比例直接当作部署场景发生率。

## 两个优先追查的假设

1. **家族或来源线索替代了毒性判别。** 数据里有毒组与无毒组的 Cys 分布很不相同。模型可能学会了“更像常见富 Cys 毒素”的识别规则，因而漏掉无 Cys 的膜活性肽或毒液肽，并误伤部分富 Cys 非毒肽。首先回查原始 accession、真实家族和终点，建立同家族或组成相近的困难对照。下一轮可固定验证 Cys/长度简单基线，并检验完整模型对低 Cys 毒肽和富 Cys 无毒肽的增量价值；本轮没有训练该基线。
2. **一个二分类标签混合了不同效应、修饰和证据强度。** 下面的样本分别匹配到溶血肽、NPY-like 毒液肽以及人体肽身份。应先恢复实验终点、剂量和修饰，区分标签问题与表征缺失；不能看到错误就提高其训练权重。

## 为什么暂不添加空间纠错模块

第一次组成匹配得到 75 对 FP/TN 和 76 对 FN/TP。在预测 C-alpha 接触图中，“正电荷残基—疏水残基接触富集”在两类错误里都偏低（探索性 q 分别约 0.020 和 0.041）。低序列邻域相似度与漏检、种子分歧与错误也有关。

但进一步加入平均 pLDDT、低置信残基比例、序列邻近代理和种子分歧卡钳，并严格匹配 Cys 数后，只得到 12 对 FP/TN 和 8 对 FN/TP；空间指标没有通过这组次级比较的 BH 校正。样本过少意味着既不能确认空间规律独立于这些因素，也不能证明空间规律不存在。两类错误同向变化更像“困难样本属性”的候选信号，尚不能直接当作毒性增减规则。

因此本轮不训练纠错专家、不改变阈值、不声称性能提高。完成的是统计小实验；独立的预测修正实验需要先补齐严格嵌套的基础预测流和终点证据。

## 代表性错题的文献核查

以下只恢复外部序列身份和文献线索，不代表找到了这条记录最初进入 GHXTox 数据集时的标签依据。未修改任何标签。

|样本|本地标签与预测|回查结果|解释边界|
|---|---|---|---|
|peptide_424|有毒；融合概率约 0.0073|精确序列匹配 Hemolytik2 的 H2C 溶血肽条目，关联原始论文 PMID 18752624|支持优先调查膜相关效应与剂量；不能据此给全部低 Cys 漏检赋相同机制|
|peptide_322|有毒；融合概率约 0.0072|精确序列匹配 ConoNPY-Bt1，ConoServer 记录芋螺毒液来源及 C 端酰胺化|无 Cys 毒液肽是家族简化规则的候选例外；具体受体不能由同源命名推断，序列输入也未表达酰胺化|
|peptide_1392|有毒；融合概率约 0.0055|精确序列匹配 GPCRdb 的 human xenin-25|人体肽身份既不证明安全，也不证明标签错误；优先追查原始毒性测量及来源，不能强迫模型背标签|

来源：[H2C 序列与效应记录](https://webs.iiitd.edu.in/raghava/hemolytik2/Browse/cytotoxic.php)、[H2C 原始研究](https://pubmed.ncbi.nlm.nih.gov/18752624/)；[ConoNPY-Bt1 序列及修饰](https://www.conoserver.org/?id=7536&page=card&table=protein)、[Cono-NPY 原始研究](https://doi.org/10.1093/abbs/gmq042)；[xenin 精确序列](https://gpcrdb.org/ligand/8890/info/)、[xenin 原始研究](https://doi.org/10.1016/S0021-9258(18)41670-5)。

保留数据的阴性构建采用数据库排除毒素/过敏原关键词，这并不等价于每条阴性都经过一致的毒性实验。因此“FP”在此首先指相对原始数据标签的错误，不等于生物学上已确证的错误。[ToxMSRC 数据来源](https://pmc.ncbi.nlm.nih.gov/articles/PMC12512130/)、[CAPTP 原始数据说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC11654579/)。

## 已完成与尚未完成

- 已完成：全体训练 OOF 概率复核、错误清单、组成匹配、配对统计、次级敏感性分析、20 条代表性证据回查队列，以及 3 条具体身份/文献线索恢复。
- 已验证：样本身份、折/组对齐、原始融合指标、阈值边界决策、配对卡钳与组不复用、空间指标旋转平移不变性。
- 尚未完成：全部错误的原始证据追溯、真实家族/来源匹配、严格嵌套重训练及新独立集验证。没有用缺失证据自动补机制标签。

'''
(OUT/'REPORT.md').write_text(original.replace('## 核心结果',lead+'## 核心结果',1),encoding='utf-8')
# Preserve queue fields and append external identities in a separate derived output.
queue=pd.read_csv(OUT/'evidence_review_queue.csv')
lookup={c['sample_id']:c for c in evidence['cases']}
for col in ['external_name','identity_source','primary_paper','supported','not_established','implication']:
    queue[col]=[lookup.get(i,{}).get(col,'') for i in queue.sample_id]
queue['external_identity_status']=['identity_and_literature_lead_found' if i in lookup else 'not_recovered' for i in queue.sample_id]
save('evidence_review_queue_annotated.csv',queue)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
for ax,label,title in zip(axes,[0,1],['Non-toxic labels: false-positive rate','Toxic labels: false-negative rate']):
    g=risk[risk.label==label].sort_values('cys_ge4')
    rates=g.error_rate.to_numpy()*100
    yerr=np.vstack([rates-g.group_bootstrap_ci_low.to_numpy()*100,g.group_bootstrap_ci_high.to_numpy()*100-rates])
    ax.bar([0,1],rates,color=['#40668A','#B65C43'],width=.55,yerr=yerr,capsize=5)
    for x,(_,row) in enumerate(g.iterrows()):ax.text(x,row.group_bootstrap_ci_high*100+1.3,f'{int(row.errors)}/{int(row.n)}\n{row.error_rate:.1%}',ha='center',fontsize=10)
    ax.set_xticks([0,1],['Cys < 4','Cys >= 4']);ax.set_title(title,fontsize=11);ax.set_ylim(0,57)
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
axes[0].set_ylabel('Error rate (%)')
fig.suptitle('GHXTox archived training OOF: a cysteine-associated error pattern',fontsize=12)
fig.text(.5,.015,'Exploratory association; 95% sequence-group bootstrap intervals. Not independent validation.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(OUT/'cys_error_rates.png',dpi=180);plt.close(fig)
print('Final report, annotated evidence queue and figure saved.')
