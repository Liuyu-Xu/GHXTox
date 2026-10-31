"""Build an endpoint-specific evidence sidecar; preserve original labels/predictions."""
from pathlib import Path
import json,hashlib

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
INPUT=ROOT/'analysis/membrane_expansion_20261025/endpoint_comparison.json'
ATLAS=ROOT/'analysis/expert_disagreement_20261025/training_expert_atlas.csv'
before=hashlib.sha256(ATLAS.read_bytes()).hexdigest()
samples=json.loads(INPUT.read_text(encoding='utf-8'))
unknown='未确认，不能用其他肽/实验填充'
terminal={
 'peptide_249':('未独立确认','游离酸端为合成方法推断','原文2-chlorotrityl resin；未找到逐条明确末端声明','推断，非直接确认'),
 'peptide_3912':('未独立确认','游离酸端为合成方法推断','同上；表格质量与天然肽共洗脱支持身份但不单独证明全部修饰','推断，非直接确认'),
 'peptide_463':('未独立确认','游离酸端为合成方法推断','同上','推断，非直接确认'),
 'peptide_444':('未独立确认','游离酸端为合成方法推断','同上','推断，非直接确认'),
 'peptide_94':('未见额外N端修饰声明','C端酰胺化','原文Primary Structures及TableII，质谱比较','C端直接确认；N端不填无修饰真值'),
 'peptide_160':('未见额外N端修饰声明','C端游离酸','原文Primary Structures及TableIII，质谱比较','C端直接确认；无本条溶血曲线'),
 'peptide_1434':('游离N端，H-','C端酰胺化，-NH2','2010原文印刷139页序列；137页Rink-amide合成','原文直接确认'),
 'peptide_288':('未独立确认','本条未独立确认','合成Oxki2b-OH的末端声明不可归给2a','实验对象归属已更正'),
 'peptide_652':('未独立确认','本条未独立确认','不得借用2b的末端或曲线','实验对象未确认'),
}
sources={
 'P2021':{'url':'https://doi.org/10.1016/j.bcp.2021.114693','local':'analysis/assay_evidence_20261025/sources/Na1b_passages.json','level':'NCBI BioC原文机器提取；未视觉核对PDF','locations':'Methods offset14460/20073/21390；Table1 offset70913；Table3 offset72913'},
 'LY1998':{'url':'https://doi.org/10.1074/jbc.273.4.2059','local':'analysis/assay_completion_20261025/sources/remaining_scope_search.json','level':'出版商原文索引；直连403','locations':'Primary Structures/TableII/III；Hemolysis Assay；Fig7'},
 'L2010':{'url':'https://doi.org/10.1007/s00726-009-0385-x','local':'analysis/assay_completion_20261025/sources/LyeTx2010.pdf','level':'完整原文PDF；方法137页及Fig3的140页已渲染视觉核实','locations':'PDF第3页/印刷137页；第5页/139页；第6页/140页Fig3'},
 'L2018':{'url':'https://doi.org/10.3389/fmicb.2018.00667','local':'analysis/assay_completion_20261025/sources/LyeTx2018_passages.json','level':'NCBI BioC原文文本','locations':'Hemolytic Assay offset12715；Results26932；Fig7 caption28097'},
 'OX2002':{'url':'https://doi.org/10.1074/jbc.M200511200','local':'analysis/assay_completion_20261025/sources/oxy_identity_correction_search.json','level':'原研究全文索引；合成方法与结果段交叉核对','locations':'印刷23628页Peptide Synthesis；23630–23631页Results，明确Oxki2b'}
}
records=[]
def add(sid,endpoint,organism,result,exposure,temperature,dose,source,location,status='有直接测量记录',notes=''):
 records.append(dict(record_id=f'A{len(records)+1:02}',sample_id=sid,endpoint=endpoint,assay_object=organism,result=result,exposure_minutes=exposure,temperature=temperature,reported_dose_series=dose,source_id=source,location=location,evidence_status=status,notes=notes))
vals={249:('4.5–5.9','39.9–61.5'),3912:('12.6–13.5','28.6–48.2'),463:('48.2–56.9','>250'),444:('>300','>300')}
for n,(cc,hc) in vals.items():
 sid='peptide_'+str(n)
 add(sid,'细胞活性/细胞毒性','HEK293 ATCC CRL-1573，人源细胞系',{'metric':'CC50','raw':cc,'unit':'μM'},1440,'37°C；5%CO2','方法记0.02–50 μg/mL；三倍稀释','P2021','Methods cytotoxicity；Table3','有直接测量记录，剂量覆盖待核','24h暴露后加10μM resazurin再3h；读数560/590nm。方法梯度与部分高μM结果/下限不相容，不能假定逐肽梯度已补齐。CC50按活性降低代理定义，不当作直接计数死亡。'+('正文对Nc1a概括与Table3不完全一致，保留原表值。' if n==3912 else ''))
 add(sid,'溶血','人红细胞',{'metric':'HC50','raw':hc,'unit':'μM'},60,'37°C','方法未明确列出逐肽浓度梯度','P2021','Methods hemolysis offset21390；Table3','有直接测量记录','0.5×10^8细胞/mL；405nm；1%Triton X-100阳性/1%DMSO in PBS阴性；两次独立实验，每次重复孔。大于号为未达半数效应，非绝对无毒。')
add('peptide_94','溶血','兔红细胞',{'metric':'溶血比例','raw':'200 μM约55%；<30 μM未检出','unit':'%及μM'},60,'室温，未给精确°C','6.25–200 μM','LY1998','Hemolysis Assay；Fig7','有直接测量记录','合成Lycotoxin I；PBS pH7.0；570nm；去离子水阳性、PBS阴性。不为II填同样数值。')
add('peptide_160','溶血','未确认本条单独测定',None,None,unknown,unknown,'LY1998','摘要及Hemolysis Assay逐肽范围','本轮未找到本条直接溶血实验','本条有天然肽序列和抗微生物实验，不能将I的哺乳动物实验扩展给II。')
add('peptide_1434','溶血','兔红细胞',{'metric':'ED50','raw':'1.3×10^-4','unit':'M','converted_value':130,'converted_unit':'μM'},60,'37°C','方法仅写不同浓度；Fig3显示对数曲线，不从坐标反推精确梯度','L2010','印刷137页Haemolytic assay；140页Fig3','完整PDF及关键页视觉核实','PBS pH7；405nm；1%Triton X-100阳性。原文剂量点未逐项列出，不填完整梯度。')
add('peptide_1434','溶血','羔羊红细胞',{'metric':'ED50','raw':'42.54','unit':'μmol/L','converted_value':42.54,'converted_unit':'μM'},60,'37°C','Fig7A：0.35–44.8 μmol/L；Methods总述另写0.17–91.25 μmol/L','L2018','Methods Hemolytic Assay；Fig7A','原研究中原版LyeTxI对照组的直接测量','记录原版A组，非删His且N端乙酰化的I-b/B组。保留方法总范围与图示组范围的差别。跨研究差异不能全归因物种。')
for sid in ['peptide_288','peptide_652']:
 add(sid,'溶血','本条测定对象未确认',None,None,unknown,unknown,'OX2002','Peptide Synthesis；Results代表肽选择','撤回/不采用家族曲线归属','合成段及结果段均明确Oxki2b；后续简写Oxki2。不得归给2a/2c。原文通用方法的羊/猪/豚鼠、室温1h、1.6–50μM只属于被选代表肽测定，未写进本条确定字段。')

byid={r['sample_id']:r for r in samples}
for sid,(nt,ct,basis,stat) in terminal.items():
 byid[sid].update(n_terminal_state=nt,c_terminal_state=ct,chemical_evidence=basis,chemical_status=stat,assay_record_ids=[r['record_id'] for r in records if r['sample_id']==sid])
payload={'date':'2026-10-25','scope':'9条训练代表样本的实验条件旁表，不构建新训练标签','sources':sources,'samples':samples,'assays':records}
(OUT/'conditions.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
table=['# 实验条件总表','', '原标签与预测保留在conditions.json；下表只列实验依据。空缺明确保留，不借用其他肽结果。','', '| 样本 | 对象/终点 | 浓度或结果 | 时间/温度 | 修饰与证据状态 |','|---|---|---|---|---|']
for a in records:
 s=byid[a['sample_id']];r=a['result'];v=(r['metric']+' '+r['raw']+' '+r['unit']) if r else '本条未确认'
 time=f"{a['exposure_minutes']} min" if a['exposure_minutes'] is not None else '未确认'
 table.append(f"| {s['name']} ({a['sample_id']}) | {a['assay_object']}；{a['endpoint']} | {v} | {time}；{a['temperature']} | {s['c_terminal_state']}；{a['evidence_status']} |")
(OUT/'CONDITIONS_TABLE.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
assert len(samples)==9 and len(records)==14
assert set(byid)=={a['sample_id'] for a in records}
assert len({a['record_id'] for a in records})==14
assert all(a['result'] is None for a in records if a['sample_id'] in ['peptide_288','peptide_652','peptide_160'])
assert all(a['source_id'] in sources for a in records)
assert all(a['original_label_unchanged'] for a in samples)
after=hashlib.sha256(ATLAS.read_bytes()).hexdigest();assert before==after
(OUT/'verification.json').write_text(json.dumps({'sample_count':9,'endpoint_rows':14,'new_original_LyeTxI_assay':True,'oxy2a_direct_control_withdrawn':True,'original_labels_and_predictions_unchanged':True,'atlas_sha256_before':before,'atlas_sha256_after':after,'unresolved_fields_preserved':True},ensure_ascii=False,indent=2),encoding='utf-8')
print('Verified 9 samples, 14 endpoint records, source links, identity sidecar and unchanged frozen input.')
