"""Read frozen training atlas and join a manually checked endpoint ledger.
No network, dataset mutation, relabeling, fitting, or test-set prediction access.
Run from any directory with the project Python interpreter.
"""
from pathlib import Path
import csv, json, hashlib

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ATLAS=ROOT/'analysis/expert_disagreement_20261025/training_expert_atlas.csv'
before=hashlib.sha256(ATLAS.read_bytes()).hexdigest()
with ATLAS.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==6387
by_id={x['sample_id']:x for x in rows}
assert len({x['sequence'] for x in rows})==len(rows)

# Eligibility follows exact tested entity and assay scope, not model correctness.
# Core here means a direct endpoint record, not matched experimental protocols.
ledger=[
 (249,'Na1b','P0DSM4','core','人HEK293/人红细胞','CC50 4.5–5.9 μM；HC50 39.9–61.5 μM','24h/1h；37°C','2021 Table1/3；Methods','10.1016/j.bcp.2021.114693','原文BioC文本；PDF未视觉复核','末端状态未独立确认'),
 (3912,'Nc1a','P0DV15','core','人HEK293/人红细胞','CC50 12.6–13.5 μM；HC50 28.6–48.2 μM','同上','2021 Table1/3；Methods','10.1016/j.bcp.2021.114693','原文BioC文本；PDF未视觉复核','原标签0，实验有活性；原标签判定终点/标准未追溯；正文概括与Table3不完全一致，保留表值待图像复核'),
 (463,'Nc2a','P0DV16','core','人HEK293/人红细胞','CC50 48.2–56.9 μM；HC50 >250 μM','同上','2021 Table1/3；Methods','10.1016/j.bcp.2021.114693','原文BioC文本；PDF未视觉复核','细胞毒性较弱；不能称强溶血肽'),
 (444,'Nc3b','P0DV18','context','人HEK293/人红细胞；另有昆虫终点','CC50 >300 μM；HC50 >300 μM','同上','2021 Table1/3；Methods','10.1016/j.bcp.2021.114693','原文BioC文本；PDF未视觉复核','限测量范围内弱/未达半数效应；不是绝对无毒；原广义毒性标签可能含其他终点'),
 (94,'Lycotoxin I','P61507','core','兔红细胞','200 μM时约55%溶血；<30 μM未检出溶血','室温1h；剂量6.25–200 μM','Hemolysis Assay；Fig7；Lysis of Erythrocytes','10.1074/jbc.273.4.2059','出版商原文索引；直接页面访问受限','合成C端酰胺化；不同于Na1b的人红细胞体系'),
 (160,'Lycotoxin II','P61508','context','原文另有膜/离子梯度实验','不复制Lycotoxin I的溶血数值','unknown','同篇两条序列及不同实验范围','10.1074/jbc.273.4.2059','原文及UniProt混合实验/相似性注释','本轮没有确认其单独直接红细胞剂量曲线'),
 (1434,'LyeTx I','C0HJU9','core','兔红细胞','ED50 1.3×10^-4 M =130 μM','1h；温度本轮未核实','Fig3；Antimicrobial and haemolytic assays','10.1007/s00726-009-0385-x','原文PDF提取；截图接口失败，未视觉复核','合成C端酰胺化；不能把LyeTxI-b等改造体实验归给本序列'),
 (288,'Oxyopinin-2a','P83248','context','本条直接红细胞测定尚未确认','撤回将Oxki2代表肽曲线归给2a的认定','unknown','原文Peptide Synthesis与Results均写Oxki2b；不能将简写Oxki2直接当2a','10.1074/jbc.M200511200','原文进一步核查更正；数据库FUNCTION仍保留但不替代逐肽归属','2a与2b是不同序列；本条不计为直接实验正确阳性'),
 (652,'Oxyopinin-2c','P83250','context','同家族原文及数据库注释','未把Oxki2代表肽曲线当作2c单独曲线','unknown','原文五条肽序列；后续选择两条合成代表肽','10.1074/jbc.M200511200','UniProt有实验来源注释，原文具体测定覆盖仍需区分','只作家族背景；不计为第二个独立直接终点正确对照'),
]
out=[]
for n,name,acc,elig,obj,value,time,loc,doi,tier,limit in ledger:
    r=by_id['peptide_'+str(n)]
    source=OUT/'sources'/f'{acc}.json'
    if source.exists():
        assert json.loads(source.read_text(encoding='utf-8'))['sequence']['value']==r['sequence']
    else:
        assert acc in r['exact_match_accessions'].split(';')
    v={k:r[k] for k in ['sample_id','sequence','label','fold','p_seq','p_struct','p_fusion','sequence_category','structure_category','fusion_category','length','cys_count','net_charge_proxy','mean_hydropathy','ideal_alpha_hydrophobic_moment','mean_plddt','hydrophobic_patch_fraction']}
    v.update(name=name,accession=acc,eligibility=elig,assay_object=obj,assay_value=value,exposure=time,source_location=loc,source_url='https://doi.org/'+doi,evidence_tier=tier,limits=limit,original_label_unchanged=True)
    out.append(v)
(OUT/'endpoint_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')

table=['| 样本 | 原标签 | 1D/3D/融合 | 长度 | 电荷近似 | 平均疏水性 | 理想α螺旋疏水矩 | 平均pLDDT |','|---|---:|---|---:|---:|---:|---:|---:|']
for r in out:
    table.append(f"| {r['sample_id']} {r['name']} | {r['label']} | {r['sequence_category']}/{r['structure_category']}/{r['fusion_category']} | {r['length']} | {r['net_charge_proxy']} | {float(r['mean_hydropathy']):.3f} | {float(r['ideal_alpha_hydrophobic_moment']):.3f} | {float(r['mean_plddt']):.3f} |")
(OUT/'feature_comparison.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
core=[r for r in out if r['eligibility']=='core']
positive=[r for r in core if r['label']=='1']
checks={'training_rows':len(rows),'shortlist':len(out),'direct_endpoint_core':len(core),'context_only':len(out)-len(core),'core_original_positive':len(positive),'core_original_positive_FN':sum(r['fusion_category']=='FN' for r in positive),'core_original_positive_TP':sum(r['fusion_category']=='TP' for r in positive),'independent_matched_assay_pairs':0,'original_label_changes':0,'model_training':False,'test_prediction_access':False,'atlas_sha256_before':before,'atlas_sha256_after':hashlib.sha256(ATLAS.read_bytes()).hexdigest(),'interpretation':'目的性小样本；跨研究/物种/剂量混杂；不得视为该机制全局错误率或模型提升证据'}
assert checks['atlas_sha256_after']==before
(OUT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False))
