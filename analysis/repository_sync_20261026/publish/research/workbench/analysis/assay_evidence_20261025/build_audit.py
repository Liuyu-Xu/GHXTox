"""Join frozen training predictions with a manually reviewed evidence ledger.
Run from the GHXTox root. Does not edit datasets or model predictions.
"""
from pathlib import Path
import csv, json, hashlib

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ATLAS = ROOT / 'analysis/expert_disagreement_20261025/training_expert_atlas.csv'
items = [
 (249,'Na1b','P0DSM4','case',308,'direct_assay_supported','有直接细胞毒性与溶血实验；尚不能确定漏判原因'),
 (201,'Ectatomin A','P49343','case',19,'tested_entity_mismatch','原文活性对象是完整A+B双链毒素；本轮未找到孤立A链的直接毒性测定'),
 (251,'H1C/H1U','P85219;P85222','case',139,'activity_supported_form_unresolved','摘要支持生物效应；末端形式跨库不一致，剂量单位与时间待原文核对'),
 (424,'H2C/H2U','P85220;P85223','case',None,'activity_supported_form_unresolved','同H1；没有强行添加匹配对照'),
 (308,'Cupiennin-4a','B3EWV6','composition_matched_control',249,'sequence_identified_activity_unresolved','测序/酰胺化注释有依据；本轮未确认本条肽直接毒性测定'),
 (19,'Augerpeptide hhe53','P0CI21','composition_matched_control',201,'predicted_sequence_activity_unresolved','cDNA预测成熟肽；本轮未找到直接毒性测定，不作为可靠阳性标准'),
 (139,'Antibacterial protein 2','P11698;Q4L5M2','composition_matched_control',251,'database_direct_activity_annotation','P11698有实验来源溶血注释；同序列Q4L5M2仅相似性推断；具体条件待核实'),
 (74,'Ectatomin B','P49344','molecular_counterpart',201,'tested_entity_mismatch','模型判对不等于孤立B链已证实有毒；与A来自同一完整毒素'),
 (4117,'Na1a','P0DSM2','family_context_negative',249,'negative_label_unvalidated','原标签0；数据库活性注释来自相似性推断，不证明本条有毒或无毒'),
]
before = hashlib.sha256(ATLAS.read_bytes()).hexdigest()
with ATLAS.open(encoding='utf-8-sig',newline='') as f:
    all_rows = {v['sample_id']:v for v in csv.DictReader(f)}
records=[]
for num,name,acc,role,partner,status,note in items:
    r=all_rows[f'peptide_{num}']
    rec={k:r[k] for k in ['sample_id','sequence','label','fold','p_seq','p_struct','p_fusion','sequence_category','structure_category','fusion_category','expert_decision_disagreement','exact_match_accessions']}
    rec.update(name=name,reviewed_accessions=acc,role=role,partner_id=f'peptide_{partner}' if partner else None,audit_status=status,audit_note=note,original_label_unchanged=True)
    assert all(a in r['exact_match_accessions'].split(';') for a in acc.split(';')), (num,acc,r['exact_match_accessions'])
    records.append(rec)
(OUT/'sample_audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
checks={'sample_count':len(records),'case_count':sum(r['role']=='case' for r in records),'composition_controls':sum(r['role']=='composition_matched_control' for r in records),'atlas_sha256_before':before,'atlas_sha256_after':hashlib.sha256(ATLAS.read_bytes()).hexdigest(),'label_changes':0,'retraining':False,'test_predictions_used':False,'note':'标签与分支结果来自冻结训练OOF，仅描述性分析；OOF独立性限制见原报告'}
assert checks['atlas_sha256_before']==checks['atlas_sha256_after']
assert len({r['sample_id'] for r in records})==9
assert [r for r in records if r['sample_id']=='peptide_424'][0]['partner_id'] is None
(OUT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['| 样本 | 原标签 | 1D判定 | 3D判定 | 融合判定 | 核查状态 |','|---|---:|---|---|---|---|']
for r in records:
    lines.append(f"| {r['sample_id']} {r['name']} | {r['label']} | {r['sequence_category']} | {r['structure_category']} | {r['fusion_category']} | {r['audit_note']} |")
(OUT/'sample_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False))
