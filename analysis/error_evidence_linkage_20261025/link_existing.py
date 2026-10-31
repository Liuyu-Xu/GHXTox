"""Join existing evidence to existing outcomes; no inference, matching or model rerun."""
import csv, json, hashlib
from pathlib import Path
from collections import defaultdict, Counter
ROOT=Path('D:/dachuang2026/GHXTox'); OUT=Path(__file__).resolve().parent
paths={
 'atlas':ROOT/'analysis/expert_disagreement_20261025/training_expert_atlas.csv',
 'reviewed':ROOT/'analysis/label_ambiguity_map_20261025/reviewed_cases.json',
 'conditions':ROOT/'analysis/assay_completion_20261025/conditions.json',
 'annotations':ROOT/'analysis/ghxtox_error_atlas_20261025/evidence_phase2/error_accession_evidence_ledger.csv',
 'case_annotations':ROOT/'analysis/expert_disagreement_20261025/case_annotation_ledger.csv',
 'training_matches':ROOT/'analysis/ghxtox_error_atlas_20261025/evidence_phase2/training_uniprot_exact_matches.csv',
 'existing_summary':ROOT/'analysis/expert_disagreement_20261025/summary.json',
 'existing_verification':ROOT/'analysis/expert_disagreement_20261025/verification.json',
 'entity_audit':ROOT/'analysis/entity_input_audit_20261025/panel.json'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def csvread(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writejson(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def writecsv(n,rows):
 with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
before={k:sha(v) for k,v in paths.items()}
atlas=csvread(paths['atlas']); reviewed={r['sample_id']:r for r in read(paths['reviewed'])}
conditions=read(paths['conditions']); cond={r['sample_id']:r for r in conditions['samples']}
ann=defaultdict(list); ca=defaultdict(list); trainann=defaultdict(list)
for r in csvread(paths['annotations']):ann[r['sample_id']].append(r)
for r in csvread(paths['case_annotations']):ca[r['sample_id']].append(r)
for r in csvread(paths['training_matches']):
 if not r['Entry'].strip(): continue  # source contains unresolved placeholder rows too
 trainann[r['sample_id']].append(dict(sequence=r['sequence'],label=r['label'],accession=r['Entry'],protein_name=r['Protein names'],function_annotation=r['Function [CC]'],chain=r['Chain'],peptide=r['Peptide'],identity_relation=r['identity_relation'],source_url='https://www.uniprot.org/uniprotkb/'+r['Entry']+'/entry'))
entities={r['sample_id']:r for r in read(paths['entity_audit'])}
existing=read(paths['existing_summary']); assert read(paths['existing_verification'])['passed']
# Map existing categorical results to display labels; do not recompute probabilities or thresholds.
def pattern(r):
 if r['expert_group']=='both_correct': return '全部正确'
 if r['expert_group']=='both_wrong': return 'B_两路及融合都错'
 if r['expert_group']=='sequence_only_correct': return 'A_D_一维对三维错融合错' if r['fusion_error']=='True' else 'C_一维对三维错融合对'
 if r['expert_group']=='structure_only_correct': return 'D其余_一维错三维对融合错' if r['fusion_error']=='True' else 'C_一维错三维对融合对'
 raise AssertionError(r['expert_group'])
joined=[]; flat=[]
for r in atlas:
 sid=r['sample_id']; manual=reviewed.get(sid); c=cond.get(sid)
 for evidence in ([manual] if manual else [])+([c] if c else [])+ann[sid]+ca[sid]+trainann[sid]:
  assert evidence['sequence']==r['sequence'],sid
  assert int(evidence['label'])==int(r['label']),sid
 status='已有人工作例核查' if manual else '仅有缓存注释，未作例核查' if ann[sid] or ca[sid] or trainann[sid] else '未有具体核查，不代表无生物学因素'
 review=manual['manual_review'] if manual else None
 sources=list(dict.fromkeys((review.get('source_urls',[]) if review else [])+([c['source_url']] if c else [])+[v['source_url'] for v in ann[sid]+ca[sid]+trainann[sid] if v.get('source_url')]))
 x={**r,'existing_pattern':pattern(r),'evidence_link_status':status,'manual_review':review,'manual_primary_status':manual['primary_review_status'] if manual else '未核查',
    'assay_summary':c,'assay_records':[v for v in conditions['assays'] if v.get('sample_id')==sid],
    'cached_error_annotations':ann[sid],'cached_case_annotations':ca[sid],'cached_training_annotations':trainann[sid],'entity_input_audit':entities.get(sid),'source_urls':sources,
    'causal_error_explanation_established':False}
 joined.append(x)
 flat.append(dict(sample_id=sid,sequence=r['sequence'],original_label=r['label'],fold=r['fold'],existing_pattern=x['existing_pattern'],sequence_result=r['sequence_category'],structure_result=r['structure_category'],fusion_result=r['fusion_category'],p_seq=r['p_seq'],p_struct=r['p_struct'],p_fusion=r['p_fusion'],
    name=review['name'] if review else ';'.join(sorted(set(v['protein_name'] for v in ann[sid]+trainann[sid]))) or r['exact_match_accessions'],
    evidence_status=status,manual_status=x['manual_primary_status'],manual_tags=';'.join(review['review_tags']) if review else '',manual_note=review['review_note'] if review else '',
    automatic_annotation_accessions=';'.join(sorted(set(v['accession'] for v in ann[sid]+trainann[sid]))),automatic_function_annotations=' || '.join(dict.fromkeys(v['function_annotation'] for v in ann[sid]+trainann[sid] if v['function_annotation'])),
    fragment_reference_flag=r['any_fragment_match'],cross_source_label_disagreement=r['cross_source_label_disagreement'],source_urls=';'.join(sources),
    no_automatic_relabel_or_causal_claim=True))
writejson('all_training_linked.json',joined)
writecsv('all_training_evidence_index.csv',flat)
byid={r['sample_id']:r for r in flat}
writecsv('reviewed36_by_error_pattern.csv',[r for r in flat if r['sample_id'] in reviewed])
writecsv('errors327_evidence_index.csv',[r for r in flat if r['fusion_result'] in ['FP','FN']])
writecsv('disagreements334_evidence_index.csv',[r for r in flat if r['sequence_result'][0]!=r['structure_result'][0]])
gap=[r for r in flat if r['existing_pattern']=='D其余_一维错三维对融合错']
writecsv('D_remaining50_existing_evidence.csv',gap)
writecsv('C_structure_rescue98_existing_evidence.csv',[r for r in flat if r['existing_pattern']=='C_一维错三维对融合对'])
coverage=[]
for group in dict.fromkeys(r['existing_pattern'] for r in flat):
 rows=[r for r in flat if r['existing_pattern']==group]
 coverage.append(dict(existing_pattern=group,n_existing=len(rows),manual_case_review_n=sum(r['sample_id'] in reviewed for r in rows),has_cached_annotation_n=sum(bool(ann[r['sample_id']] or ca[r['sample_id']] or trainann[r['sample_id']]) for r in rows)))
writejson('linkage_coverage.json',coverage)
writecsv('linkage_coverage.csv',coverage)
lines=['# 已核查36案例对应哪些错误模式','', '此表只关联既有资料，不新增机制认定。多个标签可以同时出现；样本对错仍以原标签定义。','', '|样本|名称|已有模式|已有证据结论|具体说明|','|---|---|---|---|---|']
for r in flat:
 if r['sample_id'] in reviewed:
  cells=[r['sample_id'],r['name'],r['existing_pattern'],r['manual_status'],r['manual_note']]
  lines.append('|'+ '|'.join(v.replace('|','/').replace('\n',' ') for v in cells)+'|')
(OUT/'REVIEWED_CASES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
after={k:sha(v) for k,v in paths.items()};assert before==after
assert len(flat)==existing['n'] and sum(r['fusion_result'] in ['FP','FN'] for r in flat)==327
assert len(reviewed)==36 and len(gap)==50
writejson('verification.json',dict(inputs={k:dict(path=str(paths[k]),sha256=v) for k,v in before.items()},inputs_unchanged=True,sequence_label_links_verified=True,all_original_rows_and_probabilities_preserved=True,n_manual_cases=36,existing_verification_reused=True,grouping_or_matching_or_tests_rerun=False,new_statistics='only evidence linkage coverage; old group counts reused',models_trained=False,test_predictions_read=False))
print(json.dumps(coverage,ensure_ascii=False,indent=2))
