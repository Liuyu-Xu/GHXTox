"""Evidence triage sidecar. Flags do not alter labels or establish error causes."""
from pathlib import Path
import csv,json,hashlib,collections,itertools

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;OUT.mkdir(parents=True,exist_ok=True)
ATLAS=ROOT/'analysis/expert_disagreement_20261025/training_expert_atlas.csv'
E=ROOT/'analysis/ghxtox_error_atlas_20261025/evidence_phase2'
def csvread(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
sources=[ATLAS,E/'error_accession_evidence_ledger.csv',E/'representative20_review.csv',ROOT/'analysis/assay_evidence_20261025/sample_audit.json',ROOT/'analysis/assay_completion_20261025/conditions.json']
before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
atlas=csvread(ATLAS);assert len(atlas)==6387
annotations=csvread(E/'error_accession_evidence_ledger.csv');ann=collections.defaultdict(list)
for a in annotations:ann[a['sample_id']].append(a)
curated={}
def put(sid,name,tags,grade,note,source,urls=None):
 curated[sid]=dict(name=name,review_tags=tags,review_grade=grade,review_note=note,evidence_files=source,source_urls=urls or [],causal_error_explanation_established=False,original_binary_label_provenance='未恢复完整历史判定链')
manual=csvread(E/'representative20_review.csv')
for a in manual:
 sid=a['sample_id'];tags=[]
 if a['any_fragment_match']=='True' or sid=='peptide_4096':tags.append('molecular_entity')
 if sid in ['peptide_5033','peptide_2653','peptide_322','peptide_332','peptide_33']:tags.append('assay_endpoint_context')
 if sid in ['peptide_3648','peptide_2726','peptide_5902','peptide_1623','peptide_806','peptide_1392']:tags.append('insufficient_specific_evidence')
 if sid in ['peptide_33','peptide_322','peptide_2726']:tags.append('chemical_state')
 if not tags:tags=['insufficient_specific_evidence']
 put(sid,a['recovered_identity'],tags,'已有具体资料线索，非全部原文方法核实',a['review_note'],['analysis/ghxtox_error_atlas_20261025/evidence_phase2/representative20_review.csv'],[s for s in a['source_links'].split(';') if s])

first=read(ROOT/'analysis/assay_evidence_20261025/sample_audit.json')
for a in first:
 sid=a['sample_id']
 tags={'peptide_249':['direct_effect_supported_error'],'peptide_201':['molecular_entity'],'peptide_251':['chemical_state','insufficient_specific_evidence'],'peptide_424':['chemical_state','insufficient_specific_evidence'],'peptide_308':['insufficient_specific_evidence','chemical_state'],'peptide_19':['insufficient_specific_evidence'],'peptide_139':['insufficient_specific_evidence','chemical_state'],'peptide_74':['molecular_entity'],'peptide_4117':['insufficient_specific_evidence']}[sid]
 note=a['audit_note']
 if sid in ['peptide_251','peptide_424']:note='原摘要支持溶血/细胞效应，但UniProt N端甲酰化与Hemolytik Free记录冲突；原表单位/条件未全部核实，未证明修饰导致漏判。'
 put(sid,a['name'],tags,'指定分子/终点资料核查',note,['analysis/assay_evidence_20261025/REPORT.md','analysis/assay_completion_20261025/REPORT.md'])

latest=read(ROOT/'analysis/assay_completion_20261025/conditions.json')
for a in latest['samples']:
 sid=a['sample_id'];tags=[]
 if sid in ['peptide_249','peptide_463','peptide_94','peptide_1434']:tags=['direct_effect_supported_error']
 elif sid in ['peptide_3912','peptide_444']:tags=['assay_endpoint_context']
 else:tags=['insufficient_specific_evidence']
 if sid in ['peptide_94','peptide_160','peptide_1434']:tags.append('chemical_state')
 note=a['limits']+'；'+a['chemical_status']
 if sid=='peptide_288':note='已撤回直接溶血正确对照认定：原文后续合成代表肽是2b；不能将曲线归给2a。'
 if sid=='peptide_652':note='数据库活性来源不能代替逐肽测定覆盖；不能将2b的后续曲线归给2c。'
 if sid=='peptide_249':note='原文精确序列有HEK293活性与人红细胞溶血测量；融合FN；这支持研究该难例，不证明具体致错机制。'
 if sid=='peptide_463':note='原文有较弱HEK293效应，HC50>250μM；融合FN；具体终点阈值与原标签生成链仍未知。'
 if sid=='peptide_94':note='合成LycotoxinI C端酰胺化；兔红细胞室温1h，200μM约55%溶血；1D TP/3D FN/融合FN；不证明修饰导致错误。'
 if sid=='peptide_1434':note='LyeTxI为游离N端、酰胺化C端；兔红细胞37°C1h，ED50=130μM；1D TP/3D FN/融合FN；无同序列修饰因果对照。'
 if sid=='peptide_3912':note='原标签0且模型TN；原研究有HEK293与人红细胞效应。原标签终点未知，不能认定原标错或可靠无毒。'
 if sid=='peptide_444':note='原标签1且融合FN；原研究哺乳动物CC50/HC50均>300μM，另有昆虫作用。不同终点不能混为通用毒性阈值。'
 put(sid,a['name'],tags,'指定终点直接记录' if 'direct_effect_supported_error' in tags else '具体证据限制已记录',note,['analysis/assay_completion_20261025/conditions.json','analysis/assay_completion_20261025/REPORT.md'],[a['source_url']])

rows=[]
for a in atlas:
 sid=a['sample_id'];e=ann.get(sid,[]);features=[f for v in e for f in json.loads(v['documented_ptm_features'] or '[]')]
 hints=[]
 if a['any_fragment_match']=='True':hints.append('fragment_reference_match')
 if a['cross_source_label_disagreement']=='True':hints.append('cross_source_label_disagreement')
 if a['identity_status']=='no_short_entry_match':hints.append('identity_unresolved')
 if any('interchain' in f.get('description','').lower() for f in features):hints.append('interchain_annotation')
 if any(f['type'] in ['Modified residue','Cross-link'] for f in features):hints.append('non_disulfide_modification_annotation')
 c=curated.get(sid)
 status='尚未逐条归因'
 if c:
  t=c['review_tags']
  if 'molecular_entity' in t:status='分子对象问题的具体案例/线索'
  elif 'chemical_state' in t and sid in ['peptide_251','peptide_424']:status='化学形式证据冲突待确认'
  elif 'assay_endpoint_context' in t:status='实验终点或作用背景需区分'
  elif 'direct_effect_supported_error' in t:status='直接效应支持的漏判候选'
  else:status='该具体分子/终点证据不足'
  # Parent-protein fragment examples with primary tested-entity evidence are explicit.
  if sid in ['peptide_201','peptide_74','peptide_309']:c['review_grade']='原研究显示所测分子与孤立输入不同'
 out={k:a[k] for k in ['sample_id','sequence','label','fold','p_seq','p_struct','p_fusion','fusion_category','sequence_category','structure_category','expert_decision_disagreement','cys_count','exact_match_accessions','identity_status']}
 out.update(machine_review_hints=hints,manual_review=c,primary_review_status=status,original_label_unchanged=True,suggested_weight_change=None)
 rows.append(out)
save('training_triage.json',rows);errors=[r for r in rows if r['fusion_category'] in ['FP','FN']];save('error_triage.json',errors)
assert len(errors)==327

def summarize(rs):return dict(samples=len(rs),fusion_errors=sum(r['fusion_category'] in ['FP','FN'] for r in rs),FP=sum(r['fusion_category']=='FP' for r in rs),FN=sum(r['fusion_category']=='FN' for r in rs),expert_disagreements=sum(r['expert_decision_disagreement']=='True' for r in rs))
bulk=[]
for key,title in [('fragment_reference_match','精确匹配到Fragment参考条目'),('cross_source_label_disagreement','其他来源有相反广义标签'),('identity_unresolved','未恢复短条目身份'),('interchain_annotation','错误样本有链间键注释（仅错误缓存覆盖）'),('non_disulfide_modification_annotation','错误样本有非二硫键修饰注释（仅错误缓存覆盖）')]:
 rs=[r for r in rows if key in r['machine_review_hints']];bulk.append(dict(flag=key,title=title,coverage='all_train' if key in ['fragment_reference_match','cross_source_label_disagreement','identity_unresolved'] else 'error_records_only',**summarize(rs)))
manual_rows=[r for r in rows if r['manual_review']];save('reviewed_cases.json',manual_rows)
statuses=[]
for key in sorted({r['primary_review_status'] for r in rows}):statuses.append(dict(status=key,**summarize([r for r in rows if r['primary_review_status']==key])))
tags=[]
for key in ['molecular_entity','assay_endpoint_context','chemical_state','insufficient_specific_evidence','direct_effect_supported_error']:
 tags.append(dict(tag=key,**summarize([r for r in manual_rows if key in r['manual_review']['review_tags']])))
summary={'total':summarize(rows),'machine_hints':bulk,'manual_reviewed':summarize(manual_rows),'manual_primary_statuses':statuses,'manual_overlapping_tags':tags,'historical_fragment_or_crosssource_flag':summarize([r for r in rows if any(k in r['machine_review_hints'] for k in ['fragment_reference_match','cross_source_label_disagreement'])]),'unreviewed_is_not_insufficient_evidence':True,'causal_error_mechanisms_proven':0}
save('summary.json',summary)

# Fixed computation-first priority panels. Not ready-made clean training subsets.
panels=[{'priority':1,'topic':'分子对象边界','anchor_ids':['peptide_201','peptide_74','peptide_309','peptide_4096'],'computational_next':'在现有记录中扩展片段/亚基面板，区分整分子与片段标签，先报告同标签层内的错误与分歧；不把Toxin关键词作为模型输入','evidence_limit':'只说明输入与实验对象不同，不能自动判定标签错'}, {'priority':2,'topic':'终点明确的真实难例候选','anchor_ids':['peptide_249','peptide_463','peptide_94','peptide_1434'],'computational_next':'固定为病例分析面板，对照同终点的可靠正确样本；不先加权；原OOF只发现假设','evidence_limit':'4例不能证明普遍机制；定量弱效应与广义标签判定链未恢复'}, {'priority':3,'topic':'化学形式','anchor_ids':['peptide_251','peptide_424','peptide_94','peptide_1434','peptide_160'],'computational_next':'先用公开同序列不同末端记录形成成对计算面板；H1/H2因形式冲突不能做已确认修饰纠错样本','evidence_limit':'末端修饰与错误相关不等于因果；需同化学身份/背景对应'}]
save('research_priority_panels.json',panels)

# Public chemistry-pair feasibility using the already frozen environmental dataset.
public_path=ROOT/'analysis/environment_feasibility_20261025/model_records.json'
public=read(public_path);before[str(public_path.relative_to(ROOT))]=hashlib.sha256(public_path.read_bytes()).hexdigest()
bucket=collections.defaultdict(list)
for r in public:bucket[(r['sequence'],r['pmid'],r['species'],r['dose_uM'],r['pH'])].append(r)
pairs=[]
for key,rs in bucket.items():
 if len({r['chemical_key'] for r in rs})>1:
  pairs.append(dict(sequence=key[0],pmid=key[1],species=key[2],dose_uM=key[3],pH=key[4],records=rs,uncontrolled_fields=['temperature','exposure_time','RBC_density','buffer'],status='数据库同已记录背景的末端形式对照线索，非全条件匹配证明'))
save('public_terminal_pair_blocks.json',pairs)
chemcounts={'blocks':len(pairs),'unique_backbones':len({r['sequence'] for r in pairs}),'PMID_identifiers':len({r['pmid'] for r in pairs}),'records':sum(len(p['records']) for p in pairs),'H1_H2_exact_present':any(r['sequence'] in ['MSGIVEAISNAVKSGLDHDWVNMGTSIADVVAKGADFIAGFFS','MDFIIDIIKKIVGLFTGK'] for r in pairs),'GHXTox_training_exact_backbones':len({p['sequence'] for p in pairs}&{r['sequence'] for r in rows}),'warning':'只能说明公开成对计算资料的可获得性，不归因H1/H2或Ectatomin错误'}
save('chemistry_pair_feasibility.json',chemcounts)

# Human-readable finite queue: reviewed cases first, no invented confidence scores.
queue=sorted(errors,key=lambda r:(r['manual_review'] is None, r['primary_review_status'],int(r['sample_id'].split('_')[1])))
lines=['# 争议样本分类与错误清单','', 'FP/FN均相对原标签；分类只是证据状态，不是重新判标签。具体标签可重叠，主分类仅便于浏览。','', '| 样本 | 原标签 | 1D/3D/融合 | 主分类 | 依据 |','|---|---:|---|---|---|']
for r in queue:
 c=r['manual_review'];note=c['review_note'] if c else '；'.join(r['machine_review_hints']) or '本轮未逐条归因，不能当可靠标签'
 lines.append(f"| {r['sample_id']} {c['name'] if c else ''} | {r['label']} | {r['sequence_category']}/{r['structure_category']}/{r['fusion_category']} | {r['primary_review_status']} | {note.replace('|','/')} |")
(OUT/'ERROR_CASES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
allpaths={**{str(p.relative_to(ROOT)):p for p in sources},str(public_path.relative_to(ROOT)):public_path};after={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in allpaths.items()};assert before==after
assert len({r['sample_id'] for r in rows})==6387 and len({r['sample_id'] for r in manual_rows})==len(curated)
assert {r['sample_id'] for r in rows if r['manual_review'] and 'direct_effect_supported_error' in r['manual_review']['review_tags']}=={'peptide_249','peptide_463','peptide_94','peptide_1434'}
assert next(r for r in rows if r['sample_id']=='peptide_288')['primary_review_status']=='该具体分子/终点证据不足'
save('verification.json',{'inputs_sha256_before':before,'inputs_sha256_after':after,'train_rows':6387,'error_rows':327,'manual_rows':len(manual_rows),'original_labels_and_predictions_unchanged':True,'automatic_relabeling':False,'unreviewed_not_gold':True,'error_annotation_coverage_only_for_errors':True,'new_model_training':False,'test_predictions_used':False,'Oxy2a_correction_preserved':True})
print(json.dumps(summary,ensure_ascii=False));print(json.dumps(chemcounts,ensure_ascii=False))
