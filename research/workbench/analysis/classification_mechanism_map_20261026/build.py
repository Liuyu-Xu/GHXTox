"""Reuse existing OOF decisions/evidence; do not infer mechanism gold labels."""
from pathlib import Path
import json,re,hashlib,collections
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
paths={
 'linked':R/'analysis/error_evidence_linkage_20261025/all_training_linked.json',
 'annotations':R/'analysis/family_mechanism_comparison_20261026/all_training_annotation_index.json',
 'public':R/'analysis/environment_feasibility_20261025/ghxtox_training_links.json',
 'pairs':R/'analysis/family_mechanism_comparison_20261026/class_pair_evidence.json',
 'review':R/'analysis/label_ambiguity_map_20261025/reviewed_cases.json'}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={k:sha(p) for k,p in paths.items()};d={k:read(p) for k,p in paths.items()}
public=collections.defaultdict(list)
for x in d['public']: public[x['sample_id']].append(x)
patterns=list(dict.fromkeys(x['existing_pattern'] for x in d['linked']))
names={'membrane_or_lysis':'膜作用/裂解注释线索','ion_channel':'离子通道注释线索','receptor_or_signaling':'受体/信号注释线索','hemostasis_or_proteolysis':'凝血/蛋白水解注释线索','antimicrobial':'抗微生物注释线索','hemolysis_text':'溶血文字线索（含弱活性/否定）','public_hemolysis_record':'公开溶血测量关联（形式/阈值未确证）','manual_direct_hemolysis':'人工已查到直接溶血测量（非统一阳性）','disulfide_annotation':'存在显式二硫键注释（覆盖标记）','chemical_annotation':'存在化学状态特征记录（覆盖标记）'}
rows=[]
for x in d['linked']:
 a=d['annotations'][x['sample_id']];cats=[];evidence={}
 for key in ['membrane_or_lysis','ion_channel','receptor_or_signaling','hemostasis_or_proteolysis','antimicrobial']:
  if a['function_mentions'].get(key):cats.append(key);evidence[key]=a['function_mentions'][key]
 text=' '.join(a.get('function_texts',[]))
 if re.search(r'ha?emol',text,re.I):cats.append('hemolysis_text');evidence['hemolysis_text']=a['function_texts']
 if public[x['sample_id']]:cats.append('public_hemolysis_record');evidence['public_hemolysis_record']=public[x['sample_id']]
 direct_status={'有直接测量记录','完整PDF及关键页视觉核实','原研究中原版LyeTxI对照组的直接测量'}
 direct=[z for z in x.get('assay_records',[]) if z.get('endpoint')=='溶血' and z.get('evidence_status') in direct_status]
 if direct:cats.append('manual_direct_hemolysis');evidence['manual_direct_hemolysis']=direct
 if a.get('annotated_intra_pairs') or a.get('annotated_interchain_records'):cats.append('disulfide_annotation')
 if a.get('chemical_feature_records'):cats.append('chemical_annotation')
 rows.append(dict(sample_id=x['sample_id'],sequence=x['sequence'],label=int(x['label']),fold=int(x['fold']),group_id=x['group_id'],existing_pattern=x['existing_pattern'],sequence_category=x['sequence_category'],structure_category=x['structure_category'],fusion_category=x['fusion_category'],p_seq=x['p_seq'],p_struct=x['p_struct'],p_fusion=x['p_fusion'],categories=cats,evidence=evidence,annotation_name=a['name'],curated_family=a['curated_family'],name_class=a['name_class'],annotation_negation_warning=a['annotation_negation_warning'],fragment_flag=a['fragment_reference_flag'],cross_source_label_disagreement=x['cross_source_label_disagreement']=='True',manual_status=x.get('manual_primary_status'),full_record_available=bool(a['full_record_accessions']),source_urls=x['source_urls'],role='diagnostic_metadata_only_never_model_input'))
assert len(rows)==6387 and len({r['sample_id'] for r in rows})==6387
lookup={r['sample_id']:r for r in rows}
summary=[]
def stat(g):
 return dict(n=len(g),n_groups=len({r['group_id'] for r in g}),n_folds=len({r['fold'] for r in g}),patterns=dict(collections.Counter(r['existing_pattern'] for r in g)),seq_errors=sum(r['sequence_category'] in ['FN','FP'] for r in g),struct_errors=sum(r['structure_category'] in ['FN','FP'] for r in g),fusion_errors=sum(r['fusion_category'] in ['FN','FP'] for r in g),fragment_flagged=sum(r['fragment_flag'] for r in g),label_conflict=sum(r['cross_source_label_disagreement'] for r in g),negation_caution=sum(r['annotation_negation_warning'] for r in g),full_records=sum(r['full_record_available'] for r in g))
for cat in names:
 for label in [0,1]:
  g=[r for r in rows if cat in r['categories'] and r['label']==label]
  screened=[r for r in g if not r['fragment_flag'] and not r['cross_source_label_disagreement']]
  summary.append(dict(category=cat,title=names[cat],original_label=label,all_cached=stat(g),no_flag_subset_not_gold=stat(screened),sample_ids=[r['sample_id'] for r in g]))
pairs=[]
for x in d['pairs']:
 p=x['pair'];a=lookup[p['case_id']];b=lookup[p['control_id']]
 assert a['label']==b['label'] and a['fusion_category'] in ['FN','FP'] and b['fusion_category'] in ['TN','TP']
 pairs.append(dict(**p,case_categories=a['categories'],control_categories=b['categories'],shared_categories=sorted(set(a['categories'])&set(b['categories'])),both_without_fragment_or_conflict=not any([a['fragment_flag'],b['fragment_flag'],a['cross_source_label_disagreement'],b['cross_source_label_disagreement']]),both_full_records=a['full_record_available'] and b['full_record_available'],scope='existing exploratory pair, no new matching; name class proxy not curated family; OOF development folds already inspected'))
for name,obj in [('all_training_category_index.json',rows),('category_matrix.json',summary),('existing_pairs_category_overlay.json',pairs)]: (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
v=dict(source_hashes=before,source_unchanged=before=={k:sha(p) for k,p in paths.items()},n_training=len(rows),original_patterns=dict(collections.Counter(r['existing_pattern'] for r in rows)),public_link_rows=len(d['public']),unique_public_linked_samples=len(public.keys()-{r['sample_id'] for r in rows if not public[r['sample_id']]}),existing_pairs_reused=len(pairs),new_fits=0,new_pair_matching=0,test_predictions_read=False,probabilities_and_labels_copied_unchanged=True)
assert v['source_unchanged'];(O/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([dict(category=x['title'],label=x['original_label'],n=x['all_cached']['n'],seq=x['all_cached']['seq_errors'],struct=x['all_cached']['struct_errors'],fuse=x['all_cached']['fusion_errors'],screened=x['no_flag_subset_not_gold']['n'],screened_errors=x['no_flag_subset_not_gold']['fusion_errors']) for x in summary],ensure_ascii=False,indent=2))
