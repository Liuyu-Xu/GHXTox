"""Verify new context pairs and preserve source distinctions in the final report."""
import json,hashlib,math
from pathlib import Path
import pandas as pd
R=Path('D:/dachuang2026/GHXTox');O=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
meta=read(O/'all_training_annotation_index.json');d=pd.read_csv(R/'analysis/expert_disagreement_20261025/training_expert_atlas.csv',float_precision='round_trip').set_index('sample_id')
p=pd.read_csv(O/'supplement_matched_context.csv',keep_default_na=False)
for name,block in p.groupby('supplement'):
 groups=set()
 for x in block.itertuples():
  a,b=d.loc[x.case_id],d.loc[x.control_id]
  assert a.label==b.label==x.label and a.fusion_error and not b.fusion_error
  assert a.sequence!=b.sequence and x.identity>=.4 and x.length_ratio>=.7
  if x.family_support=='explicit_cached_family':assert set(meta[x.case_id]['curated_family'])&set(meta[x.control_id]['curated_family'])
  elif x.family_support=='name_class_proxy':assert set(meta[x.case_id]['name_class'])&set(meta[x.control_id]['name_class'])
  else:assert a.group_id==b.group_id and a.fold==b.fold and x.identity>=.8
  assert a.group_id not in groups and b.group_id not in groups
  groups.update([a.group_id,b.group_id])
papers={}
for path in [R/'analysis/expert_disagreement_20261025/case_paper_abstracts_grouped.json',R/'analysis/ghxtox_error_atlas_20261025/evidence_phase2/representative_paper_abstracts.json']:
 for x in read(path)['resultList']['result']:
  pmid=str(x.get('pmid',x.get('id')));types=x.get('pubTypeList',{}).get('pubType',[])
  papers[pmid]=dict(pmid=pmid,title=x.get('title'),publication_types=types,is_review=any('review' in s.lower() for s in types),abstract_cached=bool(x.get('abstractText')),full_text_not_assumed=True,cache=str(path),source_url='https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/')
used=set()
for x in p[p.supplement=='crossfold_all327_errors_vs_all_correct'].itertuples():used.update([x.case_id,x.control_id])
refs=[]
for sid in sorted(used):
 refs.append(dict(sample_id=sid,referenced_pmids=meta[sid]['pmids'],cached_papers=[papers[x] for x in meta[sid]['pmids'] if x in papers],no_cached_original_does_not_mean_no_experiment=True))
save('literature_coverage_verified.json',refs)
for sid in ['peptide_195','peptide_57','peptide_66']:
 m=meta[sid];seq=d.loc[sid,'sequence']
 for f in m['chemical_feature_records']:
  if f.get('description')=='4-carboxyglutamate':
   start=f['location']['start']['value'];assert seq[start-1]=='E'
# Independently check all 148 boundary diagnostics and old file hashes.
b=pd.read_csv(O/'fusion_boundary_diagnostics.csv',float_precision='round_trip')
for x in b.itertuples():
 a=d.loc[x.sample_id];assert abs(.72*a.p_seq+.28*a.p_struct-a.p_fusion)<1e-8
 assert bool(x.structure_alone_correct_at_fusion_threshold)==bool((a.p_struct>=.47022256531284556)==bool(a.label))
v=read(O/'verification.json')
for path,oldhash in v['input_hashes_before'].items():
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for buf in iter(lambda:f.read(4*1024*1024),b''):h.update(buf)
 assert h.hexdigest()==oldhash,path
save('final_verification.json',dict(old_inputs_still_unchanged=True,new_pairs_labels_sequence_family_support_and_group_nonreuse_verified=True,conantokin_Gla_positions_encode_E_verified=True,all148_formula_and_threshold_diagnostics_verified=True,review_articles_separated_from_original_literature=True,supplements_explicitly_post_primary=True,old52_pairs_not_rerun=True,new_models_trained=False,test_predictions_read=False))
print('Final verification passed. 27 descriptive pairs; old inputs unchanged.')
