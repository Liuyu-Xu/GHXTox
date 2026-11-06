"""Clearly post-primary descriptive family context; no significance tests."""
import json, math, hashlib
from pathlib import Path
from collections import defaultdict
import pandas as pd
from Bio.Align import PairwiseAligner
ROOT=Path('D:/dachuang2026/GHXTox'); O=Path(__file__).resolve().parent
meta=json.loads((O/'all_training_annotation_index.json').read_text(encoding='utf-8'))
d=pd.read_csv(ROOT/'analysis/expert_disagreement_20261025/training_expert_atlas.csv',float_precision='round_trip')
by={r.sample_id:r for r in d.itertuples()}; target=d[d.expert_group=='structure_only_correct']; fail=target[target.fusion_error];rescue=target[~target.fusion_error]
aligner=PairwiseAligner(mode='global',match_score=2,mismatch_score=-1,open_gap_score=-3,extend_gap_score=-.5);aligncache={}
def ident(s,t):
 k=tuple(sorted([s,t]))
 if k not in aligncache:
  a,b=by[s].sequence,by[t].sequence;al=aligner.align(a,b)[0]
  match=sum(sum(x==y for x,y in zip(a[i:j],b[k:l])) for (i,j),(k,l) in zip(*al.aligned));aligncache[k]=match/max(len(a),len(b))
 return aligncache[k]
def support(s,t):
 a,b=meta[s],meta[t];cf=set(a['curated_family'])&set(b['curated_family']);nc=set(a['name_class'])&set(b['name_class'])
 if cf:return 'explicit_cached_family',';'.join(sorted(cf))
 if a['curated_family'] and b['curated_family']:return None,''
 return ('name_class_proxy',';'.join(sorted(nc))) if nc else (None,'')
out=[];missing=[]
contrasts=[('same_group_failure50_vs_all_correct',fail,d[~d.fusion_error],True),('crossfold_failure50_vs_rescue98',fail,rescue,False),('crossfold_failure50_vs_all_correct',fail,d[~d.fusion_error],False),('crossfold_all327_errors_vs_all_correct',d[d.fusion_error],d[~d.fusion_error],False)]
for name,cases,controls,samegroup in contrasts:
 pools={}
 for r in cases.itertuples():
  cc=controls[controls.label==r.label]
  cc=cc[(cc.group_id==r.group_id)&(cc.fold==r.fold)] if samegroup else cc[cc.group_id!=r.group_id]
  candidates=[]
  for c in cc.itertuples():
   lr=min(r.length,c.length)/max(r.length,c.length)
   if lr<.7:continue
   sup,family=support(r.sample_id,c.sample_id)
   if not sup and not samegroup:continue
   score=ident(r.sample_id,c.sample_id)
   if score<(.8 if not sup else .4):continue
   candidates.append(dict(supplement=name,case_id=r.sample_id,control_id=c.sample_id,label=int(r.label),case_fold=int(r.fold),control_fold=int(c.fold),case_group=r.group_id,control_group=c.group_id,case_pattern=r.expert_group,family_support=sup or 'sequence_group_proxy',family=family,identity=score,length_ratio=lr))
  pools[r.sample_id]=sorted(candidates,key=lambda x:(x['family_support']!='explicit_cached_family',-x['identity'],x['case_fold']!=x['control_fold'],-x['length_ratio'],x['control_id']))
 used=set()
 for sid in sorted(pools,key=lambda s:(len(set(x['control_group'] for x in pools[s])),s)):
  candidates=[x for x in pools[sid] if x['case_group'] not in used and x['control_group'] not in used]
  if candidates:
   pair=candidates[0];out.append(pair);used.update([pair['case_group'],pair['control_group']])
  else:missing.append(dict(supplement=name,case_id=sid,label=int(by[sid].label),reason='no_comparable_family_sequence_control' if not pools[sid] else 'group_no_reuse'))
 print(name,sum(x['supplement']==name for x in out),flush=True)
pd.DataFrame(out).to_csv(O/'supplement_matched_context.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(missing).to_csv(O/'supplement_unmatched.csv',index=False,encoding='utf-8-sig')
features=['cys_count','cys_span_fraction','net_charge_proxy','mean_hydropathy','hydrophobic_contact_excess','positive_hydrophobic_contact_excess','like_charge_contact_excess','charge_hydrophobic_separation','hydrophobic_patch_fraction','local_positive_excess','ideal_alpha_hydrophobic_moment','mean_plddt','sequence_seed_std']
paired=[]
for p in out:
 r=dict(p)
 for f in features:r['case_'+f]=getattr(by[p['case_id']],f);r['control_'+f]=getattr(by[p['control_id']],f)
 paired.append(r)
pd.DataFrame(paired).to_csv(O/'supplement_features_descriptive_only.csv',index=False,encoding='utf-8-sig')
summary={name:dict(pairs=sum(x['supplement']==name for x in out),original_negative_pairs=sum(x['supplement']==name and x['label']==0 for x in out),original_positive_pairs=sum(x['supplement']==name and x['label']==1 for x in out),explicit_family_pairs=sum(x['supplement']==name and x['family_support']=='explicit_cached_family' for x in out),cross_fold_pairs=sum(x['supplement']==name and x['case_fold']!=x['control_fold'] for x in out),unmatched=sum(x['supplement']==name for x in missing)) for name,*_ in contrasts}
(O/'supplement_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
# Whole name classes, not just selected matched pairs. Multiple classes may overlap.
rows=[]
for family in sorted({c for s in target.sample_id for c in meta[s]['name_class']}):
 for label in [0,1]:
  for block,frame in [('failure50',fail),('rescue98',rescue),('full_training_correct_background',d[~d.fusion_error])]:
   ids=[r.sample_id for r in frame[frame.label==label].itertuples() if family in meta[r.sample_id]['name_class']]
   if not ids:continue
   r=dict(name_class=family,label=label,block=block,n=len(ids),sample_ids=';'.join(ids),fragment_reference_n=sum(meta[s]['fragment_reference_flag'] for s in ids),full_feature_record_n=sum(bool(meta[s]['full_record_accessions']) for s in ids))
   for f in features:r['median_'+f]=pd.Series([getattr(by[s],f) for s in ids]).median()
   rows.append(r)
pd.DataFrame(rows).to_csv(O/'whole_family_background_descriptive.csv',index=False,encoding='utf-8-sig')
print(json.dumps(summary,ensure_ascii=False,indent=2))
