import csv,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

P=Path(__file__).resolve().parent; root=P.parents[1]
with (P.parent/'expert_disagreement_20261025/training_expert_atlas.csv').open(encoding='utf-8-sig',newline='') as f: atlas=list(csv.DictReader(f))
meta=json.loads((P.parent/'lowcys_background_20261027/all_samples_with_symmetric_family.json').read_text(encoding='utf8'))
assert len(atlas)==len(meta)==6387 and all(a['sample_id']==m['sample_id'] for a,m in zip(atlas,meta))
for a,m in zip(atlas,meta):
 a['flag']=bool(m['uniform_flag']); a['families']=set((m.get('current_family_sidecar') or {}).get('parent_families') or []);a['seq']=a['sequence'];a['length_n']=len(a['seq']);a['cys_n']=a['seq'].count('C')

def features(s):
 n=len(s); loc=[i for i,x in enumerate(s) if x=='C']; hyd=set('AILMFWVY');charged=set('KRDE')
 gaps=[(loc[i+1]-loc[i])/(n-1) for i in range(len(loc)-1)]
 return {'cys_mean_gap_norm':float(np.mean(gaps)) if gaps else None,
         'cys_span_norm':(loc[-1]-loc[0])/(n-1) if len(loc)>1 and n>1 else None,
         'hydrophobic_adjacent_fraction':sum(s[i] in hyd and s[i+1] in hyd for i in range(n-1))/(n-1) if n>1 else None,
         'charged_hydrophobic_boundaries':sum((s[i] in hyd and s[i+1] in charged) or (s[i] in charged and s[i+1] in hyd) for i in range(n-1))/(n-1) if n>1 else None,
         'cys_neighbor_hydrophobic_fraction':sum(s[j] in hyd for i in loc for j in (i-1,i+1) if 0<=j<n)/sum(1 for i in loc for j in (i-1,i+1) if 0<=j<n) if loc else None}

def run(label,low,high):
 target=[a for a in atlas if a['expert_group']=='both_wrong' and int(a['label'])==label and low<=a['cys_n']<=high]
 controls=[a for a in atlas if a['expert_group']=='both_correct' and int(a['label'])==label and low<=a['cys_n']<=high]
 # One-to-one matching within fold, identical Cys and flags, length difference <=5, different existing group.
 # Family agreement and composition similarity are NOT required; prefer shared known family, then length and composition.
 pairs=[]; unmatch=[]
 for fold in range(5):
  t=[a for a in target if int(a['fold'])==fold];c=[a for a in controls if int(a['fold'])==fold]
  if not t:continue
  cost=np.full((len(t),len(c)+len(t)),1000.0);cost[:,len(c):]=100.
  for i,a in enumerate(t):
   for j,b in enumerate(c):
    if a['group_id']==b['group_id'] or a['cys_n']!=b['cys_n'] or a['flag']!=b['flag'] or abs(a['length_n']-b['length_n'])>5:continue
    aa=Counter(a['seq']);bb=Counter(b['seq']); composition=sum(abs(aa[k]/len(a['seq'])-bb[k]/len(b['seq'])) for k in set(aa)|set(bb))
    family=bool(a['families'] & b['families']);cost[i,j]=(0 if family else 10)+abs(a['length_n']-b['length_n'])+composition
  ii,jj=linear_sum_assignment(cost)
  for i,j in zip(ii,jj):
   a=t[i]
   if j>=len(c) or cost[i,j]>=100:unmatch.append(a['sample_id']);continue
   b=c[j];fa=features(a['seq']);fb=features(b['seq'])
   row={'wrong_id':a['sample_id'],'control_id':b['sample_id'],'fold':fold,'label':label,'cys_count':a['cys_n'],'wrong_length':a['length_n'],'control_length':b['length_n'],'flag':int(a['flag']),'wrong_group':a['group_id'],'control_group':b['group_id'],'shared_annotated_family':int(bool(a['families']&b['families'])),'wrong_sequence':a['seq'],'control_sequence':b['seq']}
   for k in fa:row[k+'_wrong']=fa[k];row[k+'_control']=fb[k];row[k+'_difference']=None if fa[k] is None or fb[k] is None else fa[k]-fb[k]
   pairs.append(row)
 return {'target':len(target),'controls':len(controls),'pairs':pairs,'unmatched_ids':unmatch}

out={'low_pos':run(1,0,1),'high_neg':run(0,6,999)}
allpairs=[]
for name,d in out.items():
 for row in d['pairs']:allpairs.append({'analysis_group':name,**row})
with (P/'matched_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=allpairs[0].keys());w.writeheader();w.writerows(allpairs)
summary={}
for name,d in out.items():
 rows=d['pairs'];fs={}
 for k in features('ACDC'):
  v=[r[k+'_difference'] for r in rows if r[k+'_difference'] is not None]
  fs[k]={'n':len(v),'median_difference':float(np.median(v)) if v else None,'wrong_greater':sum(x>0 for x in v),'wrong_smaller':sum(x<0 for x in v),'ties':sum(x==0 for x in v)}
 summary[name]={'targets':d['target'],'controls':d['controls'],'matched':len(rows),'unmatched':len(d['unmatched_ids']),'unmatched_ids':d['unmatched_ids'],'shared_family_pairs':sum(r['shared_annotated_family'] for r in rows),'flagged_pairs':sum(r['flag'] for r in rows),'fold_counts':dict(Counter(str(r['fold']) for r in rows)),'features':fs}
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,ensure_ascii=False,indent=2))
