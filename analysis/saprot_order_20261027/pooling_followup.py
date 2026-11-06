from pathlib import Path
import sys,json
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from scipy.special import expit,log_expit,logsumexp
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 return x
def save(n,x):(O/n).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf-8')
P=read(O/'pooling_followup_protocol.json');original=np.load(O/'originals_features.npz');mutant=np.load(O/'mutants_features.npz');origscore=np.load(O/'original_scores.npz');mutscore=np.load(O/'mutant_scores.npz');cohort={r['sample_id']:r for r in read(O/'cohort.json')};records=[r for r in read(O/'sequence_manifest.json') if r['mode']!='original' and r['eligible']];seqs=original['sequences'].tolist()+mutant['sequences'].tolist();si={s:i for i,s in enumerate(seqs)};base=np.concatenate([original['base'],mutant['base']]);sap=np.concatenate([original['saprot'],mutant['saprot']]);oi={k:i for i,k in enumerate(origscore['sample_ids'].tolist())};mi={k:i for i,k in enumerate(mutscore['keys'].tolist())};old=np.load(A/'saprot_and_robust_classifier_pilot_20261026/frozen_features.npz');oldbase=old['base'];oldi={k:i for i,k in enumerate(old['sample_ids'].tolist())};torch.set_num_threads(2)
class Head(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(3561,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)
combine=lambda a:logsumexp(log_expit(np.stack(a)),axis=0)-logsumexp(log_expit(-np.stack(a)),axis=0)
out={k:np.full(len(records),np.nan) for k in ['mean_only','max_only','std_only']}
for fold in range(5):
 ii=[i for i,r in enumerate(records) if r['fold']==fold];oldraw=np.stack([base[si[records[i]['original_sequence']]] for i in ii]);newraw=np.stack([base[si[records[i]['sequence']]] for i in ii]);oldraw[:,3072:]=np.stack([oldbase[oldi[records[i]['sample_id']],3072:] for i in ii]);models=[]
 for seed in [20261026,20261027]:
  c=torch.load(A/'saprot_and_robust_classifier_pilot_20261026/checkpoints'/f'{seed}_fold{fold}_baseline_BCE.pt',map_location='cpu');m=Head();m.load_state_dict(c['state_dict']);m.eval();models.append((m,np.array(c['scaler_mean']),np.array(c['scaler_scale'])))
 for block,lo in [('mean_only',0),('max_only',1024),('std_only',2048)]:
  raw=oldraw.copy();raw[:,lo:lo+1024]=newraw[:,lo:lo+1024];raw=np.concatenate([raw,np.zeros((len(ii),480),np.float32)],axis=1);pred=[]
  for m,mu,sd in models:
   v=raw.copy();v-=mu;v/=sd
   with torch.inference_mode():pred.append(np.array(m(torch.tensor(v.tolist(),dtype=torch.float32)).tolist()))
  out[block][ii]=combine(pred)
 print(f'Pooling feature interventions fold{fold} done',flush=True)
np.savez_compressed(O/'pooling_intervention_scores.npz',keys=np.array([r['key'] for r in records]),**out)
allrows=[]
for i,r in enumerate(records):
 c=cohort[r['sample_id']];a,b=si[r['original_sequence']],si[r['sequence']];orig=origscore['baseline2_clamped'][oi[r['sample_id']]];row=dict(sample_id=r['sample_id'],mode=r['mode'],replicate=r['replicate'],fold=r['fold'],role=c['fusion_masked4']['role'],changed_fraction=r['changed_fraction'],block_responses={})
 for block in out:row['block_responses'][block]=abs(float(expit(out[block][i])-expit(orig)))
 row['block_responses']['all_tristat_changed']=abs(float(expit(mutscore['baseline2_clamped'][mi[r['key']]])-expit(orig)))
 for label,vec in [('ProtT5_mean',base[:,:1024]),('SaProt_mean',sap)]:
  x,y=vec[a].astype(float),vec[b].astype(float);row[label]=dict(cosine_distance=float(1-np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))),relative_L2=float(np.linalg.norm(x-y)/np.linalg.norm(x)))
 allrows.append(row)
save('pooling_per_perturbation.json',allrows);agg=[];bucket=defaultdict(list)
for r in allrows:bucket[r['sample_id'],r['mode']].append(r)
for (sid,mode),rr in bucket.items():
 agg.append(dict(sample_id=sid,mode=mode,fold=rr[0]['fold'],role=rr[0]['role'],changed_fraction=float(np.mean([r['changed_fraction'] for r in rr])),block_responses={k:float(np.mean([r['block_responses'][k] for r in rr])) for k in rr[0]['block_responses']},**{label:{k:float(np.mean([r[label][k] for r in rr])) for k in ['cosine_distance','relative_L2']} for label in ['ProtT5_mean','SaProt_mean']}))
summary=[]
for mode in sorted({r['mode'] for r in agg}):
 for role in ['all','corrected','harmed','persistent_wrong','persistent_correct']:
  rr=[r for r in agg if r['mode']==mode and (role=='all' or r['role']==role)];summary.append(dict(mode=mode,role=role,n=len(rr),block_mean_probability_responses={k:float(np.mean([r['block_responses'][k] for r in rr])) for k in rr[0]['block_responses']},mean_representation_response={label:{k:float(np.mean([r[label][k] for r in rr])) for k in ['cosine_distance','relative_L2']} for label in ['ProtT5_mean','SaProt_mean']}))
save('pooling_sample_means.json',agg);save('pooling_summary.json',summary);print(json.dumps(clean(summary),ensure_ascii=False,indent=2))
