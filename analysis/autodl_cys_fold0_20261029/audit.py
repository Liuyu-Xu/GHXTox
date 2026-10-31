import json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.metrics import matthews_corrcoef,average_precision_score
read=lambda p:json.loads(p.read_text(encoding='utf8'))
base=P/'runs/fold0';runs={k:read(base/k/'predictions.json') for k in 'ABCD'};res={k:read(base/k/'result.json') for k in 'ABCD'};meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json');bg=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json');ids=[x['source_index'] for x in roles if x['outer_fold']==0 and x['role']=='test'];assert all([x['sample_id'] for x in runs[k]]==[meta[i]['sample_id'] for i in ids] for k in runs);assert len(set(r['initial_head_sha256'] for r in res.values()))==1
for k in runs:
 for j,i in enumerate(ids):assert abs(runs[k][j]['fusion_probability']-(.72*runs[k][j]['sequence_probability']+.28*float(bg[i]['p_struct'])))<1e-9
Y=np.array([meta[i]['label'] for i in ids]);G=np.array([meta[i]['group_id'] for i in ids]);c=np.array([meta[i]['cys_count'] for i in ids]);flag=np.array([meta[i]['uniform_flag'] for i in ids],dtype=bool);p={k:np.array([x['fusion_probability'] for x in runs[k]]) for k in runs};d={k:p[k]>=.47022256531284556 for k in runs};u,inv=np.unique(G,return_inverse=True);rng=np.random.default_rng(20261029);nboot=2000;counts=np.zeros((len(u),4,4))
for k,arm in enumerate('ABCD'):
 for idx in range(len(Y)):counts[inv[idx],k,2*Y[idx]+int(d[arm][idx])]+=1
bs=rng.multinomial(len(u),np.full(len(u),1/len(u)),size=nboot)@counts.reshape(len(u),-1);bs=bs.reshape(nboot,4,4);tn,fp,fn,tp=[bs[:,:,i] for i in range(4)];den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));mcc=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den),where=den>0)
comparisons={}
for x,z in [('A','B'),('B','C'),('C','D'),('A','D')]:
 dx, dz=d[x],d[z];difference=mcc[:,ord(z)-65]-mcc[:,ord(x)-65];comparisons[x+'->'+z]={'mcc_delta':float(matthews_corrcoef(Y,dz)-matthews_corrcoef(Y,dx)),'ci95_fixed_predictions':np.quantile(difference,[.025,.975]).tolist(),'corrected':int(((dx!=Y)&(dz==Y)).sum()),'harmed':int(((dx==Y)&(dz!=Y)).sum())}
groups={}
for label,mask in [('low_pos',(c<=1)&(Y==1)),('low_neg',(c<=1)&(Y==0)),('high_pos',(c>=6)&(Y==1)),('high_neg',(c>=6)&(Y==0)),('mid',(c>=2)&(c<=5)),('flag',flag),('unflag',~flag)]:groups[label]={'n':int(mask.sum()),**{k:int((d[k][mask]!=Y[mask]).sum()) for k in d}}
output={'n':len(Y),'n_groups':len(u),'comparisons':comparisons,'groups':groups,'verification':{'same_samples_order':True,'same_initial_head':True,'fusion_reconstructed':True,'weights_absent':True,'scope':'single fold seed; conditional on fixed historical 3D'}};(P/'audit.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf8');print(json.dumps(output,indent=2))
