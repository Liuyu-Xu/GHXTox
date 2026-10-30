import json,csv,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
read=lambda p:json.loads(p.read_text(encoding='utf8'))
with (R/'original_protocol_bcd_preparation_20261029/data/folds.csv').open(encoding='utf8',newline='') as f:folds=list(csv.DictReader(f))
with (R/'original_protocol_bcd_preparation_20261029/data/structure_oof.csv').open(encoding='utf8',newline='') as f:struct=list(csv.DictReader(f))
meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');assert len(folds)==len(meta)==len(struct)==6387 and all(x['sample_id']==m['sample_id']==s['sample_id'] and x['label']==s['label'] and x['group_id']==m['group_id'] for x,m,s in zip(folds,meta,struct));y=np.array([int(x['label']) for x in folds]);cs=np.array([x['cys_count'] for x in meta]);fids=np.array([int(x['fold']) for x in folds]);flags=np.array([x['uniform_flag'] for x in meta],dtype=bool);groups=np.array([x['group_id'] for x in folds]);p3=np.array([float(s['toxicity_probability']) for s in struct]);t=.47022256531284556
raw={};logs={}
for arm in 'BCD':
 raw[arm]={}
 for seed in [42,123,2025]:
  data={}
  for f in range(5):
   path=P/f'runs/{arm}/seed{seed}/fold{f}';res=read(path/'result.json');rr=read(path/'oof.json');expected=[x for x in folds if int(x['fold'])==f];assert res['n_train']==6387-len(expected) and res['n_eval']==len(expected) and len(rr)==len(expected) and all(x['sample_id']==z['sample_id'] and int(x['label'])==z['label'] and z['fold']==f for x,z in zip(expected,rr));logs[f'{arm}_{seed}_{f}']=res['initial_head_sha256'];data.update({z['sample_id']:z['probability'] for z in rr})
  raw[arm][seed]=np.array([data[x['sample_id']] for x in folds])
for seed in [42,123,2025]:
 for f in range(5):assert len({logs[f'{a}_{seed}_{f}'] for a in 'BCD'})==1
A={}
for f in range(5):
 path=R/f'analysis/retrained_a_seed42_20261029/runs/A/seed42/fold{f}';ar=read(path/'result.json');logs[f'A_42_{f}']=ar['initial_head_sha256'];assert ar['initial_head_sha256']==logs[f'B_42_{f}'];A.update({r['sample_id']:r['probability'] for r in read(path/'oof.json')})
raw['A']={42:np.array([A[x['sample_id']] for x in folds])};orig=[]
for seed in [42,123,2025]:
 z=read(R/f'runs/lora_mamba_full/oof_seed{seed}.json');assert z['sample_ids']==[x['sample_id'] for x in folds];orig.append(z['oof'])
raw['historical']={'three':np.mean(orig,axis=0)}
def logit(x):
 x=np.clip(x,1e-6,1-1e-6);return np.log(x)-np.log1p(-x)
models={};calibrators={}
for a,v in raw.items():
 arrangements={'single42':v[42]} if a in 'ABCD' else {}
 if a in 'BCD':arrangements['three']=np.mean([v[k] for k in [42,123,2025]],axis=0)
 if a=='historical':arrangements['three']=v['three']
 for kind,arr in arrangements.items():
  cal=LogisticRegression(C=1e6,max_iter=1000).fit(logit(arr)[:,None],y);assert cal.coef_[0,0]>0;p1=cal.predict_proba(logit(arr)[:,None])[:,1];models[a+'_'+kind]=.72*p1+.28*p3;calibrators[a+'_'+kind]={'slope':float(cal.coef_[0,0]),'intercept':float(cal.intercept_[0])}
def metrics(mask):
 yy=y[mask];out={}
 for name,p in models.items():
  z=p[mask];d=z>=t;out[name]={'n':int(mask.sum()),'MCC':float(matthews_corrcoef(yy,d)) if len(np.unique(yy))==2 else None,'AUPRC':float(average_precision_score(yy,z)) if len(np.unique(yy))==2 else None,'errors':int((yy!=d).sum()),'FN':int(((yy==1)&~d).sum()),'FP':int(((yy==0)&d).sum())}
 return out
panels={'all':np.ones(len(y),dtype=bool),'low_pos':(cs<=1)&(y==1),'low_neg':(cs<=1)&(y==0),'high_pos':(cs>=6)&(y==1),'high_neg':(cs>=6)&(y==0),'flagged':flags,'unflagged':~flags};results={name:metrics(mask) for name,mask in panels.items()};fold_results={str(f):metrics(fids==f) for f in range(5)};comp={}
for x,z in [('A_single42','B_single42'),('B_single42','C_single42'),('C_single42','D_single42'),('A_single42','C_single42'),('historical_three','B_three'),('historical_three','C_three'),('historical_three','D_three')]:
 a=models[x]>=t;b=models[z]>=t;comp[x+'__'+z]={'corrected':int(((a!=y)&(b==y)).sum()),'harmed':int(((a==y)&(b!=y)).sum())}
out={'panels':results,'folds':fold_results,'comparisons':comp,'calibrators':calibrators,'init_match_A42_BCD42':True,'n':6387,'limitation':'global OOF Platt uses all OOF labels; historical 3D and threshold not fully nested; no Test1/Test2; predictions only'};(P/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(out,ensure_ascii=False,indent=2))
