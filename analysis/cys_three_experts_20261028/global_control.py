"""Three global, group-bootstrap frozen-PLM linear models as capacity control."""
import os,sys,json,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,matthews_corrcoef,average_precision_score
from collections import defaultdict
def read(p):return json.loads(p.read_text(encoding='utf-8'))
meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json');pred={x['sample_id']:x for x in read(P/'predictions.json')};records=torch.load(R/'runs/prottox_tsk_v1/features.pt',map_location='cpu',weights_only=False)['records'];assert len(records)==len(meta)==6387 and all(a['sequence']==b['sequence'] and a['label']==int(b['label']) for a,b in zip(meta,records))
X=np.stack([x['tristat'].numpy().astype('float64') for x in records]);out=[];audit=[]
for f in range(5):
 ix={k:[x['source_index'] for x in roles if x['outer_fold']==f and x['role']==k] for k in ['train','validation','calibration','test']};groupmap=defaultdict(list)
 for i in ix['train']:groupmap[meta[i]['group_id']].append(i)
 gs=list(groupmap.values());sc=StandardScaler().fit(X[ix['train']]);xt=sc.transform(X[ix['train']]);xv=sc.transform(X[ix['validation']]);xc=sc.transform(X[ix['calibration']]);xe=sc.transform(X[ix['test']]);loc={i:j for j,i in enumerate(ix['train'])};yl=lambda ids:np.array([meta[i]['label'] for i in ids]);members=[];chosen=[]
 for seed in [42,123,2025]:
  rng=np.random.default_rng(seed);sample=[loc[i] for j in rng.integers(0,len(gs),len(gs)) for i in gs[j]];best=None
  for C in [.001,.01,.1]:
   model=LogisticRegression(C=C,class_weight='balanced',solver='liblinear',max_iter=500,random_state=seed).fit(xt[sample],yl(ix['train'])[sample]);key=(log_loss(yl(ix['validation']),model.predict_proba(xv)[:,1],labels=[0,1]),C)
   if best is None or key<best[0]:best=(key,model)
  chosen.append(best[0][1]);members.append(best[1])
 cal_logit=np.mean([m.decision_function(xc) for m in members],axis=0);platt=LogisticRegression(C=1,max_iter=500).fit(cal_logit[:,None],yl(ix['calibration']));p=platt.predict_proba(np.mean([m.decision_function(xe) for m in members],axis=0)[:,None])[:,1]
 for i,z in zip(ix['test'],p):
  a=pred[meta[i]['sample_id']];assert a['label']==meta[i]['label'] and a['fold']==f;out.append({'sample_id':a['sample_id'],'label':a['label'],'fold':f,'group_id':a['group_id'],'D_probability':float(z),'E_probability':float(.5*a['A_probability']+.5*z)})
 audit.append({'fold':f,'train_groups':len(gs),'selected_C':chosen});print('global control fold',f,'done',flush=True)
assert len(out)==6387 and len({x['sample_id'] for x in out})==6387
metrics={}
for model in ['D','E']:
 y=np.array([x['label'] for x in out]);prob=np.array([x[model+'_probability'] for x in out]);thr=.5 if model=='D' else .47022256531284556;decision=(prob>=thr).astype(int);metrics[model]={'n':len(y),'errors':int((y!=decision).sum()),'FP':int(((y==0)&(decision==1)).sum()),'FN':int(((y==1)&(decision==0)).sum()),'MCC':float(matthews_corrcoef(y,decision)),'AUPRC':float(average_precision_score(y,prob)),'Brier':float(np.mean((prob-y)**2))}
(P/'global_control_predictions.json').write_text(json.dumps(out,ensure_ascii=False),encoding='utf-8');(P/'global_control_metrics.json').write_text(json.dumps({'metrics':metrics,'audit':audit,'note':'3 bootstrap global models, same 3 classifiers count but each sees full sample pool; fixed .5 baseline combination; not matched FLOPs'},ensure_ascii=False,indent=2),encoding='utf-8');print('GLOBAL COMPLETE',json.dumps(metrics),flush=True)
