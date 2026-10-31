from pathlib import Path
import json,collections
import numpy as np
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,x):(P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');md={r['sample_id']:r for r in meta}
ARMS=read(P/'execution_status.json')['expected_completed_arms'] if (P/'execution_status.json').exists() else ['A','B','C']
runs={a:read(P/'runs'/a/'result.json') for a in ARMS}
ids=[r['sample_id'] for r in runs['A']['predictions']]
assert all([r['sample_id'] for r in run['predictions']]==ids for run in runs.values())
y=np.array([md[k]['label'] for k in ids]);panels={'all':list(range(len(ids))),'flagged':[i for i,k in enumerate(ids) if md[k]['uniform_flag']],'unflagged_not_gold':[i for i,k in enumerate(ids) if not md[k]['uniform_flag']]}
for key in range(4):panels['physical_'+str(key)]=[i for i,k in enumerate(ids) if 'physical_'+str(key) in md[k]['category_keys']]
for family,title in [('belongs to the mcd family','MCD'),('belongs to the non-disulfide-bridged peptide (ndbp) superfamily','NDBP')]:
 panels[title]=[i for i,k in enumerate(ids) if md[k]['current_family_sidecar']['parent_families']==[family]]
for f in sorted({f for k in ids for f in md[k]['current_family_sidecar']['parent_families']}):
 for label in [0,1]:
  ix=[i for i,k in enumerate(ids) if md[k]['label']==label and md[k]['current_family_sidecar']['parent_families']==[f]]
  if ix:panels[f'family_label{label}:'+f]=ix
metrics={};decisions={}
for mode in ['sequence','fusion']:
 metrics[mode]={};decisions[mode]={}
 for arm,run in runs.items():
  prob=np.array([r[mode+'_probability'] for r in run['predictions']]);d=np.array([r[mode+'_prediction'] for r in run['predictions']]);decisions[mode][arm]=d
  metrics[mode][arm]={}
  for name,ix in panels.items():
   yy=y[ix];dd=d[ix];pp=prob[ix]
   q={'n':len(ix),'TP':int(((yy==1)&(dd==1)).sum()),'TN':int(((yy==0)&(dd==0)).sum()),'FP':int(((yy==0)&(dd==1)).sum()),'FN':int(((yy==1)&(dd==0)).sum()),'errors':int((yy!=dd).sum())}
   two=len(set(yy.tolist()))==2
   q.update(MCC=float(matthews_corrcoef(yy,dd)) if two else None,AUROC=float(roc_auc_score(yy,pp)) if two else None,AUPRC=float(average_precision_score(yy,pp)) if two else None)
   metrics[mode][arm][name]=q
save('metrics.json',metrics)
g=sorted({md[k]['group_id'] for k in ids});gi={k:i for i,k in enumerate(g)};groups=np.array([gi[md[k]['group_id']] for k in ids])
def mcc(cm):
 tn,fp,fn,tp=[cm[...,i] for i in range(4)];den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));return np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den,dtype=float),where=den>0)
comps={}
for mode in metrics:
 arrays=[]
 for a in ARMS:
  arr=np.zeros((len(g),4));np.add.at(arr,(groups,2*y+decisions[mode][a]),1);arrays.append(arr)
 flat=np.concatenate(arrays,axis=1);rng=np.random.default_rng(20261027);boot=[]
 for k in range(0,2000,100):
  w=rng.multinomial(len(g),np.full(len(g),1/len(g)),size=100);boot.append(mcc((w@flat).reshape(-1,len(ARMS),4)))
 boot=np.concatenate(boot);comps[mode]=[]
 for a,b in [(a,b) for a,b in [('A','B'),('B','C'),('A','C')] if a in ARMS and b in ARMS]:
  aa,bb=decisions[mode][a],decisions[mode][b];corr=np.flatnonzero((aa!=y)&(bb==y));harm=np.flatnonzero((aa==y)&(bb!=y));ds=boot[:,ARMS.index(b)]-boot[:,ARMS.index(a)]
  comps[mode].append({'control':a,'treatment':b,'delta_MCC':metrics[mode][b]['all']['MCC']-metrics[mode][a]['all']['MCC'],'CI95_group_bootstrap':np.quantile(ds,[.025,.975]).tolist(),'CI97_5_two_primary_contrasts':np.quantile(ds,[.0125,.9875]).tolist() if len(ARMS)>2 and (a,b) in [('A','B'),('B','C')] else None,'corrected':len(corr),'harmed':len(harm),'net_fewer_errors':len(corr)-len(harm),'corrected_ids':[ids[i] for i in corr],'harmed_ids':[ids[i] for i in harm],'scope':'One fold,one seed,historical development population; conditional fixed-prediction uncertainty only, no seed/fold selection uncertainty.'})
save('comparisons.json',comps)
save('runtime.json',{a:{'seconds':r['seconds'],'selected_epoch':r['selected_epoch'],'peak_cuda_bytes':r['peak_cuda_bytes']} for a,r in runs.items()})
print('METRICS',{mode:{a:q['all'] for a,q in arms.items()} for mode,arms in metrics.items()},flush=True)
print('CONTRASTS',{mode:[{k:v for k,v in q.items() if not k.endswith('_ids')} for q in cs] for mode,cs in comps.items()},flush=True)
