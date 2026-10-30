import csv,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;root=P.parents[1]
sys.path.insert(0,str(root/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
with (P.parent/'expert_disagreement_20261025/training_expert_atlas.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
cohort=json.loads((P.parent/'saprot_order_20261027/cohort.json').read_text(encoding='utf8')); ids={r['sample_id'] for r in cohort}
aa='ACDEFGHIKLMNPQRSTVWY';lookup={v:i for i,v in enumerate(aa)}
def feat(s):
 n=len(s); comp=np.zeros(21); pairs=np.zeros(400); pos=[i for i,a in enumerate(s) if a=='C']
 for a in s:
  if a in lookup:comp[lookup[a]]+=1/n
 comp[20]=n
 for a,b in zip(s,s[1:]):
  if a in lookup and b in lookup:pairs[lookup[a]*20+lookup[b]]+=1/max(n-1,1)
 ordered=[pos[0]/max(n-1,1) if pos else 0,pos[-1]/max(n-1,1) if pos else 0,(pos[-1]-pos[0])/max(n-1,1) if len(pos)>1 else 0]
 return comp,np.r_[pairs,ordered]
X,Z=zip(*(feat(r['sequence']) for r in rows));X=np.array(X);Z=np.array(Z);y=np.array([int(r['label']) for r in rows]);fold=np.array([int(r['fold']) for r in rows]);cs=np.array([int(r['cys_count']) for r in rows]);groups=np.array([r['group_id'] for r in rows]);probs={}
for kind,F in [('composition',X),('order',np.c_[X,Z])]:
 p=np.zeros(len(y))
 for k in range(5):
  tr=fold!=k;ev=~tr;assert not set(groups[tr])&set(groups[ev]);model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42));model.fit(F[tr],y[tr]);p[ev]=model.predict_proba(F[ev])[:,1]
 probs[kind]=p
def met(p,mask):
 yy=y[mask];v=p[mask];pred=v>=.5
 return {'n':int(mask.sum()),'errors':int((pred!=yy).sum()),'MCC':float(matthews_corrcoef(yy,pred)) if len(set(yy))==2 else None,'AUROC':float(roc_auc_score(yy,v)) if len(set(yy))==2 else None,'AUPRC':float(average_precision_score(yy,v)) if len(set(yy))==2 else None}
panels={'all':np.ones(len(y),dtype=bool),'low_positive':(cs<=1)&(y==1),'low_negative':(cs<=1)&(y==0),'high_negative':(cs>=6)&(y==0),'high_positive':(cs>=6)&(y==1)}
summ={}
for name,m in panels.items():
 a=probs['composition']>=.5;b=probs['order']>=.5;summ[name]={'composition':met(probs['composition'],m),'order':met(probs['order'],m),'corrected':int((m&(a!=y)&(b==y)).sum()),'harmed':int((m&(a==y)&(b!=y)).sum())}
both={r['sample_id'] for r in rows if r['expert_group']=='both_wrong'}; summ['previous_perturbation_coverage']={'cohort_total':len(ids),'both_wrong_total':len(both),'intersection':len(ids&both)}
with (P/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['sample_id','fold','label','cys_count','composition_probability','order_probability']);w.writerows((r['sample_id'],fold[i],y[i],cs[i],probs['composition'][i],probs['order'][i]) for i,r in enumerate(rows))
(P/'summary.json').write_text(json.dumps(summ,indent=2,ensure_ascii=False),encoding='utf8');print(json.dumps(summ,indent=2))
