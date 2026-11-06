import csv,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
with (P.parent/'expert_disagreement_20261025/training_expert_atlas.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==6387
y=np.array([int(r['label']) for r in rows]);fold=np.array([int(r['fold']) for r in rows]);groups=np.array([r['group_id'] for r in rows]);cys=np.array([int(r['cys_count']) for r in rows]);p=np.array([float(r['p_struct']) for r in rows]);p=np.clip(p,1e-6,1-1e-6);logit=np.log(p/(1-p))[:,None]
def columns(keys):
 def val(r,k):
  try:
   x=float(r[k]);return x if np.isfinite(x) else np.nan
  except (ValueError,KeyError,TypeError):return np.nan
 return np.array([[val(r,k) for k in keys] for r in rows])
basic=columns(['length','cys_count']);local=columns(['local_positive_excess','local_hydrophobic_excess','hydrophobic_contact_excess','positive_hydrophobic_contact_excess','charge_hydrophobic_separation','nonlocal_contact_density','ca_radius_gyration','mean_plddt'])
inputs={'score_only':logit,'score_basic':np.c_[logit,basic],'score_basic_local':np.c_[logit,basic,local]};preds={}
for name,X in inputs.items():
 v=np.zeros(len(y))
 for k in range(5):
  tr=fold!=k;ev=~tr;assert not set(groups[tr])&set(groups[ev]);model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=0.1,max_iter=3000));model.fit(X[tr],y[tr]);v[ev]=model.predict_proba(X[ev])[:,1]
 preds[name]=v
panels={'all':np.ones(len(y),bool),'low_positive':(cys<=1)&(y==1),'low_negative':(cys<=1)&(y==0),'high_negative':(cys>=6)&(y==0),'high_positive':(cys>=6)&(y==1)}
out={}
for key,m in panels.items():
 d={}
 for name,v in preds.items():
  z=v[m];yy=y[m];decision=z>=.5;d[name]={'n':int(m.sum()),'errors':int((decision!=yy).sum()),'mcc':float(matthews_corrcoef(yy,decision)) if len(np.unique(yy))==2 else None,'auroc':float(roc_auc_score(yy,z)) if len(np.unique(yy))==2 else None,'auprc':float(average_precision_score(yy,z)) if len(np.unique(yy))==2 else None}
 a=preds['score_basic']>=.5;b=preds['score_basic_local']>=.5;d['local_vs_basic']={'corrected':int((m&(a!=y)&(b==y)).sum()),'harmed':int((m&(a==y)&(b!=y)).sum())};out[key]=d
out['folds']={str(k):{'score_basic_errors':int((((preds['score_basic']>=.5)!=y)&(fold==k)).sum()),'score_basic_local_errors':int((((preds['score_basic_local']>=.5)!=y)&(fold==k)).sum())} for k in range(5)}
with (P/'feature_increment_predictions.csv').open('w',encoding='utf-8-sig',newline='') as h:
 w=csv.writer(h);w.writerow(['sample_id','fold','label','cys_count',*preds]);w.writerows([r['sample_id'],fold[i],y[i],cys[i],*[v[i] for v in preds.values()]] for i,r in enumerate(rows))
(P/'feature_increment_results.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
