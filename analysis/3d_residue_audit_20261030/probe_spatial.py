import csv,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;root=P.parents[1];sys.path.insert(0,str(root/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import roc_auc_score,mean_absolute_error,r2_score
with (root/'data/folds/train_cdhit080_fallback_folds.csv').open(encoding='utf8',newline='') as f: rows=list(csv.DictReader(f))
lookup={r['sample_id']:r for r in rows}; ids=[]; features=[]; fs=[]
for f in range(5):
 z=np.load(P/f'spatial_fold{f}.npz');ids.extend(z['ids'].tolist());features.extend(z['features']);fs.extend([f]*len(z['ids']))
assert len(set(ids))==len(rows)==6387 and set(ids)==set(lookup)
X=np.asarray(features); f=np.asarray(fs); lengths=np.array([len(lookup[s]['sequence']) for s in ids]);c=np.array([lookup[s]['sequence'].count('C') for s in ids]); binary=(c>=6).astype(int)
out={}
for name,Z in [('length',lengths[:,None]),('spatial_plus_length',np.c_[lengths,X])]:
 pred=np.zeros(len(ids));count=np.zeros(len(ids))
 for fold in range(5):
  tr=f!=fold;ev=~tr
  cl=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000));cl.fit(Z[tr],binary[tr]);pred[ev]=cl.predict_proba(Z[ev])[:,1]
  reg=make_pipeline(StandardScaler(),Ridge(alpha=100));reg.fit(Z[tr],c[tr]);count[ev]=reg.predict(Z[ev])
 out[name]={'high_cys_auroc':float(roc_auc_score(binary,pred)),'cys_count_mae':float(mean_absolute_error(c,count)),'cys_count_r2':float(r2_score(c,count)),'fold_auroc':[float(roc_auc_score(binary[f==k],pred[f==k])) for k in range(5)]}
(P/'spatial_probe_results.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
