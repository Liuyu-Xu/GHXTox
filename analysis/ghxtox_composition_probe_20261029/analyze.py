import json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error,roc_auc_score
b=json.loads((R/'analysis/error_evidence_linkage_20261025/all_training_linked.json').read_text(encoding='utf8'));r=json.loads((R/'analysis/cys_probability_adjustment_20261028/predictions.json').read_text(encoding='utf8'));assert len(b)==len(r)==6387 and all(u['sample_id']==v['sample_id'] for u,v in zip(b,r))
seq=[v['sequence'] for v in b];fold=np.array([v['fold'] for v in r]);group=np.array([v['group_id'] for v in r]);y=np.array([v['label'] for v in r]);p=np.array([v['A_probability'] for v in r]);th=.47022256531284556
letters='ACDEFGHIKLMNPQRSTVWY';lens=np.array([len(s) for s in seq]);cys=np.array([s.count('C') for s in seq]);X1=np.column_stack([lens,cys]);X2=np.column_stack([lens,[[s.count(a)/len(s) for a in letters] for s in seq]]) if False else np.array([[len(s)]+[s.count(a)/len(s) for a in letters] for s in seq]);res={}; pred={}
for name,X in [('cys_length',X1),('amino_acid_composition',X2)]:
 out=np.zeros(len(y));
 for f in range(5):
  train=fold!=f;test=~train;assert not set(group[train])&set(group[test]); model=RandomForestRegressor(n_estimators=100,min_samples_leaf=5,max_features=1.,n_jobs=2,random_state=20261029);model.fit(X[train],p[train]);out[test]=model.predict(X[test])
 pred[name]=out; d=(out>=th); original=(p>=th);truth=y==1;bad=original!=truth;corr=~bad
 res[name]={'r2_probability':float(r2_score(p,out)),'mae_probability':float(mean_absolute_error(p,out)),'agreement':float(np.mean(original==d)),'corrected_reproduced':int(np.sum(bad&(d==truth))),'new_errors':int(np.sum(corr&(d!=truth))),'original_errors':int(bad.sum()),'folds':[{'fold':f,'r2':float(r2_score(p[fold==f],out[fold==f])),'agreement':float(np.mean(original[fold==f]==d[fold==f]))} for f in range(5)]}
(P/'results.json').write_text(json.dumps(res,indent=2),encoding='utf8');print(json.dumps(res,indent=2))
