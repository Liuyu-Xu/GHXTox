import csv,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
with (R/'data/folds/train_cdhit080_fallback_folds.csv').open(encoding='utf8') as h:rows={r['sample_id']:r for r in csv.DictReader(h)}
results=[]
for fold in range(5):
 z=np.load(P/f'spatial_fold{fold}.npz'); kept=[i for i,s in enumerate(z['ids']) if rows[s]['sequence'].count('C')>=2]; ids=z['ids'][kept];X=z['features'][kept];seq=[rows[s]['sequence'] for s in ids]; pos=[[i for i,a in enumerate(s) if a=='C'] for s in seq];length=np.array([len(s) for s in seq]);count=np.array([len(q) for q in pos]);groups=[rows[s]['group_id'] for s in ids];targets={'first_position':np.array([p[0]/(len(s)-1) for p,s in zip(pos,seq)]),'span':np.array([(p[-1]-p[0])/(len(s)-1) for p,s in zip(pos,seq)])};tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=42).split(X,groups=groups)); entry={'fold':fold,'n':len(ids),'test_n':len(te)}
 for name,y in targets.items():
  for model,F in [('length_count',np.c_[length,count]),('spatial_length_count',np.c_[length,count,X])]:
   reg=make_pipeline(StandardScaler(),Ridge(alpha=100));reg.fit(F[tr],y[tr]);entry[name+'_'+model+'_mae']=float(mean_absolute_error(y[te],reg.predict(F[te])))
 results.append(entry)
(P/'position_probe_results.json').write_text(json.dumps(results,indent=2),encoding='utf8');print(json.dumps(results,indent=2))
