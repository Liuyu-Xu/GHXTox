import json,csv,sys
from pathlib import Path
import numpy as np
P=Path('analysis/autodl_cys_test_retro_20261029');sys.path.insert(0,'analysis/repository_sync_20261026/validation_numpy1_clean')
from sklearn.metrics import matthews_corrcoef
r=json.loads((P/'test1_predictions.json').read_text(encoding='utf8'))
with Path('analysis/test_cys_length_20261028/test2_samples.csv').open(encoding='utf-8-sig',newline='') as f:seq={x['sequence'] for x in csv.DictReader(f)}
output={}; rng=np.random.default_rng(20261029)
for split,rows in [('test1',r),('test2',[x for x in r if x['sequence'] in seq])]:
 y=np.array([x['label'] for x in rows]);p={a:np.array([x[a]>=.47022256531284556 for x in rows]) for a in ['A','B','C','D','historical']};ref=p['historical'];out={}
 pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
 for a in 'ABCD':
  v=p[a];delta=[]
  for _ in range(2000):
   ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)];delta.append(matthews_corrcoef(y[ix],v[ix])-matthews_corrcoef(y[ix],ref[ix]))
  out[a]={'corrected':int(((ref!=y)&(v==y)).sum()),'harmed':int(((ref==y)&(v!=y)).sum()),'delta_MCC':float(matthews_corrcoef(y,v)-matthews_corrcoef(y,ref)),'CI95_stratified_paired':np.quantile(delta,[.025,.975]).tolist()}
 output[split]=out
(P/'comparisons.json').write_text(json.dumps(output,indent=2),encoding='utf8');print(json.dumps(output,indent=2))
