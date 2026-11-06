"""Fixed OOF prediction paired group bootstrap; no fitting, no data filtering by model."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
P=Path(__file__).resolve().parent
with (P/'cases.csv').open(encoding='utf-8-sig',newline='') as f:r=[x for x in csv.DictReader(f) if x['split']=='training_oof']
assert len(r)==6387
y=np.array([int(x['label']) for x in r]);seq=np.array([float(x['p_seq6']) for x in r]);fu=np.array([float(x['p_fused6']) for x in r]);group=np.array([x['group_id'] for x in r]);unique=np.unique(group);chunks=[np.flatnonzero(group==g) for g in unique];rng=np.random.default_rng(20261106);res=[]
for _ in range(2000):
 idx=np.concatenate([chunks[i] for i in rng.integers(len(chunks),size=len(chunks))]);z=y[idx]
 if len(np.unique(z))!=2:raise ValueError('single-class resample')
 a=seq[idx];b=fu[idx];res.append([matthews_corrcoef(z,b>=.47022256531284556)-matthews_corrcoef(z,a>=.47022256531284556),roc_auc_score(z,b)-roc_auc_score(z,a),average_precision_score(z,b)-average_precision_score(z,a)])
z=np.array(res);out={'n_boot':2000,'seed':20261106,'resampling':'paired sequence group; not stratified by label or fold','deltas':{k:{'estimate':float(est),'ci95':np.quantile(z[:,i],[.025,.975]).tolist()} for i,(k,est) in enumerate([('mcc',matthews_corrcoef(y,fu>=.47022256531284556)-matthews_corrcoef(y,seq>=.47022256531284556)),('auroc',roc_auc_score(y,fu)-roc_auc_score(y,seq)),('auprc',average_precision_score(y,fu)-average_precision_score(y,seq))])},'limitation':'post-hoc fixed predictions only; no training, selection, or model uncertainty'}
(P/'oof_bootstrap.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
