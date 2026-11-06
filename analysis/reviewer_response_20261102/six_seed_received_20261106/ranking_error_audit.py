"""Retrospective paired group bootstrap and error accounting; no fitting or selection."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef

HERE=Path(__file__).resolve().parent
P=HERE/'comparison/predictions.csv'
MANIFEST=HERE.parent/'six_seed_nested_autodl_20261106/data/nested_folds.csv'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=read(P);manifest=read(MANIFEST);lookup={(x['sample_id'],int(x['outer_fold'])):x for x in manifest if x['role']=='test'}
assert len(rows)==len(lookup)==6387 and len({x['sample_id'] for x in rows})==6387
for r in rows:
 m=lookup[(r['sample_id'],int(r['fold']))];assert r['label']==m['label'] and r['group_id']==m['group_id'];r['sequence']=m['sequence'];r['length']=len(m['sequence']);r['cys']=m['sequence'].count('C')
y=np.array([int(r['label']) for r in rows]); group=np.array([r['group_id'] for r in rows]); ids=np.unique(group); chunks=[np.flatnonzero(group==g) for g in ids];rng=np.random.default_rng(20261106);nboot=2000
result={'protocol':{'seed':20261106,'resamples':nboot,'sampling':'paired sequence-group bootstrap, no class/fold stratification','difference':'fusion minus six-seed','ranking':'pooled across folds, AP for AUPRC','warning':'exploratory after prior results; fixed predictions, does not include retraining or model selection uncertainty'},'prediction_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'modes':{}}
for mode in ('fixed','tuned'):
 f=np.array([float(r['fusion_'+mode]) for r in rows]);s=np.array([float(r['six_'+mode]) for r in rows])
 fd=(f>=.47022256531284556).astype(int) if mode=='fixed' else np.array([int(r['fusion_tuned_pred']) for r in rows]);sd=(s>=.47022256531284556).astype(int) if mode=='fixed' else np.array([int(r['six_tuned_pred']) for r in rows]); diffs=[]
 for _ in range(nboot):
  ix=np.concatenate([chunks[i] for i in rng.integers(0,len(chunks),len(chunks))]); assert len(np.unique(y[ix]))==2
  diffs.append((roc_auc_score(y[ix],f[ix])-roc_auc_score(y[ix],s[ix]),average_precision_score(y[ix],f[ix])-average_precision_score(y[ix],s[ix])))
 z=np.array(diffs);categories={};folds=[]
 for label in (0,1):
  for cat,mask in [('both_correct',(fd==y)&(sd==y)),('fusion_only_correct',(fd==y)&(sd!=y)),('six_only_correct',(fd!=y)&(sd==y)),('both_wrong',(fd!=y)&(sd!=y))]:categories[f'label{label}_{cat}']=int(((y==label)&mask).sum())
 for fold in range(5):
  ix=np.array([int(r['fold'])==fold for r in rows]);folds.append({'fold':fold,'n':int(ix.sum()),'auroc_delta':float(roc_auc_score(y[ix],f[ix])-roc_auc_score(y[ix],s[ix])),'auprc_delta':float(average_precision_score(y[ix],f[ix])-average_precision_score(y[ix],s[ix])),'mcc_delta':float(matthews_corrcoef(y[ix],fd[ix])-matthews_corrcoef(y[ix],sd[ix]))})
 result['modes'][mode]={'auroc_fusion_minus_six':float(roc_auc_score(y,f)-roc_auc_score(y,s)),'auroc_delta_ci95':np.quantile(z[:,0],[.025,.975]).tolist(),'auprc_fusion_minus_six':float(average_precision_score(y,f)-average_precision_score(y,s)),'auprc_delta_ci95':np.quantile(z[:,1],[.025,.975]).tolist(),'categories':categories,'folds':folds}
 for i,r in enumerate(rows):r[mode+'_case']='both_correct' if fd[i]==y[i] and sd[i]==y[i] else 'fusion_only_correct' if fd[i]==y[i] else 'six_only_correct' if sd[i]==y[i] else 'both_wrong'
(HERE/'ranking_error_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
with (HERE/'ranking_error_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(result['modes'],indent=2))
