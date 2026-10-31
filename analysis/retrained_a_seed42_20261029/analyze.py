import json,csv,sys
from pathlib import Path
import numpy as np
P=Path('analysis/retrained_a_seed42_20261029');root=Path.cwd();sys.path.insert(0,str(root/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
read=lambda p:json.loads(p.read_text(encoding='utf8'))
with open('original_protocol_bcd_preparation_20261029/data/folds.csv',encoding='utf8',newline='') as f:fold=list(csv.DictReader(f))
with open('original_protocol_bcd_preparation_20261029/data/structure_oof.csv',encoding='utf8',newline='') as f:s=list(csv.DictReader(f))
old=read(Path('runs/lora_mamba_full/oof_seed42.json'));new={}
for f in range(5):
 for x in read(P/f'runs/A/seed42/fold{f}/oof.json'):
  assert x['sample_id'] not in new and x['fold']==f;new[x['sample_id']]=x
assert len(new)==len(fold)==len(s)==6387 and old['sample_ids']==[x['sample_id'] for x in fold] and all(x['sample_id']==z['sample_id'] and int(x['label'])==z['label'] and int(x['fold'])==z['fold'] and x['sample_id']==u['sample_id'] for x,z,u in zip(fold,[new[x['sample_id']] for x in fold],s))
y=np.array([int(x['label']) for x in fold]);raw={'old':np.array(old['oof']),'new':np.array([new[x['sample_id']]['probability'] for x in fold])};q3=np.array([float(x['toxicity_probability']) for x in s]);threshold=.47022256531284556;out={};pred={}
for name,p in raw.items():
 p=np.clip(p,1e-6,1-1e-6);log=np.log(p)-np.log1p(-p);c=LogisticRegression(C=1e6,max_iter=1000).fit(log[:,None],y);assert c.coef_[0,0]>0;pr=c.predict_proba(log[:,None])[:,1];z=.72*pr+.28*q3;d=z>=threshold;pred[name]=d;out[name]={'MCC':float(matthews_corrcoef(y,d)),'AUPRC':float(average_precision_score(y,z)),'AUROC':float(roc_auc_score(y,z)),'errors':int((d!=y).sum()),'FN':int(((y==1)&~d).sum()),'FP':int(((y==0)&d).sum()),'slope':float(c.coef_[0,0]),'intercept':float(c.intercept_[0])}
out['paired']={'corrected':int(((pred['old']!=y)&(pred['new']==y)).sum()),'harmed':int(((pred['old']==y)&(pred['new']!=y)).sum())};(P/'oof_comparison.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
