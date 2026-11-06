import json,sys
from pathlib import Path
import numpy as np
P=Path('analysis/official_bcd_oof_20261030');R=Path.cwd();sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.linear_model import LogisticRegression
import csv
with (R/'original_protocol_bcd_preparation_20261029/data/folds.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
with (R/'original_protocol_bcd_preparation_20261029/data/structure_oof.csv').open(encoding='utf8',newline='') as f:sf=list(csv.DictReader(f))
y=np.array([int(r['label']) for r in rows]);fl=np.array([int(r['fold']) for r in rows]);groups=np.array([r['group_id'] for r in rows]);p3=np.array([float(s['toxicity_probability']) for s in sf]);d={}
for a in 'BCD':
 seeds=[]
 for seed in (42,123,2025):
  lookup={}
  for f in range(5):
   for r in json.loads((P/f'runs/{a}/seed{seed}/fold{f}/oof.json').read_text(encoding='utf8')):lookup[r['sample_id']]=r['probability']
  seeds.append(np.array([lookup[r['sample_id']] for r in rows]))
 raw=np.mean(seeds,axis=0);raw=np.clip(raw,1e-6,1-1e-6);l=np.log(raw)-np.log1p(-raw);cal=LogisticRegression(C=1e6,max_iter=1000).fit(l[:,None],y);d[a]=(.72*cal.predict_proba(l[:,None])[:,1]+.28*p3)>=.47022256531284556
orig=json.loads((R/'analysis/error_evidence_linkage_20261025/all_training_linked.json').read_text(encoding='utf8'));assert all(a['sample_id']==b['sample_id'] for a,b in zip(rows,orig));d['historical']=np.array([float(x['p_fusion'])>=.47022256531284556 for x in orig]);names=['historical','B','C','D'];rng=np.random.default_rng(20261030);s=np.zeros((2000,len(names),4))
for fold in range(5):
 mask=fl==fold;unique,inv=np.unique(groups[mask],return_inverse=True);yy=y[mask];cnt=np.zeros((len(unique),len(names),4))
 for j,name in enumerate(names):
  for i,(a,b) in enumerate(zip(yy,d[name][mask])):cnt[inv[i],j,2*a+int(b)]+=1
 s+=(rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000)@cnt.reshape(len(unique),-1)).reshape(2000,len(names),4)
tn,fp,fn,tp=[s[:,:,i] for i in range(4)];den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));m=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den),where=den>0);out={}
for a,b in [('historical','B'),('historical','C'),('historical','D'),('B','C'),('C','D')]:
 delta=m[:,names.index(b)]-m[:,names.index(a)];out[a+'->'+b]=np.quantile(delta,[.025,.975]).tolist()
(P/'paired_bootstrap.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))
