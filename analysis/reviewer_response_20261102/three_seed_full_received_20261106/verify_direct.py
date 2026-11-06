"""Recompute the fixed direct six-seed vs six-seed-plus-3D paired comparison."""
import csv,json,numpy as np
from pathlib import Path
from sklearn.metrics import matthews_corrcoef
P=Path(__file__).resolve().parent;out={};TH=.47022256531284556
for split in ('test1','test2'):
 with (P/f'{split}_predictions.csv').open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
 y=np.array([int(x['label']) for x in r]);a=np.array([float(x['six_sequence']) for x in r]);b=np.array([float(x['six_plus_3d']) for x in r]);p3=np.array([float(x['p3d']) for x in r]);assert np.allclose(b,.72*a+.28*p3,atol=1e-14)
 pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);rng=np.random.default_rng(20261106+(split=='test2'));ix=np.c_[rng.choice(pos,(5000,len(pos))),rng.choice(neg,(5000,len(neg)))]
 def boot(p):
  t=y[ix];d=(p[ix]>=TH);tp=((t==1)&d).sum(1);tn=((t==0)&~d).sum(1);fp=((t==0)&d).sum(1);fn=((t==1)&~d).sum(1);den=np.sqrt((tp+fp).astype(float)*(tp+fn)*(tn+fp)*(tn+fn));return np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den),where=den!=0)
 da=a>=TH;db=b>=TH;diff=boot(b)-boot(a)
 out[split]={'mcc_sixplus_minus_six':float(matthews_corrcoef(y,db)-matthews_corrcoef(y,da)),'ci95':np.quantile(diff,[.025,.975]).tolist(),'sixplus_only_correct':int(((db==y)&(da!=y)).sum()),'six_only_correct':int(((da==y)&(db!=y)).sum())}
(P/'direct_six_vs_sixplus.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
