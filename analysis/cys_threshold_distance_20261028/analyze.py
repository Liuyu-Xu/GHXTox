import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1]
rows=json.loads((R/'analysis/cys_probability_adjustment_20261028/predictions.json').read_text(encoding='utf-8'))
y=np.array([r['label'] for r in rows]); c=np.array([r['cys_count'] for r in rows]); p=np.array([r['A_probability'] for r in rows]); f=np.array([r['fold'] for r in rows]); threshold=0.47022256531284556
pred=p>=threshold; err=pred!=y; dist=np.abs(p-threshold)
groups={'low_positive':(c<=1)&(y==1),'low_negative':(c<=1)&(y==0),'high_positive':(c>=6)&(y==1),'high_negative':(c>=6)&(y==0)}
def dist_counts(mask):
 return {'total':int(mask.sum()),'error':int((mask&err).sum()),'correct':int((mask&~err).sum()),'error_distance_bins':{'under_0.05':int((mask&err&(dist<.05)).sum()),'0.05_to_0.15':int((mask&err&(dist>=.05)&(dist<.15)).sum()),'at_least_0.15':int((mask&err&(dist>=.15)).sum())},'correct_distance_bins':{'under_0.05':int((mask&~err&(dist<.05)).sum()),'0.05_to_0.15':int((mask&~err&(dist>=.05)&(dist<.15)).sum()),'at_least_0.15':int((mask&~err&(dist>=.15)).sum())},'error_median_probability':float(np.median(p[mask&err])) if (mask&err).sum() else None}
counts={g:dist_counts(m) for g,m in groups.items()}
scenarios={}
for name,mask,sign in [('low',c<=1,1),('high',c>=6,-1)]:
 scenarios[name]={}
 for delta in [.025,.05,.1,.15]:
  q=np.clip(p+np.where(mask,sign*delta,0),0,1); pr=q>=threshold
  scenarios[name][str(delta)]={'corrected':int((err&(pr==y)).sum()),'harmed':int((~err&(pr!=y)).sum()),'corrected_target':int((err&(pr==y)&groups['low_positive' if name=='low' else 'high_negative']).sum()),'harmed_opposite':int((~err&(pr!=y)&groups['low_negative' if name=='low' else 'high_positive']).sum())}
folds={str(k):{g:dist_counts(mask&(f==k)) for g,mask in groups.items()} for k in range(5)}
output={'threshold':threshold,'n':len(y),'distance_bins':'[0,0.05), [0.05,0.15), [0.15,infinity)','counts':counts,'scenarios_descriptive_only':scenarios,'folds':folds,'limitation':'Historical original upstream OOF is not fully nested. Directions and groups were informed by prior development. Test sets unused. Exploratory fixed changes are not independently validated.'}
(P/'results.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'counts':counts,'scenarios':scenarios},indent=2))
