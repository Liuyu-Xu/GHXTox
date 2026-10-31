import json
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parent;R=P.parents[1]
a=json.loads((R/'analysis/saprot_fullmodel_fusion_20261027/all_sample_predictions.json').read_text(encoding='utf8'))
assert len(a)==6387
out={}
for group,fn in [('all',lambda r:True),('low_pos',lambda r:r['sequence'].count('C')<=1 and r['label']==1),('low_neg',lambda r:r['sequence'].count('C')<=1 and r['label']==0),('high_neg',lambda r:r['sequence'].count('C')>=6 and r['label']==0),('high_pos',lambda r:r['sequence'].count('C')>=6 and r['label']==1)]:
 rows=[r for r in a if fn(r)];out[group]={}
 for kind in ['masked','3Di']:
  for approach in ['saprot','fusion20']:
   counts=Counter();folds={str(k):Counter() for k in range(5)}
   for r in rows:
    y=r['label'];base=r['old_cal_prediction'];mix=r[f'old_{kind}_w20_probability'];prob=(mix-.8*r['old_cal_probability'])/.2
    assert -1e-8<=prob<=1+1e-8
    other=int(prob>=.5) if approach=='saprot' else r[f'old_{kind}_w20_prediction']
    status=('corrected' if base!=y and other==y else 'harmed' if base==y and other!=y else 'both_wrong' if base!=y else 'both_correct');counts[status]+=1;folds[str(r['fold'])][status]+=1
   out[group][kind+'_'+approach]={'n':len(rows),'counts':dict(counts),'folds':{k:dict(v) for k,v in folds.items()}}
(P/'results.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
