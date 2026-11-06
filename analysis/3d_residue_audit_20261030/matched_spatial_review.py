import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;A=P.parent/'expert_disagreement_20261025/training_expert_atlas.csv';Q=P.parent/'both_wrong_order_matched_20261030/matched_pairs.csv'
with A.open(encoding='utf-8-sig',newline='') as f:atlas={r['sample_id']:r for r in csv.DictReader(f)}
with Q.open(encoding='utf-8-sig',newline='') as f:pairs=list(csv.DictReader(f))
features=['mean_plddt','local_positive_excess','local_hydrophobic_excess','hydrophobic_contact_excess','positive_hydrophobic_contact_excess','charge_hydrophobic_separation','nonlocal_contact_density','ca_radius_gyration']
out=[]
for group in ('low_pos','high_neg'):
 for feat in features:
  ds=[];folds={}
  for r in pairs:
   if r['analysis_group']!=group:continue
   try:a=float(atlas[r['wrong_id']][feat]);b=float(atlas[r['control_id']][feat])
   except (KeyError,ValueError,TypeError):continue
   if not np.isfinite(a) or not np.isfinite(b):continue
   d=a-b;ds.append(d);folds.setdefault(r['fold'],[]).append(d)
  out.append({'group':group,'feature':feat,'n_pairs':len(ds),'median_wrong_minus_correct':float(np.median(ds)) if ds else None,'wrong_greater':sum(d>0 for d in ds),'wrong_less':sum(d<0 for d in ds),'ties':sum(d==0 for d in ds),'fold_median':{f:float(np.median(v)) for f,v in folds.items()}})
(P/'matched_spatial_summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8');print(json.dumps(out,indent=2))
