"""Post-hoc sensitivity: select each branch threshold on the existing group-isolated calibration selection half."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import matthews_corrcoef
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;B=R/'analysis/reviewer_response_20261102/six_seed_received_20261106';MAN=R/'analysis/reviewer_response_20261102/six_seed_nested_autodl_20261106/data/nested_folds.csv';OLD=R/'analysis/reviewer_response_20261102/nested_1d_received_20261106/nested_1d_results/runs';NEW=B/'six_seed_new_results/runs';TH=.47022256531284556;GRID=sorted(set(np.round(np.arange(.2,.801,.02),10))|{TH})
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def cal(p,c):
 p=np.clip(p,1e-6,1-1e-6);return expit(c[0]*np.log(p/(1-p))+c[1])
params={x['fold']:x for x in json.loads((B/'comparison/summary.json').read_text())['parameters']};man=read(MAN); threshold=[]
for fold in range(5):
 rows=[r for r in man if int(r['outer_fold'])==fold and r['role']=='calibration'];ids=[r['sample_id'] for r in rows];y=np.array([int(r['label']) for r in rows]);groups=np.array([r['group_id'] for r in rows]);fi,sel=next(StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=20261106+fold).split(np.zeros(len(y)),y,groups)); assert set(groups[fi]).isdisjoint(set(groups[sel]))
 ps=[]
 for seed in (42,123,2025,314,2718,2026):
  b=OLD if seed in (42,123,2025) else NEW;p=b/f'seed{seed}_fold{fold}/calibration_predictions.csv';rr=read(p);d={x['sample_id']:x for x in rr};assert len(d)==len(rr) and set(d)==set(ids)
  assert all(d[r['sample_id']]['sequence']==r['sequence'] and d[r['sample_id']]['label']==r['label'] and d[r['sample_id']]['group_id']==r['group_id'] for r in rows);ps.append([float(d[i]['probability']) for i in ids])
 rr=read(B/f'structure/fold{fold}/calibration.csv');d={x['sample_id']:x for x in rr};assert len(d)==len(rr) and set(d)==set(ids);assert all(d[r['sample_id']]['sequence']==r['sequence'] and d[r['sample_id']]['label']==r['label'] for r in rows)
 p1=cal(np.mean(ps,axis=0),params[fold]['platt_6']);p3=cal(np.array([float(d[i]['toxicity_probability']) for i in ids]),params[fold]['platt_structure'])
 def select(p):return max((matthews_corrcoef(y[sel],p[sel]>=t),-abs(t-TH),t) for t in GRID)[2]
 t1=select(p1);t3=select(p3);assert abs(t1-params[fold]['six_threshold'])<1e-10
 threshold.append({'fold':fold,'sequence_threshold':t1,'structure_threshold':t3,'fit_n':len(fi),'select_n':len(sel)})
 rows_held=[r for r in read(O/'branch_predictions.csv') if int(r['fold'])==fold]
 for r in rows_held:
  a=float(r['p_sequence'])>=t1;b=float(r['p_structure'])>=t3;yy=int(r['label']);r['category']='both_correct' if a==yy and b==yy else 'sequence_only_correct' if a==yy else 'structure_only_correct' if b==yy else 'both_wrong';r['sequence_pred']=int(a);r['structure_pred']=int(b)
 if fold==0: held=[]
 held.extend(rows_held)
assert len(held)==6387
tables=[]
for group in ('<4','>=4'):
 for label in (0,1):
  rr=[r for r in held if (int(r['cys_count'])<4)==(group=='<4') and int(r['label'])==label]; counts={k:sum(r['category']==k for r in rr) for k in ('both_correct','sequence_only_correct','structure_only_correct','both_wrong')};tables.append({'group':group,'label':label,'n':len(rr),**counts})
result={'status':'post-hoc sensitivity; thresholds selected only within each fold calibration select half, not blinded discovery','thresholds':threshold,'cys':tables,'total':{k:sum(r['category']==k for r in held) for k in ('both_correct','sequence_only_correct','structure_only_correct','both_wrong')}}
(O/'threshold_sensitivity.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
