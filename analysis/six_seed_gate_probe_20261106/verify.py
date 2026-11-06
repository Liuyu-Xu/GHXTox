"""Independent checks on saved predictions and group separation."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold
P=Path(__file__).resolve().parent;R=P.parents[1];MAN=R/'analysis/reviewer_response_20261102/six_seed_nested_autodl_20261106/data/nested_folds.csv';TH=.47022256531284556
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=read(P/'crossfit_outer_predictions.csv');report=json.loads((P/'crossfit_summary.json').read_text());manifest=read(MAN);assert len(rows)==6387 and len({r['sample_id'] for r in rows})==6387
for fold in range(5):
 dev=[r for r in manifest if int(r['outer_fold'])==fold and r['role']=='calibration'];test=[r for r in manifest if int(r['outer_fold'])==fold and r['role']=='test'];y=np.array([int(r['label']) for r in dev]);g=np.array([r['group_id'] for r in dev]);a,b=next(StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=20261106+fold).split(np.zeros(len(y)),y,g));assert not set(g[a])&set(g[b]) and not set(g)&{r['group_id'] for r in test}
 actual=[r for r in rows if int(r['fold'])==fold];by={r['sample_id']:r for r in actual};assert len(by)==len(actual)==len(test) and set(by)=={r['sample_id'] for r in test}
 for m in test:
  r=by[m['sample_id']];assert r['label']==m['label'] and r['sequence']==m['sequence'] and r['group_id']==m['group_id'] and int(r['cys'])==m['sequence'].count('C')
  p=float(r['p_sequence']);s=float(r['p_structure']);w=float(r['selected_fixed_3d_weight']);assert int(r['seq_pred'])==int(p>=TH) and int(r['fixed_pred'])==int(.72*p+.28*s>=TH) and int(r['selected_fixed_pred'])==int((1-w)*p+w*s>=TH)
  disagree=(p>=TH)!=(s>=TH);assert bool(int(r['disagreement']))==disagree
  for name in ('margin','probability','probability_cys_length'):
   choose=bool(int(r[name+'_choose_3d']));assert not choose or disagree
   assert int(r[name+'_pred'])==int((s if choose else p)>=TH)
   if name!='margin' and disagree and r[name+'_score']:assert choose==(float(r[name+'_score'])>.5)
y=np.array([int(r['label']) for r in rows])
for name,col in [('sequence','seq_pred'),('fixed_028','fixed_pred'),('selected_fixed','selected_fixed_pred'),('margin','margin_pred'),('probability','probability_pred'),('probability_cys_length','probability_cys_length_pred')]:
 d=np.array([int(r[col]) for r in rows]);assert abs(matthews_corrcoef(y,d)-report['methods'][name]['mcc'])<1e-12 and int((d!=y).sum())==report['methods'][name]['errors']
print(json.dumps({'passed':True,'n':len(rows),'folds':5,'checks':['group isolation','one-to-one identity','formula and routing consistency','reported MCC and error count'],'predictions_sha256':hashlib.sha256((P/'crossfit_outer_predictions.csv').read_bytes()).hexdigest(),'note':'checks inputs and outputs, not complete historical 3D training lineage'},indent=2))
