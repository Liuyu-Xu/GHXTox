import csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def metric(y, pred, prob):
    return {'MCC': float(matthews_corrcoef(y,pred)), 'AUROC': float(roc_auc_score(y,prob)), 'AUPRC': float(average_precision_score(y,prob)), 'sensitivity': float(np.mean(pred[y==1])), 'specificity': float(np.mean(1-pred[y==0]))}
rng = np.random.default_rng(20261106)
results = {}
for split in ('test1','test2'):
    gh = read(root/f'runs/default_no_chemical_20261106/{split}_predictions.csv')
    gd = {r['sample_id']:r for r in gh}
    assert len(gd)==len(gh)
    for name, path, probcol, predcol in (
        ('StrucToxNet',root/f'analysis/reviewer_response_20261102/structoxnet_results_20261105/{split}_paired.csv','structoxnet_probability','structoxnet_prediction'),
        ('PeptiTox',root/f'analysis/reviewer_response_20261102/peptitox_full3seed_{split}_predictions.csv','toxicity_probability','prediction')):
        rows=read(path); ids=[r['sample_id'] for r in rows]
        assert len(set(ids))==len(ids) and set(ids)==set(gd)
        assert all(int(r['label'])==int(gd[r['sample_id']]['label']) for r in rows)
        y=np.array([int(r['label']) for r in rows]); g=np.array([int(gd[i]['prediction']) for i in ids]); p=np.array([int(r[predcol]) for r in rows]); gs=np.array([float(gd[i]['toxicity_probability']) for i in ids]); ps=np.array([float(r[probcol]) for r in rows])
        assert np.isfinite(gs).all() and np.isfinite(ps).all() and ((gs>=0)&(gs<=1)&(ps>=0)&(ps<=1)).all()
        pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0); delta=[]
        for _ in range(2000):
            ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
            delta.append(matthews_corrcoef(y[ix],g[ix])-matthews_corrcoef(y[ix],p[ix]))
        results[f'{split}_{name}']={'n':len(y),'positive':len(pos),'current':metric(y,g,gs),'competitor':metric(y,p,ps),'delta_mcc':float(matthews_corrcoef(y,g)-matthews_corrcoef(y,p)),'delta_mcc_ci95':np.quantile(delta,[.025,.975]).tolist()}
(out/'current_competitor_comparison.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print(json.dumps(results,indent=2))
