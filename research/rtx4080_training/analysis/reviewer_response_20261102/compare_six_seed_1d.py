"""Cross-fit a six-sequence vs three-sequence + one saved 3D expert comparison."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'analysis/reviewer_response_20261102/six_seed_1d'
SEEDS=(42,123,2025,314,2718,2026)
GRID=(0.,.25,.50,.72,.85,1.)
THRESHOLD=.47022256531284556


def logits(x):
    x=np.clip(x,1e-6,1-1e-6)
    return np.log(x/(1-x)).reshape(-1,1)


def metrics(y,p):
    return {'MCC':float(matthews_corrcoef(y,p>=THRESHOLD)),
            'AUROC':float(roc_auc_score(y,p)), 'AUPRC':float(average_precision_score(y,p))}


def main():
    with (ROOT/'data/folds/train_cdhit080_fallback_folds.csv').open(newline='',encoding='utf-8-sig') as f:
        rows=sorted(csv.DictReader(f),key=lambda r:int(r['source_index']))
    n=len(rows)
    if n != 6387 or [int(r['source_index']) for r in rows] != list(range(n)):
        raise ValueError('Fold manifest is incomplete')
    y=np.array([int(r['label']) for r in rows]); fold=np.array([int(r['fold']) for r in rows])
    ids=[r['sample_id'] for r in rows]
    probs=np.full((len(SEEDS),n),np.nan)
    for si,seed in enumerate(SEEDS):
        for k in range(5):
            file=RUN/'runs'/f'seed{seed}_fold{k}'/'result.json'
            d=json.loads(file.read_text(encoding='utf-8'))
            idx=np.asarray(d['source_indices'],dtype=int)
            if d['smoke'] or d['epochs']!=15 or d['seed']!=seed or d['fold']!=k or not np.array_equal(idx,np.flatnonzero(fold==k)):
                raise ValueError(f'Invalid held-out rows: {file}')
            if d['sample_ids']!=[ids[i] for i in idx] or d['labels']!=y[idx].tolist():
                raise ValueError(f'ID or label mismatch: {file}')
            probs[si,idx]=d['probabilities']
    if not np.isfinite(probs).all(): raise ValueError('Missing/non-finite prediction')
    with (ROOT/'runs/conformer_ensemble_residual_oof/full_calibrated_predictions.csv').open(newline='',encoding='utf-8-sig') as f:
        structs=sorted(csv.DictReader(f),key=lambda r:int(r['source_index']))
    if len(structs)!=n or any(int(r['source_index'])!=i or r['sample_id']!=ids[i] or int(r['label'])!=int(y[i]) or int(r['fold'])!=int(fold[i]) for i,r in enumerate(structs)):
        raise ValueError('Historical 3D predictions do not align')
    p3d=np.array([float(r['toxicity_probability']) for r in structs])
    raw3=probs[:3].mean(axis=0); raw6=probs.mean(axis=0)
    o3=np.zeros(n); o6=np.zeros(n); weights=[]
    for k in range(5):
        tr=fold!=k; te=~tr
        cal3=LogisticRegression(C=1e6,max_iter=1000).fit(logits(raw3[tr]),y[tr])
        cal6=LogisticRegression(C=1e6,max_iter=1000).fit(logits(raw6[tr]),y[tr])
        q3=cal3.predict_proba(logits(raw3))[:,1]
        q6=cal6.predict_proba(logits(raw6))[:,1]
        scores=[matthews_corrcoef(y[tr],(w*q3[tr]+(1-w)*p3d[tr])>=THRESHOLD) for w in GRID]
        w=GRID[int(np.argmax(scores))]
        weights.append(w)
        o3[te]=w*q3[te]+(1-w)*p3d[te]
        o6[te]=q6[te]
    out={'seeds':list(SEEDS),'fusion_weights_1d_by_fold':weights,'threshold':THRESHOLD,
         'three_1d_plus_one_historical_3d':metrics(y,o3),'six_1d':metrics(y,o6),
         'caveat':'Development comparison; historical 3D and cross-fitted OOF meta-training are not proven end-to-end nested. '
                  'Do not claim six members in the fusion arm or independent blind validation.'}
    out['delta_six_minus_fusion']={k:out['six_1d'][k]-out['three_1d_plus_one_historical_3d'][k] for k in ('MCC','AUROC','AUPRC')}
    (RUN/'comparison.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    with (RUN/'comparison_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['source_index','sample_id','group_id','fold','label','fusion_3seq_1struct','sequence_6'])
        for i,r in enumerate(rows): w.writerow([i,ids[i],r['group_id'],fold[i],y[i],o3[i],o6[i]])
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
