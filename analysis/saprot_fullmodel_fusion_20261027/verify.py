"""Independent replay of saved scalar calibrators and fixed mixtures."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
z=np.load(O/'crossfit_predictions.npz');ids=z['sample_ids'].tolist();y=z['labels'];fold=z['folds'];half=z['halves'];groups=z['groups'];n=len(y)
old={x['sample_id']:x for x in read(A/'error_evidence_linkage_20261025/all_training_linked.json')}
h=A/'upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027'
new={x['sample_id']:x for x in csv.DictReader((h/'runs/structure_candidates_20261022/report/S3_plus_no_inner_transformer_oof.csv').open(encoding='utf-8-sig'))}
s=np.load(A/'saprot_standalone_20261026/ensemble_scores.npz');assert s['sample_ids'].tolist()==ids
x={'old':np.array([logit(float(old[k]['p_fusion']))-logit(.47022256531284556) for k in ids]),
   'new':np.array([logit(float(new[k]['probability']))-logit(float(new[k]['threshold'])) for k in ids]),
   'teacher_masked':s['SaProt_only_masked'],'teacher_3Di':s['SaProt_only_3Di']}
predrows=read(A/'saprot_standalone_20261026/oof_predictions.json')
for mode in ['masked','3Di']:
    for seed in [20261026,20261027]:
        m={a['sample_id']:a['logit'] for a in predrows if a['arm']=='SaProt_only_'+mode and a['seed']==seed}
        x[f'teacher_{mode}_seed{seed}']=np.array([m[k] for k in ids])
maxdiff=0.;count=0
for m in read(O/'calibration_models.json'):
    te=(fold==m['fold'])&(half==m['evaluation_half']);tr=(fold==m['fold'])&~te
    assert not set(groups[te])&set(groups[tr])
    v=(x[m['arm']][te]-m['mean'][0])/m['scale'][0]
    p=expit(v*m['coefficient'][0][0]+m['intercept'][0])
    diff=float(np.max(np.abs(p-z[m['arm']+'_calibrated_component'][te])))
    assert diff<1e-12;maxdiff=max(maxdiff,diff);count+=te.sum()
metrics=read(O/'metrics.json');comparisons=read(O/'comparisons.json');b=np.load(O/'bootstrap_deltas.npz')
for arm,m in metrics.items():
    p=z[arm+'_prob'];pred=z[arm+'_pred']
    assert abs(m['MCC']-matthews_corrcoef(y,pred))<1e-12
    assert abs(m['AUPRC']-average_precision_score(y,p))<1e-12
    assert abs(m['AUROC']-roc_auc_score(y,p))<1e-12
    assert m['FN']==int(((y==1)&(pred==0)).sum()) and m['FP']==int(((y==0)&(pred==1)).sum())
for c in comparisons:
    a,base=c['candidate'],c['baseline'];w=c['weight_SaProt']
    pp=(1-w)*z[c['version']+'_calibrated_component']+w*z['teacher_'+c['mode']+'_calibrated_component']
    assert np.max(np.abs(pp-z[a+'_prob']))<1e-12
    assert np.array_equal(z[a+'_pred'],(pp>=.5).astype(int))
    resc=(z[base+'_pred']!=y)&(z[a+'_pred']==y);harm=(z[base+'_pred']==y)&(z[a+'_pred']!=y)
    assert resc.sum()==c['corrected'] and harm.sum()==c['harmed']
    assert np.max(np.abs(np.quantile(b[a],[.003125,.996875])-c['delta_MCC_99_375CI']))<1e-12
rows=read(O/'all_sample_predictions.json');assert len(rows)==n
for j,row in enumerate(rows):
    assert row['sample_id']==ids[j]
    for c in comparisons:
        a,base=c['candidate'],c['baseline'];bp=z[base+'_pred'][j];cp=z[a+'_pred'][j]
        role='corrected' if bp!=y[j] and cp==y[j] else 'harmed' if bp==y[j] and cp!=y[j] else 'both_wrong' if cp!=y[j] else 'both_correct'
        assert row[a+'_role']==role
manifest=read(O/'input_manifest.json');assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in manifest.items())
out={'passed':True,'scalar_calibrator_predictions_replayed':int(count),'max_probability_difference':maxdiff,
     'eight_fixed_mixtures_replayed':True,'all_sample_role_checks':n*len(comparisons),'all_metrics_recomputed':True,
     'saved_bootstrap_interval_quantiles_verified':True,'input_hashes_unchanged':True,'new_neural_training':False}
(O/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
