"""Paired retrospective comparison; reuse old predictions, no tuning."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score
O=Path(__file__).resolve().parent;R=O.parents[1]
H=O/'handoff/GHXTox_team_handoff_20261027'
def readcsv(p):
    return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(n,d):(O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def metric(y,p,pred):
    return {'MCC':float(matthews_corrcoef(y,pred)), 'AUPRC':float(average_precision_score(y,p)),
            'AUROC':float(roc_auc_score(y,p)), 'FN':int(((y==1)&(pred==0)).sum()),
            'FP':int(((y==0)&(pred==1)).sum()), 'SN':float(pred[y==1].mean()),
            'SP':float(1-pred[y==0].mean())}
def mcc_counts(y,p):
    tp=np.sum((y==1)&(p==1),axis=-1);tn=np.sum((y==0)&(p==0),axis=-1)
    fp=np.sum((y==0)&(p==1),axis=-1);fn=np.sum((y==1)&(p==0),axis=-1)
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den,dtype=float),where=den>0)
summary={}
for dataset in ['test1','test2']:
    old=readcsv(R/f'runs/default_multidimensional_fusion_single_structure/{dataset}_predictions.csv')
    new=readcsv(O/f'new_tests/{dataset}_predictions.csv')
    ids=[x['sample_id'] for x in old];newmap={x['sample_id']:x for x in new}
    assert len(ids)==len(set(ids))==len(newmap)==len(new)
    assert set(ids)==set(newmap)
    new=[newmap[i] for i in ids]
    y=np.array([int(x['label']) for x in old]);yn=np.array([int(x['label']) for x in new]);assert np.array_equal(y,yn)
    p=np.array([float(x['toxicity_probability']) for x in old]);q=np.array([float(x['toxicity_probability']) for x in new])
    po=np.array([int(x['prediction']) for x in old]);pn=np.array([int(x['prediction']) for x in new])
    assert np.array_equal(po,(p>=.47022256531284556).astype(int))
    assert np.array_equal(pn,(q>=.5063534554819765).astype(int))
    seqdelta=np.abs(np.array([float(x['p_1d']) for x in old])-np.array([float(x['p_1d']) for x in new]))
    mo,mn=metric(y,p,po),metric(y,q,pn)
    rng=np.random.default_rng(20261027);ip=np.where(y==1)[0];ineg=np.where(y==0)[0];diff=[];apdiff=[]
    for b in range(5000):
        ix=np.r_[rng.choice(ip,len(ip),replace=True),rng.choice(ineg,len(ineg),replace=True)]
        diff.append(float(mcc_counts(y[ix],pn[ix])-mcc_counts(y[ix],po[ix])))
        apdiff.append(float(average_precision_score(y[ix],q[ix])-average_precision_score(y[ix],p[ix])))
    cases=[]
    for j,(a,b) in enumerate(zip(old,new)):
        role='corrected' if po[j]!=y[j] and pn[j]==y[j] else 'harmed' if po[j]==y[j] and pn[j]!=y[j] else 'both_wrong' if pn[j]!=y[j] else 'both_correct'
        cases.append({'sample_id':ids[j],'sequence':b['sequence'],'label':int(y[j]),'role':role,'old_p':p[j].item(),'new_p':q[j].item(),'old_prediction':int(po[j]),'new_prediction':int(pn[j])})
    dump(f'{dataset}_paired_cases.json',cases)
    summary[dataset]={'n':len(y),'old':mo,'new':mn,'corrected':sum(x['role']=='corrected' for x in cases),'harmed':sum(x['role']=='harmed' for x in cases),
                      'delta_MCC':mn['MCC']-mo['MCC'],'delta_AUPRC':mn['AUPRC']-mo['AUPRC'],
                      'delta_MCC_95CI':np.quantile(diff,[.025,.975]).tolist(),'delta_AUPRC_95CI':np.quantile(apdiff,[.025,.975]).tolist(),
                      'sequence_probability_max_abs_change':float(seqdelta.max()),
                      'bootstrap':'5000 paired label-stratified sample draws, seed20261027; retrospective, no retraining uncertainty'}
    common_p=np.array([.72*float(b['p_1d'])+.28*float(a['p_3d']) for a,b in zip(old,new)])
    summary[dataset]['old_structure_with_new_sequence_probability_control']=metric(y,common_p,(common_p>=.47022256531284556).astype(int))

# Verify new OOF alignment and use each row's own already frozen threshold.
manifest=readcsv(H/'data/folds/train_cdhit080_fallback_folds.csv');byid={x['sample_id']:x for x in manifest}
features=np.load(R/'analysis/saprot_and_robust_classifier_pilot_20261026/frozen_features.npz')
sid=features['sample_ids'].tolist();si={k:i for i,k in enumerate(sid)}
oofs={}
for name in ['no_inner_transformer','S3_plus_no_inner_transformer']:
    rows=readcsv(H/f'runs/structure_candidates_20261022/report/{name}_oof.csv')
    assert len(rows)==6387 and len({x['sample_id'] for x in rows})==6387
    for x in rows:
        m=byid[x['sample_id']];j=si[x['sample_id']]
        assert x['sequence']==m['sequence']==str(features['sequences'][j])
        assert int(x['label'])==int(m['label'])==int(features['labels'][j])
        assert int(x['outer_fold'])==int(m['fold'])==int(features['folds'][j])
        assert m['group_id']==str(features['groups'][j])
    yy=np.array([int(x['label']) for x in rows]);pp=np.array([float(x['probability']) for x in rows]);tt=np.array([float(x['threshold']) for x in rows])
    oofs[name]=metric(yy,pp,(pp>=tt).astype(int))
oofs['matched_protocol_reference_rows']=readcsv(H/'runs/structure_candidates_20261022/report/complete_oof_metrics.csv')
oofs['same_sample_sequence_label_fold_group_as_existing_saprot']=True
oofs['scope']='Existing development OOF; aligned outer folds do not establish identical inner training/calibration roles.'
dump('oof_comparison.json',oofs)
summary['decision']='Keep old local production default: new candidate does not meet user condition of higher test classification metrics. Preserve new candidate and matched OOF evidence; no test tuning.'
summary['new_checkpoint_and_oof_available']=True
summary['saprot_fusion_executed']=False
dump('comparison.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
