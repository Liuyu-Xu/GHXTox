"""Replay inference and inner selection independently from saved scalar models."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';rd=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
z=np.load(O/'evaluation_predictions.npz');ids=z['sample_ids'].tolist();idx={k:j for j,k in enumerate(ids)};N=len(ids);y=z['labels'];fold=z['folds'];groups=z['groups'];half=z['halves'];extra=z['features_extra']
old={x['sample_id']:x for x in rd(A/'error_evidence_linkage_20261025/all_training_linked.json')}
h=A/'upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027'
new={x['sample_id']:x for x in csv.DictReader((h/'runs/structure_candidates_20261022/report/S3_plus_no_inner_transformer_oof.csv').open(encoding='utf-8-sig'))}
s=np.load(A/'saprot_standalone_20261026/ensemble_scores.npz');assert s['sample_ids'].tolist()==ids
gx={'old':np.array([logit(np.clip(float(old[k]['p_fusion']),1e-12,1-1e-12))-logit(.47022256531284556) for k in ids]),
    'new':np.array([logit(np.clip(float(new[k]['probability']),1e-12,1-1e-12))-logit(float(new[k]['threshold'])) for k in ids])}
streams={};replayed=0;maxdiff=0.
for m in rd(O/'scalar_models.json'):
    pair,typ=m['arm'].rsplit('_',1);v,mode=pair.split('_');x=gx[v] if typ=='G' else s['SaProt_only_'+mode]
    tr=np.array([idx[k] for k in m['fit_ids']]);te=np.array([idx[k] for k in m['predict_ids']]);assert not set(tr)&set(te) and not set(groups[tr])&set(groups[te])
    assert np.all(fold[tr]==m['outer_fold']) and np.all(fold[te]==m['outer_fold'])
    if m['inner_eval_fold']==-1:
        assert np.all(half[tr]!=m['evaluation_half']) and np.all(half[te]==m['evaluation_half'])
    else:assert np.all(half[tr]!=m['evaluation_half']) and np.all(half[te]!=m['evaluation_half'])
    p=expit(((x[te]-m['mean'][0])/m['scale'][0])*m['coefficient'][0][0]+m['intercept'][0])
    assert abs(np.mean(x[tr])-m['mean'][0])<1e-10
    key=(pair,m['outer_fold'],m['evaluation_half'],typ,'evaluation' if m['inner_eval_fold']==-1 else 'development')
    if key not in streams:streams[key]=np.full(N,np.nan)
    assert np.isnan(streams[key][te]).all();streams[key][te]=p;replayed+=len(te)
    if m['inner_eval_fold']==-1:
        stored=z[pair+('_G_probability' if typ=='G' else '_S_probability')][te]
        diff=float(np.max(np.abs(p-stored)));assert diff<1e-12;maxdiff=max(maxdiff,diff)

def brute_best(p,yy):
    ts=np.r_[np.unique(p),.5,np.nextafter(p.max(),np.inf)];pred=p[None,:]>=ts[:,None]
    tp=(pred&(yy[None,:]==1)).sum(axis=1);fp=(pred&(yy[None,:]==0)).sum(axis=1);fn=(yy==1).sum()-tp;tn=(yy==0).sum()-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));mc=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den),where=den>0)
    candidates=np.where(mc>=mc.max()-1e-12)[0];best=candidates[np.argmin(np.abs(ts[candidates]-.5))]
    return float(ts[best]),float(mc[best])

selmap={};threshold_checks=0
for d in rd(O/'selection_records.json'):
    pair,f,h=d['pair'],d['outer_fold'],d['evaluation_half'];dev=np.array([idx[k] for k in d['development_ids']]);ev=np.array([idx[k] for k in d['evaluation_ids']])
    assert not set(groups[dev])&set(groups[ev]);assert np.all(half[dev]!=h) and np.all(half[ev]==h)
    pg=streams[pair,f,h,'G','development'][dev];ps=streams[pair,f,h,'S','development'][dev]
    tg,_=brute_best(pg,y[dev]);ts,_=brute_best(ps,y[dev]);assert tg==d['G_threshold'] and ts==d['S_threshold']
    for c in d['all_candidates']:
        p=(1-c['weight_SaProt'])*pg+c['weight_SaProt']*ps;t,mc=brute_best(p,y[dev]);assert abs(t-c['threshold'])<1e-12 and abs(mc-c['development_inner_MCC'])<1e-12;threshold_checks+=1
    best=max(c['development_inner_MCC'] for c in d['all_candidates']);choice=next(c for c in d['all_candidates'] if c['development_inner_MCC']>=best-1e-12);assert choice==d['selected']
    pg=z[pair+'_G_probability'][ev];ps=z[pair+'_S_probability'][ev];w=choice['weight_SaProt'];t=choice['threshold'];pp=(1-w)*pg+w*ps
    assert np.array_equal(z[pair+'_baseline_pred'][ev],pg>=tg);assert np.array_equal(z[pair+'_weight_selected_pred'][ev],pp>=t)
    selmap[pair,f,h]=d

gate_count=0
for d in rd(O/'gate_models.json'):
    pair,f,h=d['pair'],d['outer_fold'],d['evaluation_half'];selection=selmap[pair,f,h]
    fit=np.array([idx[k] for k in d['fit_ids']]);ev=np.array([idx[k] for k in selection['evaluation_ids']]);assert not set(groups[fit])&set(groups[ev])
    tg,ts=selection['G_threshold'],selection['S_threshold'];pg=streams[pair,f,h,'G','development'];ps=streams[pair,f,h,'S','development']
    dev=(fold==f)&(half!=h);expected=dev&((pg>=tg)!=(ps>=ts));assert set(fit)==set(np.where(expected)[0])
    assert np.array_equal(np.array(d['target']),(ps[fit]>=ts)==y[fit])
    g=z[pair+'_G_probability'][ev];s1=z[pair+'_S_probability'][ev]
    X=np.column_stack([g,s1,g-tg,s1-ts,np.abs(g-s1)])
    if d['kind']=='feature_gate':X=np.column_stack([X,extra[ev]])
    X=np.where(np.isfinite(X),X,np.array(d['median']))
    q=np.full(len(ev),d['constant']) if d['fallback'] else expit(((X-np.array(d['mean']))/np.array(d['scale']))@np.array(d['coefficient'][0])+d['intercept'][0])
    diff=float(np.max(np.abs(q-z[pair+'_'+d['kind']+'_q'][ev])));assert diff<1e-12;maxdiff=max(maxdiff,diff)
    choose=((g>=tg)!=(s1>=ts))&(q>.5)
    assert np.array_equal(choose,z[pair+'_'+d['kind']+'_choose_S'][ev])
    pred=np.where(choose,s1>=ts,g>=tg);assert np.array_equal(pred,z[pair+'_'+d['kind']+'_pred'][ev]);gate_count+=len(ev)

atlas=rd(O/'source_fourway_atlas.json');atlas_checks=0
for pair,d in atlas.items():
    v,mode=pair.split('_');gd=gx[v]>=0;sd=s['SaProt_only_'+mode]>=0;good=gd==y;sgood=sd==y
    cases={'G_wrong_S_right':(~good)&sgood,'G_right_S_wrong':good&(~sgood),'both_wrong':(~good)&(~sgood),'both_right':good&sgood}
    assert sum(x.sum() for x in cases.values())==N
    for name,mask in cases.items():assert mask.sum()==d['classes'][name]['n'] and set(np.array(ids)[mask])==set(d['classes'][name]['sample_ids'])
    atlas_checks+=N
for k,m in rd(O/'metrics.json').items():
    p=z[k+'_score'];dd=z[k+'_pred'];assert abs(m['MCC']-matthews_corrcoef(y,dd))<1e-12 and abs(m['AUPRC']-average_precision_score(y,p))<1e-12 and abs(m['AUROC']-roc_auc_score(y,p))<1e-12
bt=np.load(O/'bootstrap_deltas.npz')
for j,c in enumerate(rd(O/'comparisons.json')):assert np.max(np.abs(np.quantile(bt[f'contrast{j}'],[.00125,.99875])-c['delta_MCC_99_75CI']))<1e-12
for p,h in rd(O/'input_manifest.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
out={'passed':True,'scalar_predictions_replayed':replayed,'gate_predictions_replayed':gate_count,'max_probability_difference':maxdiff,'all_40_weight_selection_records_replayed':True,'weight_threshold_candidates_bruteforce_checked':threshold_checks,'source_fourway_checks':atlas_checks,'all_metrics_and_saved_interval_quantiles_verified':True,'input_hashes_unchanged':True,'neural_training':False,'test_data_used':False}
(O/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
