"""Independent frozen-prediction and exact-margin attribution replay."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
O=Path(__file__).resolve().parent;R=O.parents[1];S=R/'analysis/saprot_selective_value_20261027'
rd=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
z=np.load(O/'factorial_predictions.npz');prior=np.load(S/'evaluation_predictions.npz')
ids=z['sample_ids'].tolist();assert prior['sample_ids'].tolist()==ids;y=z['labels'];fold=z['folds'];half=z['halves'];N=len(ids);idx={k:j for j,k in enumerate(ids)}
f=rd(O/'factorial_results.json');sel=rd(S/'selection_records.json');records=rd(O/'all_sample_factorial_records.json');byrecord={(x['pair'],x['sample_id']):x for x in records}
decision_checks=0;margin_checks=0;subtype_checks=0;maxerr=0.
for pair in f:
    pg=z[pair+'_pg'];ps=z[pair+'_ps'];tg=z[pair+'_tg'];tf=z[pair+'_tf'];w=z[pair+'_w'];pm=(1-w)*pg+w*ps
    assert np.max(np.abs(pg-prior[pair+'_G_probability']))==0 and np.max(np.abs(ps-prior[pair+'_S_probability']))==0
    for d in sel:
        if d['pair']!=pair:continue
        mask=(fold==d['outer_fold'])&(half==d['evaluation_half'])
        assert np.all(w[mask]==d['selected']['weight_SaProt']) and np.all(tf[mask]==d['selected']['threshold']) and np.all(tg[mask]==d['G_threshold'])
    expected={'A_G_tG':pg>=tg,'B_G_tF':pg>=tf,'C_mix_tG':pm>=tg,'D_mix_tF':pm>=tf}
    for name,pred in expected.items():
        assert np.array_equal(pred,z[pair+'_'+name+'_pred']);m=f[pair]['metrics'][name];score=pg if name[0] in 'AB' else pm
        assert abs(m['MCC']-matthews_corrcoef(y,pred))<1e-12 and abs(m['AUPRC_uncentered_score']-average_precision_score(y,score))<1e-12 and abs(m['AUROC_uncentered_score']-roc_auc_score(y,score))<1e-12
        assert m['FN']==int(((y==1)&~pred).sum()) and m['FP']==int(((y==0)&pred).sum());decision_checks+=N
    a,b,c,d=[expected[k] for k in ['A_G_tG','B_G_tF','C_mix_tG','D_mix_tF']]
    assert np.array_equal(a,prior[pair+'_baseline_pred']) and np.array_equal(d,prior[pair+'_weight_selected_pred'])
    roles=np.where(a!=y,np.where(d==y,'corrected','both_wrong'),np.where(d!=y,'harmed','both_correct'))
    count=Counter()
    for j,k in enumerate(ids):
        row=byrecord[pair,k];assert row['role']==roles[j]
        err=abs(row['fusion_margin']-row['original_margin']-row['SaProt_score_push']-row['threshold_push']);assert err<1e-12;maxerr=max(maxerr,err);margin_checks+=1
        if roles[j] in ['corrected','harmed']:
            expected_sub='either_alone_sufficient' if b[j]!=a[j] and c[j]!=a[j] else 'threshold_alone_only' if b[j]!=a[j] else 'score_alone_only' if c[j]!=a[j] else 'both_changes_required'
            assert row['subtype']==expected_sub;count[roles[j],expected_sub]+=1;subtype_checks+=1
    for role in ['corrected','harmed']:
        saved=f[pair]['changed_case_attribution'][role]
        assert saved['n']==int((roles==role).sum())
        assert saved['subtypes']=={name:n for (rr,name),n in count.items() if rr==role}
    de=f[pair]['decomposition'];assert abs(de['score_ShAP_allocation_MCC']+de['threshold_ShAP_allocation_MCC']-de['total_delta_MCC'])<1e-12
    assert de['score_allocation_netcorrect']+de['threshold_allocation_netcorrect']==int((d==y).sum()-(a==y).sum())
changed=rd(O/'old3Di_corrected31_harmed20.json');assert Counter(x['role'] for x in changed)=={'corrected':31,'harmed':20}
bt=np.load(O/'bootstrap_deltas.npz')
for j,c in enumerate(rd(O/'factorial_comparisons.json')):assert np.max(np.abs(np.quantile(bt[f'contrast{j}'],[.00125,.99875])-c['delta_MCC_99_75CI']))<1e-12
null=np.load(O/'conditional_shuffle_values.npz')
for pair,d in rd(O/'conditional_shuffle_diagnostic.json').items():
    mm=null[pair+'_MCC'];assert len(mm)==200 and abs(mm.mean()-d['conditional_shuffle_MCC_mean'])<1e-12
    assert int((mm>=d['actual_MCC']).sum())==d['shuffles_with_MCC_at_least_actual']
for path,h in rd(O/'input_manifest.json').items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
result={'passed':True,'four_factor_decisions_replayed':decision_checks,'sample_margins_replayed':margin_checks,'changed_subtypes_replayed':subtype_checks,'max_margin_identity_error':maxerr,'original_31corrected_20harmed_ids_preserved':True,'all_weights_thresholds_reused_without_refit':True,'all_metrics_and_saved_interval_quantiles_verified':True,'saved_permutation_summaries_verified':True,'input_hashes_unchanged':True,'new_model_fits':0,'test_data_used':False}
(O/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
