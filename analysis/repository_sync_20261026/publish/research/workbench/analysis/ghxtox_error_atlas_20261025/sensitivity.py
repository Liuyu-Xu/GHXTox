"""Secondary, explicitly post-discovery sensitivity checks; no predictive model fit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_atlas import OUT, P, TEST_FEATURES, save, dump

def cys_audit(d):
    # Full-training descriptive risks; cluster bootstrap retains within-group dependence.
    group_codes,group_names=pd.factorize(d.group_id)
    accum=np.zeros((len(group_names),8))
    rows=[]
    for label in [0,1]:
        for high in [0,1]:
            mask=(d.label.to_numpy()==label)&((d.cys_count.to_numpy()>=4)==bool(high))
            k=label*4+high*2
            np.add.at(accum[:,k],group_codes[mask],1)
            np.add.at(accum[:,k+1],group_codes[mask],d.fusion_error.to_numpy()[mask].astype(int))
    rng=np.random.default_rng(P['seed']+1)
    weights=rng.multinomial(len(group_names),np.repeat(1/len(group_names),len(group_names)),size=2000)
    sampled=weights@accum
    for label in [0,1]:
        for high in [0,1]:
            k=label*4+high*2; total=accum[:,k].sum();err=accum[:,k+1].sum()
            rate=sampled[:,k+1]/sampled[:,k]
            rows.append(dict(label=label,cys_ge4=high,n=int(total),errors=int(err),error_rate=err/total,
                group_bootstrap_ci_low=float(np.quantile(rate,.025)),group_bootstrap_ci_high=float(np.quantile(rate,.975))))
    save('cys_stratum_risks.csv',pd.DataFrame(rows))
    folds=[]
    for (f,y),g in d.groupby(['fold','label']):
        for high in [False,True]:
            x=g[(g.cys_count>=4)==high]
            folds.append(dict(fold=f,label=y,cys_ge4=high,n=len(x),errors=int(x.fusion_error.sum()),error_rate=float(x.fusion_error.mean())))
    save('cys_stratum_by_fold.csv',pd.DataFrame(folds))

def rematch(d):
    # Chosen after primary results; it is a robustness diagnostic, not a new preregistered test.
    cols=['length','net_charge_proxy','mean_hydropathy','aromatic_fraction',
          'mean_plddt','low_plddt_fraction','max_otherfold_kmer_cosine','sequence_seed_std']
    cal=np.array([3,2,.5,.1,.1,.15,.15,.15])
    pairs=[]
    for err,correct in [('FP','TN'),('FN','TP')]:
        used=set()
        for fold in range(5):
            cases=d[(d.fusion_category==err)&(d.fold==fold)]
            controls=d[(d.fusion_category==correct)&(d.fold==fold)]
            pool={}
            for i,c in cases.iterrows():
                delta=np.abs(controls[cols].to_numpy()-c[cols].to_numpy(dtype=float))
                valid=np.all(delta<=cal+1e-12,axis=1)&(controls.cys_count.to_numpy()==c.cys_count)&(controls.group_id.to_numpy()!=c.group_id)
                pool[i]=[(int(j),float(np.sum((dv/cal)**2))) for j,dv in zip(controls.index[valid],delta[valid])]
            for i in sorted(pool,key=lambda k:(len(pool[k]),int(d.source_index.iloc[k]))):
                if d.group_id.iloc[i] in used:continue
                opt=[(j,dist) for j,dist in pool[i] if d.group_id.iloc[j] not in used]
                if not opt:continue
                j,dist=min(opt,key=lambda x:(x[1],x[0]));used.update([d.group_id.iloc[i],d.group_id.iloc[j]])
                pairs.append(dict(error_type=err,fold=fold,case_index=i,control_index=j,
                    case_id=d.sample_id.iloc[i],control_id=d.sample_id.iloc[j],distance=dist))
    pairs=pd.DataFrame(pairs);save('sensitivity_matched_pairs.csv',pairs)
    tests=[];rng=np.random.default_rng(P['seed']+2)
    for err,g in pairs.groupby('error_type'):
        ci=g.case_index.to_numpy();ti=g.control_index.to_numpy()
        for col in TEST_FEATURES[:10]:
            vals=d[col].to_numpy(float);diff=vals[ci]-vals[ti];valid=np.isfinite(diff);diff=diff[valid];fold=g.fold.to_numpy()[valid]
            if len(diff)<5:continue
            boot=diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(1)
            perm=(diff[None,:]*rng.choice([-1,1],(5000,len(diff)))).mean(1)
            test=dict(error_type=err,feature=col,n=len(diff),mean_difference=float(diff.mean()),
                ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                p=(1+int((abs(perm)>=abs(diff.mean())).sum()))/5001,
                positive_folds=int(sum(np.any(fold==f) and diff[fold==f].mean()>0 for f in range(5))),
                negative_folds=int(sum(np.any(fold==f) and diff[fold==f].mean()<0 for f in range(5))))
            tests.append(test)
    tests=pd.DataFrame(tests);o=np.argsort(tests.p);pv=tests.p.to_numpy()[o]
    q=np.minimum.accumulate((pv*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1]
    tests['q_bh_secondary_family']=0.;tests.loc[o,'q_bh_secondary_family']=np.minimum(q,1)
    save('sensitivity_feature_tests.csv',tests.sort_values('q_bh_secondary_family'))
    coverage=[]
    for err in ['FP','FN']:
        coverage.append(dict(error_type=err,total_errors=int((d.fusion_category==err).sum()),matched=int((pairs.error_type==err).sum())))
    dump('sensitivity_summary.json',dict(status='post-discovery secondary diagnostics',calipers=dict(zip(cols,cal)),exact_cys_count=True,
        coverage=coverage,tests=tests.to_dict('records'),causal_or_independent_claim=False))

def review_queue(d):
    # Select representative errors for evidence recovery, not automatic relabeling.
    selected=[]
    for cat in ['FP','FN']:
        for high in [False,True]:
            g=d[(d.fusion_category==cat)&((d.cys_count>=4)==high)].sort_values(['all_seed_reference_wrong','fusion_margin'],ascending=[False,False]).head(5)
            for _,r in g.iterrows():
                selected.append(dict(sample_id=r.sample_id,sequence=r.sequence,label=r.label,error_type=cat,
                    fold=r.fold,cys_count=r.cys_count,p_seq=r.p_seq,p_struct=r.p_struct,p_fusion=r.p_fusion,
                    mean_plddt=r.mean_plddt,all_seed_reference_wrong=r.all_seed_reference_wrong,
                    endpoint_evidence='unknown',original_accession='unknown',original_paper='unknown',
                    assay_conditions='unknown',terminal_modifications='unknown',review_status='pending_original_evidence'))
    save('evidence_review_queue.csv',pd.DataFrame(selected))

if __name__=='__main__':
    d=pd.read_csv(OUT/'all_training_atlas.csv',float_precision='round_trip')
    cys_audit(d);rematch(d);review_queue(d)
    print((OUT/'cys_stratum_risks.csv').read_text(encoding='utf-8-sig'))
    print((OUT/'sensitivity_summary.json').read_text()[:500])
