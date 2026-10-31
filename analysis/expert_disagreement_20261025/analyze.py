"""Training-only expert complementarity, evidence strata, and matched feature probes."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

OUT=Path(__file__).resolve().parent
ATLAS=OUT.parent/'ghxtox_error_atlas_20261025'
COMPARE=OUT.parent/'dataset_comparison_20261025'
P=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
BIO=P['features'][:10]

def save(name,df):df.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.17g')
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def matching(df,case,control,contrast,strict=False):
    columns=['length','net_charge_proxy','mean_hydropathy','cys_count','aromatic_fraction']
    scales=np.array([3,2,.5,1,.1],dtype=float)
    if strict:
        scales[3]=0
        extra=['mean_plddt','low_plddt_fraction','max_otherfold_kmer_cosine','sequence_seed_std']
        columns+=extra;scales=np.r_[scales,[.05,.1,.1,.05]]
    used=set();pairs=[]
    for fold in range(5):
        cases=df[case&(df.fold==fold)];controls=df[control&(df.fold==fold)]
        eligible={}
        for i,c in cases.iterrows():
            delta=np.abs(controls[columns].to_numpy(dtype=float)-c[columns].to_numpy(dtype=float))
            ok=np.all(delta<=scales+1e-12,axis=1)&(controls.group_id.to_numpy()!=c.group_id)&(controls.evidence_tier.to_numpy()==c.evidence_tier)
            eligible[i]=[(int(j),float(np.sum((v/np.where(scales>0,scales,1))**2))) for j,v in zip(controls.index[ok],delta[ok])]
        for i in sorted(eligible,key=lambda i:(len(eligible[i]),int(df.loc[i,'source_index']))):
            cg=df.loc[i,'group_id']
            if cg in used:continue
            opts=[(j,dist) for j,dist in eligible[i] if df.loc[j,'group_id'] not in used]
            if not opts:continue
            j,dist=min(opts,key=lambda x:(x[1],int(df.loc[x[0],'source_index'])))
            used.update([cg,df.loc[j,'group_id']])
            pairs.append(dict(contrast=contrast,strict=strict,fold=fold,evidence_tier=df.loc[i,'evidence_tier'],
                case_index=i,control_index=j,case_id=df.loc[i,'sample_id'],control_id=df.loc[j,'sample_id'],
                case_group=cg,control_group=df.loc[j,'group_id'],distance=dist))
    return pairs

def paired_tests(df,pairs,strict=False):
    rng=np.random.default_rng(P['seed']+int(strict));result=[]
    for contrast,g in pairs.groupby('contrast'):
        for col in BIO if strict else P['features']:
            delta=df.loc[g.case_index,col].to_numpy()-df.loc[g.control_index,col].to_numpy()
            ok=np.isfinite(delta);diff=delta[ok];fold=g.fold.to_numpy()[ok]
            if len(diff)<5:continue
            boot=diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(1)
            null=(rng.choice([-1,1],size=(5000,len(diff)))*diff).mean(1)
            mean=float(diff.mean());p=(1+int((abs(null)>=abs(mean)).sum()))/5001
            row=dict(contrast=contrast,strict=strict,feature=col,n=len(diff),mean_case_minus_control=mean,
                CI95_low=float(np.quantile(boot,.025)),CI95_high=float(np.quantile(boot,.975)),p=p)
            means=[float(diff[fold==f].mean()) if np.any(fold==f) else None for f in range(5)]
            for f,v in enumerate(means):row[f'fold{f}_difference']=v
            row['available_folds']=sum(v is not None for v in means)
            row['folds_same_direction']=sum(np.sign(v)==np.sign(mean) for v in means if v is not None)
            d=diff[np.isin(fold,[0,1,2])];r=diff[np.isin(fold,[3,4])]
            row['discovery_difference']=float(d.mean()) if len(d) else None
            row['internal_check_difference']=float(r.mean()) if len(r) else None
            row['discovery_internal_check_same_direction']=bool(len(d) and len(r) and d.mean()*r.mean()>0)
            result.append(row)
    if result:
        q=multipletests([r['p'] for r in result],method='fdr_bh')[1]
        for r,v in zip(result,q):r['q_BH']=float(v)
    return pd.DataFrame(result) if result else pd.DataFrame(columns=['contrast','strict','feature','n','mean_case_minus_control','CI95_low','CI95_high','p','q_BH'])

def run():
    files=[ATLAS/'evidence_phase2/training_evidence_flags.csv',COMPARE/'same_sequence_label_review.csv',ATLAS/'protocol.json',
        ATLAS/'evidence_phase2/uniprot_reviewed_length11_50_20261025.tsv']
    df=pd.read_csv(files[0],float_precision='round_trip',keep_default_na=False)
    for col in P['features']+['length','net_charge_proxy','mean_hydropathy','cys_count','aromatic_fraction']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    assert len(df)==6387 and df.sample_id.is_unique
    old=json.loads(files[2].read_text(encoding='utf8'))
    assert np.array_equal(df.sequence_error.to_numpy(),(df.p_seq>=old['sequence_reference_threshold']).to_numpy()!=df.label.to_numpy())
    assert np.array_equal(df.structure_error.to_numpy(),(df.p_struct>=old['structure_reference_threshold']).to_numpy()!=df.label.to_numpy())
    assert np.array_equal(df.fusion_error.to_numpy(),(df.p_fusion>=old['fusion_threshold']).to_numpy()!=df.label.to_numpy())
    conflicts=pd.read_csv(files[1],keep_default_na=False)
    df['cross_source_label_disagreement']=df.sequence.isin(conflicts.loc[conflicts.scope=='broad_annotation','sequence'])
    df['evidence_review_flag']=df.any_fragment_match|df.cross_source_label_disagreement
    df['evidence_tier']=np.where(df.evidence_review_flag,'fragment_or_label_review',np.where(df.identity_status=='unresolved','identity_unresolved','recovered_no_selected_flag_not_gold'))
    # Actual identity_status includes not_recovered spelling; derive directly from retained accession count.
    df.loc[~df.evidence_review_flag&(df.exact_match_count==0),'evidence_tier']='identity_unresolved'
    df['expert_group']=np.select([~df.sequence_error&~df.structure_error,df.sequence_error&~df.structure_error,~df.sequence_error&df.structure_error],
        ['both_correct','structure_only_correct','sequence_only_correct'],default='both_wrong')
    df['fusion_change']=np.where(df.fusion_corrects_sequence,'corrected_sequence',np.where(df.fusion_harms_sequence,'harmed_sequence','unchanged'))
    df['probability_gap']=abs(df.p_seq-df.p_struct)
    df['common_half_disagreement']=(df.p_seq>=.5)!=(df.p_struct>=.5)
    df['cys_stratum']=np.where(df.cys_count<4,'Cys<4','Cys>=4')
    save('training_expert_atlas.csv',df)
    save('expert_disagreements.csv',df[df.expert_decision_disagreement])
    save('fusion_corrections.csv',df[df.fusion_corrects_sequence])
    save('fusion_harms.csv',df[df.fusion_harms_sequence])
    save('both_wrong.csv',df[df.expert_group=='both_wrong'])
    save('outcome_cells.csv',df.groupby(['label','expert_group','fusion_error'],dropna=False).size().reset_index(name='n'))
    blocks=[]
    for (tier,group),g in df.groupby(['evidence_tier','expert_group']):
        blocks.append(dict(evidence_tier=tier,expert_group=group,n=len(g),positive=int(g.label.sum()),
            fusion_errors=int(g.fusion_error.sum()),fusion_corrects=int(g.fusion_corrects_sequence.sum()),fusion_harms=int(g.fusion_harms_sequence.sum()),
            median_plddt=float(g.mean_plddt.median()),median_seed_std=float(g.sequence_seed_std.median())))
    save('expert_groups_by_evidence.csv',pd.DataFrame(blocks))
    save('cys_expert_strata.csv',df.groupby(['evidence_tier','label','cys_stratum','expert_group']).agg(n=('sample_id','size'),fusion_errors=('fusion_error','sum')).reset_index())
    masks=[]
    for label in [0,1]:masks.append((f'structure_minus_sequence_label{label}',(df.expert_group=='structure_only_correct')&(df.label==label),(df.expert_group=='sequence_only_correct')&(df.label==label)))
    recovered=df.evidence_tier=='recovered_no_selected_flag_not_gold'
    for err,correct in [('FP','TN'),('FN','TP')]:masks.append((f'fusion_{err}_minus_{correct}_no_selected_flag',recovered&(df.fusion_category==err),recovered&(df.fusion_category==correct)))
    coverage=[];pair_tables={};test_tables={}
    for strict in [False,True]:
        rows=[]
        for name,case,control in masks:
            pairs=matching(df,case,control,name,strict)
            rows.extend(pairs)
            for fold in range(5):coverage.append(dict(contrast=name,strict=strict,fold=fold,cases=int((case&(df.fold==fold)).sum()),controls=int((control&(df.fold==fold)).sum()),pairs=sum(x['fold']==fold for x in pairs)))
        pairs=pd.DataFrame(rows);pair_tables[strict]=pairs
        for contrast,g in pairs.groupby('contrast'):
            ids=list(g.case_group)+list(g.control_group);assert len(ids)==len(set(ids))
            assert (df.loc[g.case_index,'label'].to_numpy()==df.loc[g.control_index,'label'].to_numpy()).all()
            assert (df.loc[g.case_index,'evidence_tier'].to_numpy()==df.loc[g.control_index,'evidence_tier'].to_numpy()).all()
        tests=paired_tests(df,pairs,strict);test_tables[strict]=tests
        prefix='strict' if strict else 'primary';save(f'{prefix}_matched_pairs.csv',pairs);save(f'{prefix}_matched_tests.csv',tests.sort_values('q_BH') if len(tests) else tests)
        balance=[]
        for contrast,g in pairs.groupby('contrast'):
            for col in ['length','net_charge_proxy','mean_hydropathy','cys_count','aromatic_fraction','mean_plddt','low_plddt_fraction','sequence_seed_std','max_otherfold_kmer_cosine']:
                vals=df.loc[g.case_index,col].to_numpy()-df.loc[g.control_index,col].to_numpy()
                balance.append(dict(contrast=contrast,strict=strict,feature=col,n=len(vals),mean_difference=float(vals.mean()),max_abs_difference=float(abs(vals).max())))
        save(f'{prefix}_matching_balance.csv',pd.DataFrame(balance))
    save('matching_coverage.csv',pd.DataFrame(coverage))
    # Illustrative pairs chosen by index, independent of tested feature effects/p values.
    queue=[]
    for name,g in pair_tables[False].groupby('contrast'):
        for _,pair in g.sort_values(['fold','case_id']).head(2).iterrows():
            for role in ['case','control']:
                row=df.loc[pair[f'{role}_index']].to_dict();row.update(review_reason=name,pair_role=role,paired_id=pair['control_id' if role=='case' else 'case_id']);queue.append(row)
    # Previously verified biological leads are explicitly carried over, not fresh statistical discoveries.
    for sid in ['peptide_424','peptide_322','peptide_1392','peptide_309','peptide_332','peptide_3648','peptide_193','peptide_251']:
        row=df[df.sample_id==sid].iloc[0].to_dict();row.update(review_reason='previous_evidence_sentinel',pair_role='sentinel',paired_id='');queue.append(row)
    q=pd.DataFrame(queue).drop_duplicates('sample_id');save('mechanism_review_queue.csv',q)
    manifest=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    dump('input_manifest.json',manifest)
    summary=dict(n=len(df),expert_disagreements=int(df.expert_decision_disagreement.sum()),expert_groups=df.expert_group.value_counts().to_dict(),
        corrections=int(df.fusion_corrects_sequence.sum()),harms=int(df.fusion_harms_sequence.sum()),
        evidence_tiers=df.evidence_tier.value_counts().to_dict(),common_half_disagreements=int(df.common_half_disagreement.sum()),
        corrections_by_evidence=df[df.fusion_corrects_sequence].evidence_tier.value_counts().to_dict(),harms_by_evidence=df[df.fusion_harms_sequence].evidence_tier.value_counts().to_dict(),
        primary_pairs=pair_tables[False].contrast.value_counts().to_dict(),strict_pairs=pair_tables[True].contrast.value_counts().to_dict(),
        primary_q_lt05=test_tables[False][test_tables[False].q_BH<.05].to_dict('records'),
        strict_q_lt05=test_tables[True][test_tables[True].q_BH<.05].to_dict('records') if len(test_tables[True]) else [],review_queue=len(q),
        external_test_predictions_used=False,model_training=False,archived_oof_not_certified_nested=True)
    dump('summary.json',summary)
    print(json.dumps(summary,ensure_ascii=True),flush=True)

if __name__=='__main__':run()
