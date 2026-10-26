"""Evidence review flags versus frozen historical test predictions; no training.
Operational uncertainty flags are fixed before joining model outcomes.
Unflagged never means experimentally certified unambiguous.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import Table2x2,StratifiedTable
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import matthews_corrcoef

OUT=Path(__file__).resolve().parent
ANALYSIS=OUT.parent
ROOT=Path('D:/dachuang2026/GHXTox')
EVIDENCE=ANALYSIS/'ghxtox_error_atlas_20261025/evidence_phase2'
COMPARE=ANALYSIS/'dataset_comparison_20261025'

def save(name,df):df.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.17g')
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def counts(df,col):
    return np.array([[int((df[col]&df.error).sum()),int((df[col]&~df.error).sum())],
        [int((~df[col]&df.error).sum()),int((~df[col]&~df.error).sum())]])

def stats(df,col,split):
    t=counts(df,col);a,b,c,d=map(int,t.ravel());n=len(df)
    result=dict(split=split,definition=col,n=n,errors=a+c,correct=b+d,
        flagged_error=a,flagged_correct=b,unflagged_error=c,unflagged_correct=d,
        flagged_n=a+b,unflagged_n=c+d,flagged_fraction=(a+b)/n,
        flagged_fraction_of_errors=a/(a+c) if a+c else None,
        flagged_fraction_of_correct=b/(b+d) if b+d else None,
        flagged_error_rate=a/(a+b) if a+b else None,
        unflagged_error_rate=c/(c+d) if c+d else None,
        fisher_two_sided_p=float(fisher_exact(t).pvalue))
    if (t>0).all():
        test=Table2x2(t,shift_zeros=False)
        result.update(error_risk_ratio=float(test.riskratio),risk_ratio_CI95_low=float(test.riskratio_confint()[0]),risk_ratio_CI95_high=float(test.riskratio_confint()[1]),
            odds_ratio=float(test.oddsratio),odds_ratio_CI95_low=float(test.oddsratio_confint()[0]),odds_ratio_CI95_high=float(test.oddsratio_confint()[1]))
    return result

def run():
    manifest=[]
    def read(path,**kwargs):
        manifest.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return pd.read_csv(path,**kwargs)
    u=read(EVIDENCE/'uniprot_reviewed_length11_50_20261025.tsv',sep='\t',keep_default_na=False)
    f=read(EVIDENCE/'uniprot_fragment_accessions.tsv',sep='\t',keep_default_na=False)
    conf=read(COMPARE/'same_sequence_label_review.csv',keep_default_na=False)
    tests=read(COMPARE/'historical_tests_identity_to_train.csv',keep_default_na=False)
    identities=set(u.Sequence);frags=set(f.Sequence)
    assert frags<=identities
    conflicting=set(conf.loc[conf.scope=='broad_annotation','sequence'])
    accessions=u.groupby('Sequence').Entry.agg(lambda x:';'.join(sorted(set(x)))).to_dict()
    fragment_accessions=f.groupby('Sequence').Entry.agg(lambda x:';'.join(sorted(set(x)))).to_dict()
    # These evidence flags do not depend on any model prediction, confidence or error.
    flags=tests.copy()
    flags['fragment_review_flag']=flags.sequence.isin(frags)
    flags['cross_source_label_disagreement']=flags.sequence.isin(conflicting)
    flags['identity_not_recovered']=~flags.sequence.isin(identities)
    flags['primary_review_flag']=flags.fragment_review_flag|flags.cross_source_label_disagreement
    flags['expanded_review_flag']=flags.primary_review_flag|flags.identity_not_recovered
    flags['exact_match_accessions']=flags.sequence.map(accessions).fillna('')
    flags['fragment_accessions']=flags.sequence.map(fragment_accessions).fillna('')
    save('evidence_flags_before_outcome_join.csv',flags)
    dump('protocol.json',dict(primary='fragment_review_flag OR cross_source_label_disagreement',
        fragment='exact same retained sequence matches any reviewed UniProt entry flagged Fragment in pinned snapshot',
        label_disagreement='same normalized sequence has both broad/operational binary labels across pinned sources; RBC labels excluded',
        unknown='no exact short-entry match is unresolved, not proof of ambiguity; separately reported and expanded sensitivity only',
        comparison_row='unflagged means no selected evidence marker, not proven clear',
        predictions='saved canonical default_multidimensional_fusion_single_structure predictions, original decisions retained',
        primary_statistics='two-sided Fisher exact, risk ratio with log95CI, two primary p values Holm adjusted',
        sensitivity='single flags and expanded union; class strata and Mantel-Haenszel odds ratio controlling original class',
        dependence='Test2 subset of Test1, not independent replication; peptide-level tests exploratory and not family-cluster adjusted',
        no_relabel=True,no_deletion=True,no_model_training=True))
    allrows=[];results=[];strata=[];adjusted=[];summary=[]
    for split in ['test1','test2']:
        path=ROOT/f'runs/default_multidimensional_fusion_single_structure/{split}_predictions.csv'
        pred=read(path,float_precision='round_trip')
        seq=read(ROOT/f'runs/single_structure_test_components/{split}_structure_predictions.csv',float_precision='round_trip')
        assert pred[['source_index','sample_id','label']].equals(seq[['source_index','sample_id','label']])
        base=flags[flags.split==split].rename(columns={'source_id':'sample_id'})
        assert base.sequence.is_unique and pred.sample_id.is_unique
        df=base.merge(pred,on=['sample_id','label'],validate='one_to_one')
        assert len(df)==len(base)==len(pred)
        assert df.sequence.tolist()==seq.sequence.tolist()
        assert np.array_equal(df.prediction.to_numpy(),(df.toxicity_probability>=df.decision_threshold).astype(int).to_numpy())
        assert np.allclose(df.toxicity_probability,df.p_1d*df.weight_1d+df.p_3d*(1-df.weight_1d),atol=2e-9,rtol=0)
        df['error']=df.prediction!=df.label
        df['confusion_category']=np.where(df.label==1,np.where(df.error,'FN','TP'),np.where(df.error,'FP','TN'))
        save(f'{split}_sample_audit.csv',df);allrows.append(df)
        for col in ['primary_review_flag','fragment_review_flag','cross_source_label_disagreement','identity_not_recovered','expanded_review_flag']:
            results.append(stats(df,col,split))
        primary=counts(df,'primary_review_flag')
        save(f'{split}_2x2.csv',pd.DataFrame(dict(evidence_status=['有待核查标记（初筛不确定）','未发现上述标记（不等于已证实明确）'],
            model_wrong=primary[:,0],model_correct=primary[:,1],total=primary.sum(axis=1))))
        perlabel=[]
        for label in [0,1]:
            g=df[df.label==label];perlabel.append(counts(g,'primary_review_flag'))
            r=stats(g,'primary_review_flag',split);r['original_class']=label;strata.append(r)
        if all(t.sum()>0 for t in perlabel):
            cmh=StratifiedTable(np.stack(perlabel,axis=2),shift_zeros=False)
            ci=cmh.oddsratio_pooled_confint()
            adjusted.append(dict(split=split,adjustment='original_binary_class',pooled_odds_ratio=float(cmh.oddsratio_pooled),
                CI95_low=float(ci[0]),CI95_high=float(ci[1]),CMH_p=float(cmh.test_null_odds().pvalue)))
        summary.append(dict(split=split,n=len(df),positive=int(df.label.sum()),errors=int(df.error.sum()),
            TP=int((df.confusion_category=='TP').sum()),TN=int((df.confusion_category=='TN').sum()),FP=int((df.confusion_category=='FP').sum()),FN=int((df.confusion_category=='FN').sum()),
            mcc=float(matthews_corrcoef(df.label,df.prediction)),fragment=int(df.fragment_review_flag.sum()),
            label_disagreement=int(df.cross_source_label_disagreement.sum()),both_flags=int((df.fragment_review_flag&df.cross_source_label_disagreement).sum()),
            unresolved_identity=int(df.identity_not_recovered.sum()),primary_flagged=int(df.primary_review_flag.sum()),
            unresolved_without_primary_flag=int((df.identity_not_recovered&~df.primary_review_flag).sum())))
    res=pd.DataFrame(results);ix=res.definition=='primary_review_flag'
    res.loc[ix,'primary_two_tests_Holm_p']=multipletests(res.loc[ix,'fisher_two_sided_p'].to_numpy(),method='holm')[1]
    save('association_statistics.csv',res);save('class_stratified_association.csv',pd.DataFrame(strata));save('class_adjusted_association.csv',pd.DataFrame(adjusted))
    joined=pd.concat(allrows,ignore_index=True)
    one=joined[joined.split=='test1'].set_index('sequence');two=joined[joined.split=='test2'].set_index('sequence')
    assert set(two.index)<=set(one.index)
    # Identical sequences have consistent evidence and labels; tiny probability/order differences retained as original.
    for col in ['label','primary_review_flag','fragment_review_flag','cross_source_label_disagreement','identity_not_recovered']:
        assert one.loc[two.index,col].tolist()==two[col].tolist(),col
    dump('input_manifest.json',manifest);dump('summary.json',summary)
    dump('verification.json',dict(passed=True,all_samples_accounted_for=True,predictions_match_original_threshold=True,
        test2_sequence_subset_test1=True,evidence_and_labels_consistent_for_shared_sequences=True,
        shared_sequences_different_frozen_decisions=int((one.loc[two.index,'prediction'].to_numpy()!=two.prediction.to_numpy()).sum()),
        original_labels_and_predictions_unchanged=True))
    for x in manifest:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
    print(json.dumps({'summary':summary,'primary':res[ix].where(pd.notna(res[ix]),None).to_dict('records'),'class_adjusted':adjusted},ensure_ascii=True,default=str))

if __name__=='__main__':run()
