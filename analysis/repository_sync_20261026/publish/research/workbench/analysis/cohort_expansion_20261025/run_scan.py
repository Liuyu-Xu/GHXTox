"""Prediction-independent cohort selection, followed by descriptive archived OOF audit."""
import csv, hashlib, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from scipy.stats import fisher_exact

ROOT=Path('D:/dachuang2026/GHXTox'); OUT=Path(__file__).resolve().parent
ATLAS=ROOT/'analysis/expert_disagreement_20261025/training_expert_atlas.csv'
ANNOT=ROOT/'analysis/ghxtox_error_atlas_20261025/evidence_phase2/training_uniprot_exact_matches.csv'
PROTOCOL=OUT/'protocol.json'
P=json.loads(PROTOCOL.read_text(encoding='utf-8')); rng=np.random.default_rng(P['seed'])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(name,df): df.to_csv(OUT/name,index=False,encoding='utf-8-sig')
before={str(p):sha(p) for p in [ATLAS,ANNOT,PROTOCOL]}
d=pd.read_csv(ATLAS,float_precision='round_trip'); assert len(d)==6387 and d.sample_id.is_unique
a=d.set_index('sample_id').loc[P['anchors']]
assert d.groupby('group_id').fold.nunique().max()==1
anchor_groups=set(a.group_id)
ann=pd.read_csv(ANNOT,keep_default_na=False)
regex=re.compile(P['annotation_screen']['regex'],re.I)
mentions=defaultdict(list); coverage=set()
for _,r in ann.iterrows():
    text=str(r['Function [CC]'])
    if text.strip(): coverage.add(r['sample_id'])
    if regex.search(text):
        mentions[r['sample_id']].append(dict(accession=r['Entry'],function=text,source='https://www.uniprot.org/uniprotkb/'+r['Entry']+'/entry'))
aligner=PairwiseAligner(mode='global',match_score=2,mismatch_score=-1,open_gap_score=-3,extend_gap_score=-0.5)
neighbor_details=[]; nearest=[]
for i,r in d.iterrows():
    seq=r.sequence; best=(0.,'',0.)
    for sid,ar in a.iterrows():
        ratio=min(len(seq),len(ar.sequence))/max(len(seq),len(ar.sequence))
        if ratio<P['primary_sequence_neighbor']['min_length_ratio']: continue
        alignment=aligner.align(seq,ar.sequence)[0]
        identical=sum(sum(x==y for x,y in zip(seq[s:e],ar.sequence[t:u])) for (s,e),(t,u) in zip(*alignment.aligned))
        identity=identical/max(len(seq),len(ar.sequence))
        if identity>best[0]: best=(identity,sid,ratio)
        if identity>=P['primary_sequence_neighbor']['min_identity']:
            neighbor_details.append(dict(sample_id=r.sample_id,anchor_id=sid,identity=identity,length_ratio=ratio))
    nearest.append(best)
    if i%1500==0: print('sequence scan',i,flush=True)
d['max_anchor_global_identity']=[v[0] for v in nearest]
d['nearest_anchor']=[v[1] for v in nearest]
d['nearest_anchor_length_ratio']=[v[2] for v in nearest]
d['sequence_neighbor']=d.max_anchor_global_identity>=P['primary_sequence_neighbor']['min_identity']
q=P['primary_physicochemical']
d['physicochemical']=d.length.between(q['min_length'],q['max_length']) & (d.cys_count<=q['max_cys']) & (d.net_charge_proxy>=q['min_net_charge_KR_minus_DE']) & (d.mean_hydropathy>=q['min_mean_Kyte_Doolittle'])
d['annotation_mention']=d.sample_id.isin(mentions)
d['function_annotation_available']=d.sample_id.isin(coverage)
d['union']=d[['sequence_neighbor','physicochemical','annotation_mention']].any(axis=1)
d['physicochemical_non_neighbor']=d.physicochemical & ~d.sequence_neighbor
d['anchor_group_excluded']=d.group_id.isin(anchor_groups)
d['candidate_annotation_accessions']=d.sample_id.map(lambda s:';'.join(x['accession'] for x in mentions[s]))
table('all_training_screen.csv',d)
table('selected_candidates.csv',d[d.union])
table('anchor_sequence_neighbors.csv',pd.DataFrame(neighbor_details))
save('annotation_mentions.json',dict(mentions))
print('selection frozen',int(d.union.sum()),flush=True)

rate_rows=[]
for excluded in [False,True]:
    base=d[~d.anchor_group_excluded] if excluded else d
    for co in P['cohorts_reported']:
        for scope,folds in [('all',[0,1,2,3,4]),('discovery',[0,1,2]),('internal_replication',[3,4])]:
            for label in [0,1]:
                sub=base[(base.label==label)&base.fold.isin(folds)]
                for inside in [True,False]:
                    b=sub[sub[co]==inside]
                    n=len(b); err=int(b.fusion_error.sum())
                    agg=b.groupby('group_id').fusion_error.agg(['sum','count']).to_numpy()
                    if len(agg):
                        draw=rng.integers(0,len(agg),size=(P['bootstrap_replicates'],len(agg)))
                        bs=agg[draw].sum(axis=1); ci=np.quantile(bs[:,0]/bs[:,1],[.025,.975]).tolist()
                    else: ci=[None,None]
                    rate_rows.append(dict(anchor_groups_excluded=excluded,cohort=co,scope=scope,original_label=label,inside=inside,n=n,n_groups=len(agg),errors=err,error_rate=err/n if n else None,ci_low=ci[0],ci_high=ci[1],FN=int((b.fusion_category=='FN').sum()),TP=int((b.fusion_category=='TP').sum()),FP=int((b.fusion_category=='FP').sum()),TN=int((b.fusion_category=='TN').sum()),sequence_errors=int(b.sequence_error.sum()),structure_errors=int(b.structure_error.sum()),disagreements=int(b.expert_decision_disagreement.sum()),fusion_harms_sequence=int(b.fusion_harms_sequence.sum()),fusion_corrects_sequence=int(b.fusion_corrects_sequence.sum())))
rates=pd.DataFrame(rate_rows); table('cohort_rates.csv',rates)
fold_rows=[]
for co in P['cohorts_reported']:
    for fold in range(5):
        for label in [0,1]:
            sub=d[(~d.anchor_group_excluded)&(d.fold==fold)&(d.label==label)]
            b=sub[sub[co]]; out=sub[~sub[co]]
            matrix=[[int(b.fusion_error.sum()),int((~b.fusion_error).sum())],[int(out.fusion_error.sum()),int((~out.fusion_error).sum())]]
            fisher_p=float(fisher_exact(matrix).pvalue) if len(b) and len(out) else None
            fold_rows.append(dict(cohort=co,fold=fold,label=label,n=len(b),errors=int(b.fusion_error.sum()),rate=float(b.fusion_error.mean()) if len(b) else None,background_n=len(out),background_errors=int(out.fusion_error.sum()),background_rate=float(out.fusion_error.mean()) if len(out) else None,descriptive_fisher_p=fisher_p))
table('fold_rates.csv',pd.DataFrame(fold_rows))

# Fixed global-composition matching among original-positive broad candidates.
b=d[d.physicochemical & ~d.anchor_group_excluded & (d.label==1)].copy()
cases=b[b.fusion_category=='FN']; controls=b[b.fusion_category=='TP']; pools={}
for ix,r in cases.iterrows():
    eligible=controls[(controls.fold==r.fold)&(controls.group_id!=r.group_id)&((controls.length-r.length).abs()<=3)&((controls.net_charge_proxy-r.net_charge_proxy).abs()<=2)&((controls.mean_hydropathy-r.mean_hydropathy).abs()<=.5)&((controls.cys_count-r.cys_count).abs()<=1)].copy()
    eligible['match_distance']=((eligible.length-r.length)/3)**2+((eligible.net_charge_proxy-r.net_charge_proxy)/2)**2+((eligible.mean_hydropathy-r.mean_hydropathy)/.5)**2+(eligible.cys_count-r.cys_count)**2
    pools[ix]=eligible.sort_values(['match_distance','sample_id'])
used=set(); pairs=[]
for ix in sorted(pools,key=lambda ix:(len(pools[ix]),cases.loc[ix,'sample_id'])):
    r=cases.loc[ix]
    if r.group_id in used: continue
    for j,c in pools[ix].iterrows():
        if c.group_id in used: continue
        used.update([r.group_id,c.group_id]); pair=dict(case_id=r.sample_id,control_id=c.sample_id,case_group=r.group_id,control_group=c.group_id,fold=int(r.fold),scope='discovery' if r.fold in [0,1,2] else 'internal_replication',distance=float(c.match_distance))
        for f in ['length','net_charge_proxy','mean_hydropathy','cys_count']+P['feature_comparison']['features']:
            pair['case_'+f]=float(r[f]); pair['control_'+f]=float(c[f])
        pairs.append(pair); break
pairdf=pd.DataFrame(pairs); table('matched_FN_TP_pairs.csv',pairdf)
tests=[]
for scope in ['discovery','internal_replication']:
    part=pairdf[pairdf.scope==scope] if len(pairdf) else pairdf
    for feature in P['feature_comparison']['features']:
        diff=(part['case_'+feature]-part['control_'+feature]).to_numpy() if len(part) else np.array([])
        diff=diff[np.isfinite(diff)]; n=len(diff)
        if n>=2:
            boot=diff[rng.integers(0,n,size=(2000,n))].mean(axis=1); ci=np.quantile(boot,[.025,.975]).tolist()
            perm=(rng.choice([-1,1],size=(5000,n))*diff).mean(axis=1)
            pval=(1+int((np.abs(perm)>=abs(diff.mean())-1e-14).sum()))/5001
        else: ci=[None,None]; pval=1.
        tests.append(dict(scope=scope,feature=feature,n_pairs=n,mean_FN_minus_TP=float(diff.mean()) if n else None,ci_low=ci[0],ci_high=ci[1],sign_flip_p=pval))
order=sorted(range(len(tests)),key=lambda i:tests[i]['sign_flip_p']); last=1.
for rank in range(len(order),0,-1):
    ix=order[rank-1]; last=min(last,tests[ix]['sign_flip_p']*len(order)/rank); tests[ix]['BH_q_all12']=last
table('matched_feature_tests.csv',pd.DataFrame(tests))
save('matched_feature_tests.json',tests)
summary=dict(n_total=len(d),original_errors=int(d.fusion_error.sum()),anchors=P['anchors'],excluded_anchor_groups=sorted(anchor_groups),n_in_anchor_groups=int(d.anchor_group_excluded.sum()),function_annotation_available=int(d.function_annotation_available.sum()),
    cohorts={co:dict(all_n=int(d[co].sum()),without_anchor_groups_n=int((d[co]&~d.anchor_group_excluded).sum()),original_positive_n=int((d[co]&(d.label==1)).sum()),original_negative_n=int((d[co]&(d.label==0)).sum())) for co in P['cohorts_reported']},
    matching=dict(eligible_FN=len(cases),eligible_TP=len(controls),matched_pairs=len(pairs),unique_groups=len(used)),
    stable_features=[f for f in P['feature_comparison']['features'] if all(next(t for t in tests if t['feature']==f and t['scope']==scope)['BH_q_all12']<.05 for scope in ['discovery','internal_replication']) and np.prod([next(t for t in tests if t['feature']==f and t['scope']==scope)['mean_FN_minus_TP'] or 0 for scope in ['discovery','internal_replication']])>0])
save('summary.json',summary)
after={str(p):sha(p) for p in [ATLAS,ANNOT,PROTOCOL]}; assert before==after
assert not (set(d.sample_id)-set(pd.read_csv(ATLAS).sample_id))
assert not set(pairdf.case_group)&set(pairdf.control_group) if len(pairdf) else True
save('verification.json',dict(input_hashes_before=before,input_hashes_after=after,inputs_unchanged=True,selection_uses_no_prediction_or_label=True,no_test_inputs=True,no_training=True,group_fold_consistent=True,matched_groups_not_reused=True,n_screened=len(d)))
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
