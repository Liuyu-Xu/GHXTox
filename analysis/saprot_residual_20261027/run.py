"""Seed decomposition of fixed composition residuals; diagnostic labels only."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,re
import numpy as np
from scipy.special import expit,log_expit,logsumexp
from scipy.linalg import solve
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import rbf_kernel
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(n,x):(O/n).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
paths=[O/'protocol.json',A/'saprot_composition_20261027/probe_predictions.npz',A/'saprot_composition_20261027/teacher_probability_models.json',A/'saprot_standalone_20261026/oof_predictions.json',A/'saprot_and_robust_classifier_pilot_20261026/oof_logits.json',A/'saprot_contribution_20261026/sample_contributions.json',A/'family_mechanism_comparison_20261026/all_training_annotation_index.json',A/'error_evidence_linkage_20261025/all_training_linked.json',A/'saprot_contribution_20261026/matched_controls.json'];manifest={str(p):sha(p) for p in paths};save('input_manifest.json',manifest)
P=read(paths[0]);z=np.load(paths[1]);ids=z['sample_ids'].tolist();ix={k:i for i,k in enumerate(ids)};labels=z['labels'];fold=z['folds'];half=z['halves'];groups=z['groups'];n=len(ids);co={r['sample_id']:r for r in read(paths[5])};meta=read(paths[6]);evidence={r['sample_id']:r for r in read(paths[7])};seedlist=[20261026,20261027];teach=[];base=[]
for seed in seedlist:
 ll=np.full(n,np.nan);bb=ll.copy()
 for r in read(paths[3]):
  if r['seed']==seed and r['arm']=='SaProt_only_masked':i=ix[r['sample_id']];assert r['label']==labels[i] and r['fold']==fold[i];ll[i]=r['logit']
 for r in read(paths[4]):
  if r['seed']==seed and r['arm']=='baseline_BCE':i=ix[r['sample_id']];assert r['fold']==fold[i];bb[i]=r['logit']
 assert np.isfinite(ll).all() and np.isfinite(bb).all();teach.append(ll);base.append(bb)
teach=np.array(teach).T;base=np.array(base).T;tp=expit(teach);bp=expit(base);assert np.max(abs(tp.mean(axis=1)-z['true_teacher_probability']))<1e-12
alphabet='ACDEFGHIKLMNPQRSTVWY';X=np.array([[co[k]['sequence'].count(a)/len(co[k]['sequence']) for a in alphabet]+[len(co[k]['sequence'])] for k in ids],float);pred=np.full((n,2),np.nan);saved=[];modelchecks=[]
for m in read(paths[2]):
 if m['arm']!='composition_RBF':continue
 ff,hh=m['fold'],m['evaluation_half'];tr=(fold==ff)&(half!=hh);te=(fold==ff)&(half==hh);assert set(groups[tr]).isdisjoint(groups[te]);assert np.array_equal(np.where(tr)[0],m['train_indices']);mu=np.array(m['composition_mean']);sd=np.array(m['composition_scale']);xt=(X[tr]-mu)/sd;xv=(X[te]-mu)/sd;K=rbf_kernel(xt,gamma=1/21);Kv=rbf_kernel(xv,xt,gamma=1/21);ym=tp[tr].mean(axis=0);coef=solve(K+np.eye(tr.sum()),tp[tr]-ym,assume_a='pos');pred[te]=Kv@coef+ym
 coefdiff=float(np.max(abs(coef.mean(axis=1)-m['teacher_coefficient'])));modelchecks.append(dict(fold=ff,half=hh,max_mean_coefficient_difference=coefdiff,groups_disjoint=True));saved.append(dict(fold=ff,half=hh,train_indices=np.where(tr)[0],teacher_target_means=ym,coefficients=coef,composition_mean=mu,composition_scale=sd));print(f'seed-target split fold{ff} half{hh}',flush=True)
assert np.isfinite(pred).all();avgdiff=float(np.max(abs(pred.mean(axis=1)-z['teacher_composition_RBF'])));assert avgdiff<1e-10
predclip=np.clip(pred,0,1);rawres=tp-pred;res=tp-predclip;oriented=res*(2*labels[:,None]-1)
def actual_score(b,t):return np.logaddexp(log_expit(b),log_expit(t))-np.logaddexp(log_expit(-b),log_expit(-t))
def approximate_score(b,p):
 with np.errstate(divide='ignore'):return np.logaddexp(log_expit(b),np.log(p))-np.logaddexp(log_expit(-b),np.log1p(-p))
actualseed=actual_score(base,teach);approxseed=approximate_score(base,predclip);baseok=(base>=0)==labels[:,None];actualok=(actualseed>=0)==labels[:,None];approxok=(approxseed>=0)==labels[:,None]
raw4=z['fusion_baseline4'];actual4=z['fusion_actual_masked_fusion'];approx4=z['fusion_composition_RBF'];b4ok=(raw4>=0)==labels;a4ok=(actual4>=0)==labels;p4ok=(approx4>=0)==labels
unexplained=~b4ok&a4ok&~p4ok;reproduced=~b4ok&a4ok&p4ok;harms=b4ok&~a4ok;wrong=~b4ok&~a4ok;correct=b4ok&a4ok
classes=np.where(unexplained,'unreproduced_rescue',np.where(reproduced,'reproduced_rescue',np.where(harms,'fusion_harm',np.where(wrong,'persistent_wrong','persistent_correct'))));assert unexplained.sum()==59 and reproduced.sum()==37 and harms.sum()==62
double_rescue=np.all(~baseok&actualok,axis=1);double_harm=np.all(baseok&~actualok,axis=1);direction_help=np.all(oriented>0,axis=1);direction_bad=np.all(oriented<0,axis=1);strict=np.all(~baseok&actualok&~approxok,axis=1)&direction_help;mirror=np.all(baseok&~actualok&approxok,axis=1)&direction_bad;strong=strict&np.all(oriented>=.05,axis=1);badstrong=mirror&np.all(oriented<=-.05,axis=1)
negation=re.compile(r'\b(?:no|not|without|lack\w*)\b|non[- ]?(?:ha?emol|cytol)',re.I)
rows=[];flags={};flagnames=set()
def directcontexts(r):
 return {(str(a.get('source_id','')),str(a.get('endpoint','')),str(a.get('assay_object','')),str(a.get('exposure_minutes',''))) for a in r.get('assay_records',[]) if a.get('evidence_status')=='有直接测量记录' and a.get('result') is not None}
for i,k in enumerate(ids):
 c,m,e=co[k],meta[k],evidence[k];fl={};sent={}
 for name,ss in m['function_mentions'].items():
  pos=[s for s in ss if not negation.search(s)];amb=[s for s in ss if negation.search(s)];fl['mention_positive_candidate:'+name]=bool(pos);fl['mention_negation_or_mixed:'+name]=bool(amb);sent[name]=dict(positive_candidates=pos,negation_or_mixed=amb)
 fl['explicit_disulfide_annotation']=m['disulfide_evidence_status']=='explicit_annotation';fl['annotated_disulfide_crossing']=bool(m['annotated_arc_crossings'] and m['annotated_arc_crossings']>0);fl['explicit_modification_annotation']=m['modification_evidence_status']=='explicit_annotation';fl['interchain_bond_annotation']=bool(m['annotated_interchain_records']);fl['fragment_reference_flag']=m['fragment_reference_flag'];fl['cross_source_label_conflict']=c['cross_source_label_conflict'];fl['full_feature_record_cached']=bool(m['full_record_accessions']);fl['any_name_class']=bool(m['name_class']);fl['any_curated_family']=bool(m['curated_family']);fl['direct_assay_record_cached']=bool(directcontexts(e))
 for name in m['name_class']:fl['name_proxy:'+name]=True
 for name in m['curated_family']:fl['curated_family:'+name]=True
 flags[k]=fl;flagnames.update(fl)
 rows.append(dict(sample_id=k,sequence=c['sequence'],original_label=int(labels[i]),fold=int(fold[i]),group_id=str(groups[i]),cohort=str(classes[i]),name=m['name'],name_class=m['name_class'],curated_family=m['curated_family'],source_urls=m['source_urls'],cached_papers=m['cached_papers'],features=c['features'],mechanism_flags=fl,mechanism_sentences=sent,explicit_disulfide_pairs=m['annotated_intra_pairs'],explicit_interchain_records=m['annotated_interchain_records'],explicit_modification_records=m['chemical_feature_records'],manual_primary_status=e.get('manual_primary_status'),assay_summary=e.get('assay_summary'),assay_records=e.get('assay_records',[]),teacher_probabilities=tp[i],composition_estimates_raw=pred[i],composition_residual_raw=rawres[i],composition_residual_clipped=res[i],label_oriented_residual=oriented[i],baseline_seed_probabilities=bp[i],seed_fusion_probabilities=expit(actualseed[i]),composition_seed_fusion_probabilities=expit(approxseed[i]),both_seed_baseline_wrong_and_actualfusion_correct=bool(double_rescue[i]),both_seed_baseline_correct_and_actualfusion_wrong=bool(double_harm[i]),both_residual_directions_helpful=bool(direction_help[i]),both_residual_directions_misleading=bool(direction_bad[i]),strict_useful_residual=bool(strict[i]),strong_useful_residual=bool(strong[i]),strict_misleading_residual=bool(mirror[i]),strong_misleading_residual=bool(badstrong[i])))
save('all_residual_records.json',rows);save('unreproduced59_all.json',[r for i,r in enumerate(rows) if unexplained[i]]);save('stable_unreproduced_rescues.json',[r for i,r in enumerate(rows) if unexplained[i]&strict[i]]);save('all_harms62.json',[r for i,r in enumerate(rows) if harms[i]])
_,gi=np.unique(groups,return_inverse=True);ng=gi.max()+1;rng=np.random.default_rng(20261027);W=np.array([np.bincount(rng.integers(0,ng,ng),minlength=ng) for _ in range(2000)],float)
summ=[]
for name in sorted(set(classes)):
 for lab in ['all',0,1]:
  mm=(classes==name)&(np.ones(n,dtype=bool) if lab=='all' else labels==lab);num=int(mm.sum())
  if not num:continue
  gc=np.bincount(gi,weights=mm.astype(int),minlength=ng);gs=np.bincount(gi,weights=(mm&strict).astype(int),minlength=ng);den=W@gc;valid=den>0;frac=(W[valid]@gs)/den[valid]
  rr=rawres[mm];pear=float(np.corrcoef(rr.T)[0,1]) if num>2 and rr[:,0].std()>0 and rr[:,1].std()>0 else None;spear=float(spearmanr(rr[:,0],rr[:,1]).statistic) if num>2 and rr[:,0].std()>0 and rr[:,1].std()>0 else None
  summ.append(dict(cohort=name,label=lab,n=num,both_seed_actual_rescues=int((mm&double_rescue).sum()),both_seed_actual_harms=int((mm&double_harm).sum()),both_residuals_helpful=int((mm&direction_help).sum()),both_residuals_misleading=int((mm&direction_bad).sum()),strict_useful_residual=int((mm&strict).sum()),strong_useful_residual=int((mm&strong).sum()),strict_misleading_residual=int((mm&mirror).sum()),strong_misleading_residual=int((mm&badstrong).sum()),strict_fraction_CI95=np.quantile(frac,[.025,.975]),raw_residual_Pearson=pear,raw_residual_Spearman=spear,fold_n=[int((mm&(fold==f)).sum()) for f in range(5)],fold_strict_useful=[int((mm&strict&(fold==f)).sum()) for f in range(5)],mean_oriented_residual_by_seed=oriented[mm].mean(axis=0),scope='outcome-selected diagnostic; shared encoders/folds, not independent biological replication'))
save('stability_summary.json',summ)
selections={'unreproduced59':unexplained,'reproduced37':reproduced,'harms62':harms,'strict_unreproduced':unexplained&strict,'strict_harms':harms&mirror,'persistent_correct':correct,'persistent_wrong':wrong};annotations=[]
for cohort,mm in selections.items():
 for lab in [0,1]:
  mask=mm&(labels==lab);ii=np.where(mask)[0]
  for f in sorted(flagnames):
   members=[ids[i] for i in ii if flags[ids[i]].get(f,False)]
   if members:annotations.append(dict(cohort=cohort,label=lab,flag=f,n_population=len(ii),n_flag=len(members),sample_ids=members,folds=sorted({int(fold[ix[k]]) for k in members}),scope='annotation coverage/sentence candidates; no causal enrichment test'))
save('annotation_class_summary.json',annotations)
pairdata=[];oldpairs=[p for p in read(paths[8]) if p['arm']=='fusion_masked4'];byrow={r['sample_id']:r for r in rows};fields=['sequence_length','cys_count','cys_span_fraction','net_charge_proxy','mean_hydropathy','mean_plddt','hydrophobic_contact_excess','positive_hydrophobic_contact_excess','charge_hydrophobic_separation','ca_radius_gyration']
for p in oldpairs:
 c,d=byrow[p['case_id']],byrow[p['control_id']];assert c['original_label']==d['original_label'] and c['fold']==d['fold'] and c['group_id']!=d['group_id'];contexts=directcontexts(evidence[c['sample_id']])&directcontexts(evidence[d['sample_id']]);diffs={}
 for f in fields:
  a,b=c['features'][f],d['features'][f];diffs[f]=a-b if a is not None and b is not None else None
 pairdata.append(dict(**{k:v for k,v in p.items() if k!='feature_differences'},case_new_cohort=c['cohort'],case_strict=c['strict_useful_residual'],case_strong=c['strong_useful_residual'],control_strict_misleading=d['strict_misleading_residual'],case_raw_residual=c['composition_residual_raw'],control_raw_residual=d['composition_residual_raw'],case_mechanism_flags=c['mechanism_flags'],control_mechanism_flags=d['mechanism_flags'],shared_direct_assay_contexts=[list(x) for x in contexts],feature_differences=diffs))
save('retained_matched_pairs.json',pairdata)
pairsummary=[]
for group in ['unreproduced_rescue','reproduced_rescue']:
 for strict_only in [False,True]:
  for role in ['persistent_wrong','persistent_correct']:
   pp=[p for p in pairdata if p['case_new_cohort']==group and p['control_role']==role and (not strict_only or p['case_strict'])]
   if not pp:continue
   pairsummary.append(dict(case_cohort=group,strict_only=strict_only,control_role=role,n_pairs=len(pp),distinct_cases=len({p['case_id'] for p in pp}),distinct_case_groups=len({byrow[p['case_id']]['group_id'] for p in pp}),folds=sorted({p['fold'] for p in pp}),family_support=dict(Counter(p['family_support'] for p in pp)),direct_context_matched_pairs=sum(bool(p['shared_direct_assay_contexts']) for p in pp),mean_case_minus_control={f:float(np.mean([p['feature_differences'][f] for p in pp if p['feature_differences'][f] is not None])) if any(p['feature_differences'][f] is not None for p in pp) else None for f in fields}))
save('matched_summary.json',pairsummary)
gate=[]
for ann in annotations:
 if ann['cohort']!='strict_unreproduced':continue
 f=ann['flag']
 if not (f.startswith('mention_positive_candidate:') or f.startswith('name_proxy:') or f.startswith('curated_family:') or f in ['explicit_disulfide_annotation','explicit_modification_annotation','interchain_bond_annotation','annotated_disulfide_crossing']):continue
 members=set(ann['sample_ids']);pp=[p for p in pairdata if p['case_id'] in members and p['case_strict']];direct=sum(bool(p['shared_direct_assay_contexts']) for p in pp);criteria=dict(atleast5_cases=ann['n_flag']>=5,atleast3folds=len(ann['folds'])>=3,atleast5_pairs=len(pp)>=5,direct_endpoint_consistent_pairs_atleast5=direct>=5);gate.append(dict(flag=f,label=ann['label'],n_stable_cases=ann['n_flag'],folds=ann['folds'],retained_pairs=len(pp),direct_context_pairs=direct,criteria=criteria,passes=all(criteria.values()),sample_ids=ann['sample_ids']))
save('mechanism_gate.json',gate);save('seed_probe_models.json',saved);save('seed_decomposition_checks.json',dict(max_mean_prediction_difference=avgdiff,max_mean_coefficient_difference=max(r['max_mean_coefficient_difference'] for r in modelchecks),per_split=modelchecks));np.savez_compressed(O/'seed_residuals.npz',sample_ids=np.array(ids),labels=labels,folds=fold,halves=half,groups=groups,teacher_logits=teach,baseline_logits=base,composition_predictions=pred,residual_raw=rawres,residual_clipped=res,oriented_residual=oriented,actual_seed_fusion_scores=actualseed,composition_seed_fusion_scores=approxseed,classes=classes,strict_useful=strict,strict_misleading=mirror,strong_useful=strong,strong_misleading=badstrong)
assert all(sha(Path(path))==h for path,h in manifest.items());save('verification.json',dict(n=n,new_seed_target_probe_solves=10,new_neural_fits=0,neural_models_loaded=0,mean_seed_predictor_matches_previous_RBF=True,max_mean_prediction_difference=avgdiff,all_meta_train_eval_groups_disjoint=True,no_toxicity_labels_for_probe_fitting=True,unchanged_source_hashes=True,no_new_test_predictions=True,all_prior_masked_matches_retained=len(pairdata),mechanism_classes_passing=sum(g['passes'] for g in gate)))
print(json.dumps(clean(dict(stability=[r for r in summ if r['label']=='all'],matched=pairsummary,mechanism_gate=gate)),ensure_ascii=False,indent=2))
