"""Read-only diagnosis of existing OOF predictions and cached biological evidence."""
from pathlib import Path
import json,hashlib,math,re
from collections import Counter
import numpy as np
from scipy.special import expit,log_expit,logsumexp
from Bio.Align import PairwiseAligner
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def save(n,x):(O/n).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
paths=[O/'protocol.json',A/'saprot_ensemble_size_control_20261026/ensemble_scores.npz',A/'saprot_and_robust_classifier_pilot_20261026/frozen_features.npz',A/'saprot_and_robust_classifier_pilot_20261026/oof_logits.json',A/'saprot_standalone_20261026/oof_predictions.json',A/'error_evidence_linkage_20261025/all_training_linked.json',A/'family_mechanism_comparison_20261026/all_training_annotation_index.json',A/'saprot_input_audit_20261026/structure_sequences.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={str(p):sha(p) for p in paths};save('input_manifest.json',manifest)
P=read(paths[0]);z=np.load(paths[2]);s=np.load(paths[1]);ids=z['sample_ids'].tolist();y=z['labels'];fold=z['folds'];groups=z['groups'];n=len(ids);ix={a:i for i,a in enumerate(ids)}
assert ids==s['sample_ids'].tolist() and np.array_equal(y,s['labels'])
old={r['sample_id']:r for r in read(paths[5])};meta=read(paths[6]);struct={r['sample_id']:r for r in read(paths[7])}
for i,k in enumerate(ids):
 assert old[k]['sequence']==str(z['sequences'][i]) and int(old[k]['label'])==y[i] and int(old[k]['fold'])==fold[i]
 assert struct[k]['sequence']==old[k]['sequence'] and meta[k]['sample_id']==k
ref=s['baseline4'];arms=P['candidates'];score={a:s[a] for a in arms};bok=(ref>=0)==y;role={}
for a in arms:
 ok=(score[a]>=0)==y;role[a]=np.where(~bok&ok,'corrected',np.where(bok&~ok,'harmed',np.where(~bok&~ok,'persistent_wrong','persistent_correct')))
features={}
for f in P['features']:
 features[f]=np.array([float(struct[k]['unmasked_structure_n']/struct[k]['length']) if f=='unmasked_fraction' else float(old[k].get(f,'nan') or 'nan') for k in ids])
_,gi=np.unique(groups,return_inverse=True);ng=gi.max()+1
combine=lambda ll:logsumexp(log_expit(np.array(ll)),axis=0)-logsumexp(log_expit(-np.array(ll)),axis=0)
src=read(paths[3]);new=read(paths[4]);seed_roles={};seed_probs={}
for seed in [20261026,20261027]:
 ll={}
 for arm,data in [('baseline_BCE',src),('SaProt_only_3Di',new),('SaProt_only_masked',new)]:
  rr=[r for r in data if r['seed']==seed and r['arm']==arm];assert len(rr)==n
  arr=np.zeros(n)
  for r in rr:arr[ix[r['sample_id']]]=r['logit']
  ll[arm]=arr
 for arm in arms:
  kind='3Di' if arm=='fusion_3Di4' else 'masked';bp=ll['baseline_BCE'];fp=combine([bp,ll['SaProt_only_'+kind]])
  b=(bp>=0)==y;f=(fp>=0)==y;seed_roles[seed,arm]=np.where(~b&f,'corrected',np.where(b&~f,'harmed',np.where(~b&~f,'persistent_wrong','persistent_correct')));seed_probs[seed,arm]=(expit(bp),expit(fp))

rows=[]
for i,k in enumerate(ids):
 m=meta[k];r=dict(sample_id=k,sequence=old[k]['sequence'],label=int(y[i]),fold=int(fold[i]),group_id=str(groups[i]),baseline4_probability=float(expit(ref[i])),baseline2_probability=float(expit(s['baseline2'][i])),name=m['name'],curated_family=m['curated_family'],name_class=m['name_class'],accessions=m['accessions'],fragment_flag=m['fragment_reference_flag'],cross_source_label_conflict=old[k]['cross_source_label_disagreement']=='True',disulfide_status=m['disulfide_evidence_status'],modification_status=m['modification_evidence_status'],annotation_negation_warning=m['annotation_negation_warning'],mechanism_mentions=m['function_mentions'],source_urls=m['source_urls'],cached_papers=m['cached_papers'],features={f:v[i] for f,v in features.items()})
 for arm in arms:
  r[arm]=dict(probability=float(expit(score[arm][i])),role=role[arm][i],seed_roles={str(seed):str(seed_roles[seed,arm][i]) for seed in [20261026,20261027]},both_seed_rescue=all(seed_roles[seed,arm][i]=='corrected' for seed in [20261026,20261027]),seed_probabilities={str(seed):dict(baseline=float(seed_probs[seed,arm][0][i]),fusion=float(seed_probs[seed,arm][1][i])) for seed in [20261026,20261027]})
 rows.append(r)
save('sample_contributions.json',rows)
save('corrected_and_harmed_samples.json',[r for i,r in enumerate(rows) if any(role[a][i] in ['corrected','harmed'] for a in arms)])

def counts(mask,arm):
 rr=role[arm][mask];ret={k:int((rr==k).sum()) for k in ['corrected','harmed','persistent_wrong','persistent_correct']};ret.update(n=int(mask.sum()),net_errors_removed=ret['corrected']-ret['harmed']);ret['correction_rate_of_baseline_errors']=ret['corrected']/(ret['corrected']+ret['persistent_wrong']) if ret['corrected']+ret['persistent_wrong'] else None
 ret['harm_rate_of_baseline_correct']=ret['harmed']/(ret['harmed']+ret['persistent_correct']) if ret['harmed']+ret['persistent_correct'] else None
 return ret
summary=[dict(arm=a,label=label,**counts(np.ones(n,dtype=bool) if label=='all' else y==label,a)) for a in arms for label in ['all',0,1]]
save('contribution_summary.json',summary)
stability=[]
for arm in arms:
 for f in range(5):stability.append(dict(arm=arm,axis='fold',value=f,**counts(fold==f,arm)))
 for seed in [20261026,20261027]:
  rr=seed_roles[seed,arm];stability.append(dict(arm=arm,axis='paired_seed',value=seed,**{k:int((rr==k).sum()) for k in ['corrected','harmed','persistent_wrong','persistent_correct']},ensemble_rescues_reproduced=int(((role[arm]=='corrected')&(rr=='corrected')).sum())))
 save('stability.json',stability)
common=(role[arms[0]]=='corrected')&(role[arms[1]]=='corrected');commonharm=(role[arms[0]]=='harmed')&(role[arms[1]]=='harmed')
flips=[]
for lab in [0,1]:
 for unmask in [False,True]:
  mask=(y==lab)&((features['unmasked_fraction']>0)==unmask);ok3=(score[arms[0]]>=0)==y;okm=(score[arms[1]]>=0)==y
  flips.append(dict(label=lab,has_structure_tokens=unmask,n=int(mask.sum()),structure_better=int((mask&ok3&~okm).sum()),masked_better=int((mask&~ok3&okm).sum())))
save('structure_specificity.json',dict(common_rescues=int(common.sum()),common_harms=int(commonharm.sum()),exclusive_rescues={a:int(((role[a]=='corrected')&~common).sum()) for a in arms},paired_flips=flips,scope='allmasked differences are fold-trained head effects, not sample geometry effects'))

strata={'all':np.ones(n,dtype=bool),'Cys_lt4':features['cys_count']<4,'Cys_ge4':features['cys_count']>=4,'length_le25':features['sequence_length']<=25,'length_gt25':features['sequence_length']>25,'all_structure_masked':features['unmasked_fraction']==0,'some_structure_tokens':features['unmasked_fraction']>0,'cross_source_label_conflict':np.array([r['cross_source_label_conflict'] for r in rows]),'no_cross_source_label_conflict':np.array([not r['cross_source_label_conflict'] for r in rows]),'fragment_flag':np.array([r['fragment_flag'] for r in rows])}
for name in sorted({c for k in ids for c in meta[k]['name_class']}):strata['name_proxy:'+name]=np.array([name in meta[k]['name_class'] for k in ids])
for key in meta[ids[0]]['function_mentions']:strata['mention:'+key]=np.array([bool(meta[k]['function_mentions'][key]) for k in ids])
for key in ['explicit_disulfide','explicit_modification','curated_family_available','cached_full_record']:
 strata[key]=np.array([bool(meta[k]['annotated_intra_pairs'] or meta[k]['annotated_interchain_records']) if key=='explicit_disulfide' else bool(meta[k]['chemical_feature_records']) if key=='explicit_modification' else bool(meta[k]['curated_family']) if key=='curated_family_available' else bool(meta[k]['full_record_accessions']) for k in ids])
sub=[]
for name,mask in strata.items():
 for arm in arms:
  for lab in [0,1]:
   mm=mask&(y==lab)
   if mm.any():sub.append(dict(stratum=name,arm=arm,label=lab,**counts(mm,arm),folds=[dict(fold=f,**counts(mm&(fold==f),arm)) for f in range(5)]))
save('subgroup_contributions.json',sub)

# Standardized descriptive contrasts with original group resampling.
tests=[];rng=np.random.default_rng(20261026);weights=np.array([np.bincount(rng.integers(0,ng,ng),minlength=ng) for _ in range(2000)],dtype=np.float64)
def difference(v,c,d):
 c=c&np.isfinite(v);d=d&np.isfinite(v)
 return float(v[c].mean()-v[d].mean()) if c.any() and d.any() else None
for arm in arms:
 for lab in [0,1]:
  for f,v in features.items():
   c=(role[arm]=='corrected')&(y==lab)&np.isfinite(v);d=(role[arm]=='persistent_wrong')&(y==lab)&np.isfinite(v);delta=difference(v,c,d)
   if delta is None:continue
   scale=float(np.std(v[(y==lab)&np.isfinite(v)]));gcs=np.bincount(gi[c],weights=v[c],minlength=ng);gcn=np.bincount(gi[c],minlength=ng);gds=np.bincount(gi[d],weights=v[d],minlength=ng);gdn=np.bincount(gi[d],minlength=ng)
   nc=weights@gcn;nd=weights@gdn;valid=(nc>0)&(nd>0);ds=(weights[valid]@gcs)/nc[valid]-(weights[valid]@gds)/nd[valid]
   tail=min(1.,2*min((1+np.sum(ds<=0))/(len(ds)+1),(1+np.sum(ds>=0))/(len(ds)+1)))
   fd=[difference(v,c&(fold==j),d&(fold==j)) for j in range(5)];sd=[]
   for seed in [20261026,20261027]:
    rr=seed_roles[seed,arm];sd.append(difference(v,(rr=='corrected')&(y==lab),(rr=='persistent_wrong')&(y==lab)))
   informative=[q for q in fd if q is not None];agree=sum(q*delta>0 for q in informative)
   tests.append(dict(arm=arm,label=lab,feature=f,n_corrected=int(c.sum()),n_persistent_wrong=int(d.sum()),mean_corrected=float(v[c].mean()),mean_persistent_wrong=float(v[d].mean()),difference=delta,standardized_difference=delta/scale if scale else 0,CI95=np.quantile(ds,[.025,.975]).tolist(),bootstrap_sign_tail=tail,fold_differences=fd,informative_folds=len(informative),agreeing_folds=agree,seed_differences=sd,stability_before_q=int(c.sum())>=10 and int(d.sum())>=10 and len(informative)>=4 and agree/len(informative)>=.8 and all(q is not None and q*delta>0 for q in sd)))
order=np.argsort([t['bootstrap_sign_tail'] for t in tests]);q=1.
for rank in range(len(order),0,-1):
 i=order[rank-1];q=min(q,tests[i]['bootstrap_sign_tail']*len(order)/rank);tests[i]['BH_q']=q;tests[i]['stable_candidate']=tests[i]['stability_before_q'] and q<.05
save('feature_comparisons.json',tests)

# Same-label/fold sequence-context matched controls; no spatial outcome-driven matching.
aligner=PairwiseAligner(mode='global',match_score=2,mismatch_score=-1,open_gap_score=-3,extend_gap_score=-.5)
seq=[old[k]['sequence'] for k in ids];tri=[set(t[j:j+3] for j in range(len(t)-2)) for t in seq];aligncache={}
def family(i,j):
 m,t=meta[ids[i]],meta[ids[j]]
 if set(m['curated_family'])&set(t['curated_family']):return 0,'curated_family'
 if m['curated_family'] and t['curated_family']:return 2,'sequence_only'
 if set(m['name_class'])&set(t['name_class']):return 1,'name_class_proxy'
 return 2,'sequence_only'
def identity(i,j):
 key=tuple(sorted((i,j)))
 if key not in aligncache:
  al=aligner.align(seq[i],seq[j])[0];a,b=al.aligned;paired=0;matches=0
  for (a0,a1),(b0,b1) in zip(a,b):
   paired+=int(a1-a0);matches+=sum(u==v for u,v in zip(seq[i][a0:a1],seq[j][b0:b1]))
  aligncache[key]=(matches/max(len(seq[i]),len(seq[j])),paired/len(seq[i]),paired/len(seq[j]))
 return aligncache[key]
pairs=[];unmatched=[]
for arm in arms:
 for targetrole in ['persistent_wrong','persistent_correct']:
  used=set()
  for i in np.where(role[arm]=='corrected')[0]:
   candidates=[j for j in np.where((y==y[i])&(fold==fold[i])&(role[arm]==targetrole))[0] if groups[j]!=groups[i] and groups[j] not in used and min(len(seq[j]),len(seq[i]))/max(len(seq[j]),len(seq[i]))>=.7]
   fam=[j for j in candidates if family(i,j)[0]<2];other=[j for j in candidates if family(i,j)[0]==2];other.sort(key=lambda j:len(tri[i]&tri[j])/max(1,len(tri[i]|tri[j])),reverse=True);candidates=fam+other[:30];qualified=[]
   for j in candidates:
    ident,ci,cj=identity(i,j)
    if ident>=.4 and min(ci,cj)>=.7:qualified.append((family(i,j)[0],-ident,ids[j],j,ci,cj))
   if not qualified:unmatched.append(dict(arm=arm,case_id=ids[i],control_role=targetrole));continue
   tier,negident,_,j,ci,cj=min(qualified);used.add(groups[j]);pairs.append(dict(arm=arm,case_id=ids[i],control_id=ids[j],label=int(y[i]),fold=int(fold[i]),control_role=targetrole,family_support=family(i,j)[1],identity=-negident,coverage_case=ci,coverage_control=cj,case_name=meta[ids[i]]['name'],control_name=meta[ids[j]]['name'],feature_differences={f:features[f][i]-features[f][j] for f in features}))
save('matched_controls.json',pairs);save('unmatched_cases.json',unmatched)
pairsummary=[]
for arm in arms:
 for label in [0,1]:
  for cr in ['persistent_wrong','persistent_correct']:
   pp=[p for p in pairs if p['arm']==arm and p['label']==label and p['control_role']==cr]
   for f in P['features']:
    vv=np.array([p['feature_differences'][f] for p in pp]);vv=vv[np.isfinite(vv)]
    if not len(vv):continue
    fd=[float(np.mean([p['feature_differences'][f] for p in pp if p['fold']==j and np.isfinite(p['feature_differences'][f])])) if any(p['fold']==j and np.isfinite(p['feature_differences'][f]) for p in pp) else None for j in range(5)]
    pairsummary.append(dict(arm=arm,label=label,control_role=cr,feature=f,n_pairs=len(vv),mean_case_minus_control=float(vv.mean()),median_case_minus_control=float(np.median(vv)),positive=int((vv>0).sum()),negative=int((vv<0).sum()),fold_differences=fd,scope='descriptive matching check, not additional significance test'))
save('matched_feature_summary.json',pairsummary)
coverage={arm:{cr:dict(n_pairs=sum(p['arm']==arm and p['control_role']==cr for p in pairs),support=dict(Counter(p['family_support'] for p in pairs if p['arm']==arm and p['control_role']==cr))) for cr in ['persistent_wrong','persistent_correct']} for arm in arms};save('matching_coverage.json',coverage)
assert all(sha(Path(p))==h for p,h in manifest.items())
save('verification.json',dict(n=n,ids_labels_sequences_folds_checked=True,source_files_unchanged=True,models_trained=0,external_tests_used=False,role_totals_valid=all(sum(counts(np.ones(n,dtype=bool),a)[k] for k in ['corrected','harmed','persistent_wrong','persistent_correct'])==n for a in arms),matched_pairs=len(pairs),alignment_pairs=len(aligncache)))
print(json.dumps(clean(dict(summary=summary,structure=read(O/'structure_specificity.json'),stability=stability,stable_features=[t for t in tests if t['stable_candidate']],matching=coverage)),ensure_ascii=False,indent=2))
