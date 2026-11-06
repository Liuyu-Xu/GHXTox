from pathlib import Path
import json,hashlib
from collections import Counter,defaultdict
import numpy as np
from scipy.special import expit
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(n,x):(O/n).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert read(O/'anchor_gate.json')['passed'];cohort=read(O/'cohort.json');by={r['sample_id']:r for r in cohort};manifest=read(O/'sequence_manifest.json');records=[r for r in manifest if r['mode']!='original' and r['eligible']];orig=np.load(O/'original_scores.npz');mut=np.load(O/'mutant_scores.npz');oi={k:i for i,k in enumerate(orig['sample_ids'].tolist())};mi={k:i for i,k in enumerate(mut['keys'].tolist())};assert len(mi)==len(records)
old=np.load(A/'saprot_standalone_20261026/ensemble_scores.npz');old4=np.load(A/'saprot_ensemble_size_control_20261026/ensemble_scores.npz');z=np.load(A/'saprot_and_robust_classifier_pilot_20261026/frozen_features.npz');fullfold=z['folds'];scale={}
for model,key in [('baseline2','baseline'),('teacher2','SaProt_only_masked')]:scale[model]={f:float(np.std(old[key][fullfold==f])) for f in range(5)}
scale['baseline4']={f:float(np.std(old4['baseline4'][fullfold==f])) for f in range(5)};scale['fusion_masked4']={f:float(np.std(old['fusion_masked'][fullfold==f])) for f in range(5)}
scale['baseline2_clamped']=scale['baseline2'];scale['baseline4_clamped']=scale['baseline4'];save('response_scales.json',scale)
from sys import path
path.insert(0,str(R/'src'))
from ghxtox.models.physchem_features import physchem_feature_vector
pr=[];maxcompositionerror=0.;max_nonCys_phys=0.
for r in records:
 i,j=oi[r['sample_id']],mi[r['key']];c=by[r['sample_id']];pold=np.array(physchem_feature_vector(r['original_sequence']));pnew=np.array(physchem_feature_vector(r['sequence']));err=float(np.max(abs(pold[:8]-pnew[:8])));maxcompositionerror=max(err,maxcompositionerror);assert err<1e-6 and Counter(r['sequence'])==Counter(r['original_sequence'])
 if 'nonCys' in r['mode']:max_nonCys_phys=max(max_nonCys_phys,float(np.max(abs(pold-pnew))));assert np.max(abs(pold-pnew))<1e-6
 row=dict(key=r['key'],sample_id=r['sample_id'],mode=r['mode'],replicate=r['replicate'],fold=r['fold'],changed_fraction=r['changed_fraction'],original_label_metadata_only=c['label'],masked_role=c['fusion_masked4']['role'],structure_role=c['fusion_3Di4']['role'],cys_count=r['original_sequence'].count('C'),cys_min_spacing_change=float(pnew[-1]-pold[-1]),responses={})
 for model in scale:
  a,b=float(orig[model][i]),float(mut[model][j]);pa,pb=float(expit(a)),float(expit(b));row['responses'][model]=dict(original_logodds=a,mutated_logodds=b,original_probability=pa,mutated_probability=pb,probability_change=pb-pa,absolute_probability_change=abs(pb-pa),absolute_logodds_change=abs(b-a),normalized_logodds_change=abs(b-a)/scale[model][r['fold']],original_decision_changed=int((a>=0)!=(b>=0)))
 gap0=float(expit(orig['teacher2'][i])-expit(orig['baseline2'][i]));gap1=float(expit(mut['teacher2'][j])-expit(mut['baseline2'][j]));row.update(teacher_base_gap_original=gap0,teacher_base_gap_mutant=gap1,teacher_gap_direction_retained=int(gap0*gap1>0),direct_physchem_probability_effect=abs(float(expit(mut['baseline2'][j])-expit(mut['baseline2_clamped'][j]))));pr.append(row)
save('per_perturbation_responses.json',pr)
aggregates=[];buckets=defaultdict(list)
for r in pr:buckets[r['sample_id'],r['mode']].append(r)
for (sid,mode),rr in buckets.items():
 c=by[sid];row=dict(sample_id=sid,mode=mode,fold=c['fold'],group_id=c['group_id'],masked_role=c['fusion_masked4']['role'],original_label_metadata_only=c['label'],cys_count=c['sequence'].count('C'),length=len(c['sequence']),replicates=len(rr),changed_fraction=float(np.mean([r['changed_fraction'] for r in rr])),teacher_gap_direction_retained=float(np.mean([r['teacher_gap_direction_retained'] for r in rr])),direct_physchem_probability_effect=float(np.mean([r['direct_physchem_probability_effect'] for r in rr])),responses={})
 for model in scale:row['responses'][model]={f:float(np.mean([r['responses'][model][f] for r in rr])) for f in ['absolute_probability_change','absolute_logodds_change','normalized_logodds_change','original_decision_changed']}
 aggregates.append(row)
save('sample_mean_responses.json',aggregates)
summary=[];modes=['local_nonCys_shuffle','global_nonCys_shuffle','Cys_relocation','global_all_shuffle'];roles=['corrected','harmed','persistent_wrong','persistent_correct']
for mode in modes:
 for role in ['all']+roles:
  rr=[r for r in aggregates if r['mode']==mode and (role=='all' or r['masked_role']==role)]
  for model in scale:
   summary.append(dict(mode=mode,role=role,model=model,n=len(rr),mean_changed_fraction=float(np.mean([r['changed_fraction'] for r in rr])),**{k:float(np.mean([r['responses'][model][k] for r in rr])) for k in ['absolute_probability_change','normalized_logodds_change','original_decision_changed']}))
save('response_summary.json',summary)
allgroups=sorted({r['group_id'] for r in cohort});gi={g:i for i,g in enumerate(allgroups)};ng=len(gi);rng=np.random.default_rng(20261027);W=np.array([np.bincount(rng.integers(0,ng,ng),minlength=ng) for _ in range(2000)],dtype=float);contrasts=[]
for mode in modes:
 for role in ['corrected','harmed']:
  rr=[r for r in aggregates if r['mode']==mode and r['masked_role']==role]
  for measure in ['absolute_probability_change','normalized_logodds_change']:
   d=np.array([r['responses']['teacher2'][measure]-r['responses']['baseline2'][measure] for r in rr]);idx=[gi[r['group_id']] for r in rr];sums=np.bincount(idx,weights=d,minlength=ng);counts=np.bincount(idx,minlength=ng);den=W@counts;ok=den>0;draw=(W[ok]@sums)/den[ok];fd=[float(np.mean([v for r,v in zip(rr,d) if r['fold']==f])) if any(r['fold']==f for r in rr) else None for f in range(5)]
   contrasts.append(dict(mode=mode,role=role,measure=measure,n=len(rr),mean_teacher_minus_baseline=float(d.mean()),CI95=np.quantile(draw,[.025,.975]),CI99_6875=np.quantile(draw,[.05/32,1-.05/32]),fold_differences=fd,interpretation='greater response is sensitivity, not improved accuracy or biology'))
save('paired_sensitivity_comparisons.json',contrasts)
sub=[]
for mode in modes:
 for label in [0,1]:
  for cysbin in ['low','high']:
   rr=[r for r in aggregates if r['mode']==mode and r['original_label_metadata_only']==label and (r['cys_count']<4)==(cysbin=='low')]
   if not rr:continue
   sub.append(dict(mode=mode,original_label_metadata_only=label,cys_bin=cysbin,n=len(rr),changed_fraction=float(np.mean([r['changed_fraction'] for r in rr])),probability_change_teacher=float(np.mean([r['responses']['teacher2']['absolute_probability_change'] for r in rr])),probability_change_baseline=float(np.mean([r['responses']['baseline2']['absolute_probability_change'] for r in rr])),normalized_logodds_teacher=float(np.mean([r['responses']['teacher2']['normalized_logodds_change'] for r in rr])),normalized_logodds_baseline=float(np.mean([r['responses']['baseline2']['normalized_logodds_change'] for r in rr]))))
save('label_Cys_strata.json',sub)
aggindex={(r['sample_id'],r['mode']):r for r in aggregates};pairs=read(A/'saprot_contribution_20261026/matched_controls.json');paired=[]
for p in pairs:
 if p['arm']!='fusion_masked4':continue
 for mode in modes:
  a,b=aggindex.get((p['case_id'],mode)),aggindex.get((p['control_id'],mode))
  if a is None or b is None:continue
  paired.append(dict(case_id=p['case_id'],control_id=p['control_id'],mode=mode,control_role=p['control_role'],family_support=p['family_support'],identity=p['identity'],fold=p['fold'],changed_fraction_case=a['changed_fraction'],changed_fraction_control=b['changed_fraction'],case_minus_control_teacher_response=a['responses']['teacher2']['normalized_logodds_change']-b['responses']['teacher2']['normalized_logodds_change'],case_minus_control_teacher_advantage=(a['responses']['teacher2']['normalized_logodds_change']-a['responses']['baseline2']['normalized_logodds_change'])-(b['responses']['teacher2']['normalized_logodds_change']-b['responses']['baseline2']['normalized_logodds_change'])))
save('matched_response_pairs.json',paired)
diagnostics=[]
for mode in modes:
 rr=[r for r in aggregates if r['mode']==mode];diagnostics.append(dict(mode=mode,n=len(rr),mean_direct_physchem_probability_effect=float(np.mean([r['direct_physchem_probability_effect'] for r in rr])),median_direct_physchem_probability_effect=float(np.median([r['direct_physchem_probability_effect'] for r in rr])),original_teacher_gap_direction_retained=float(np.mean([r['teacher_gap_direction_retained'] for r in rr])),changed_fraction=float(np.mean([r['changed_fraction'] for r in rr]))))
save('direct_descriptor_and_gap_diagnostics.json',diagnostics)
pure=[]
for role in ['corrected','harmed']:
 rr=[r for r in cohort if r['fusion_masked4']['role']==role];pure.append(dict(role=role,n=len(rr),original_baseline2_correct=sum((orig['baseline2'][oi[r['sample_id']]]>=0)==r['label'] for r in rr),original_teacher2_correct=sum((orig['teacher2'][oi[r['sample_id']]]>=0)==r['label'] for r in rr)))
save('original_member_attribution.json',pure)
checks=read(O/'source_hashes.json');sourcechanged=[path for path,h in checks.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h];assert not sourcechanged,sourcechanged
save('verification.json',dict(cohort=len(cohort),original_sequences=763,eligible_perturbation_records=len(pr),sample_mode_records=len(aggregates),composition_length_invariants=True,max_first8_physchem_difference=maxcompositionerror,max_nonCys_all9_physchem_difference=max_nonCys_phys,all_parent_samples_heldout_by_scored_neural_models=True,source_hashes_unchanged=True,neural_models_trained=0,mutant_toxicity_labels_assigned=False,mutant_classification_metrics_computed=False))
print(json.dumps(clean(dict(contrast=contrasts,descriptor_diagnostics=diagnostics,member_attribution=pure)),ensure_ascii=False,indent=2))
