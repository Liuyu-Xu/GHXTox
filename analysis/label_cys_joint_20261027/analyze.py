"""Historical label-evidence/Cys association. No predictive model or sample weighting."""
from pathlib import Path
import json,csv,hashlib,collections,math,warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

P=Path(__file__).resolve().parent;R=P.parents[1]
inputs={}
def read(rel):
 f=R/rel;inputs[rel]=hashlib.sha256(f.read_bytes()).hexdigest()
 if f.suffix=='.csv':return list(csv.DictReader(f.open(encoding='utf-8-sig',newline='')))
 return json.loads(f.read_text(encoding='utf-8'))
def save(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
s=read('analysis/ghxtox_shortfall_registry_20261027/all_sample_shortfall_index.json')
t=read('analysis/label_ambiguity_map_20261025/training_triage.json')
a=read('analysis/expert_disagreement_20261025/training_expert_atlas.csv')
oldstats=read('analysis/label_ambiguity_map_20261025/summary.json')
oldtests=read('analysis/test_ambiguity_association_20261025/summary.json')
before=inputs.copy()
assert len(s)==len(t)==len(a)==6387
rows=[]
manual_concern_states={'分子对象问题的具体案例/线索','化学形式证据冲突待确认','实验终点或作用背景需区分'}
for x,y,z in zip(s,t,a):
 assert x['sample_id']==y['sample_id']==z['sample_id'] and x['sequence']==y['sequence']==z['sequence'] and x['label']==int(z['label'])==int(y['label'])
 assert x['old_archived_wrong']==(y['fusion_category'] in ['FP','FN'])
 assert abs(x['old_archived_probability']-float(z['p_fusion']))<1e-12
 fragment=z['any_fragment_match']=='True';conflict=z['cross_source_label_disagreement']=='True'
 assert fragment==x['fragment_flag_not_confirmed_label_error'] and conflict==x['cross_source_conflict_not_confirmed_label_error']
 flag=fragment or conflict;mr=y['manual_review']
 # No source in this archive certifies the original binary labeling endpoint.
 assert z['original_database']=='unknown' and z['endpoint_evidence']=='unknown_in_retained_records'
 original_label_verified=False
 explicit_manual_concern=y['primary_review_status'] in manual_concern_states
 tier='recorded_review_concern_not_proven_error' if flag or explicit_manual_concern else 'insufficient_original_label_provenance_or_not_individually_reviewed'
 row={**x,'fragment':int(fragment),'conflict':int(conflict),'uniform_flag':int(flag),'unresolved_identity':int(z['identity_status']=='no_short_entry_match'),'identity_status':z['identity_status'],'original_database':z['original_database'],'original_endpoint':z['endpoint_evidence'],'strict_original_label_verified':original_label_verified,'evidence_tier':tier,'manual_primary_review_status':y['primary_review_status'],'manual_review_tags':mr['review_tags'] if mr else [],'manual_review_present':mr is not None,'manual_explicit_concern':explicit_manual_concern,'direct_effect_candidate_not_gold':bool(mr and 'direct_effect_supported_error' in mr['review_tags']),'length':len(x['sequence']),'difficult_cys':int((x['label']==1 and x['cys_count']<4) or (x['label']==0 and x['cys_count']>=4))}
 rows.append(row)
assert sum(r['uniform_flag'] for r in rows)==1875
assert sum(r['uniform_flag'] and r['old_archived_wrong'] for r in rows)==158

def describe(rs):
 n=len(rs)
 return {'n':n,'fragment':sum(r['fragment'] for r in rs),'conflict':sum(r['conflict'] for r in rs),'both_flags':sum(r['fragment'] and r['conflict'] for r in rs),'uniform_flagged':sum(r['uniform_flag'] for r in rs),'uniform_flagged_fraction':sum(r['uniform_flag'] for r in rs)/n if n else None,'unresolved_identity':sum(r['unresolved_identity'] for r in rs),'manual_reviewed':sum(r['manual_review_present'] for r in rs),'manual_explicit_concern':sum(r['manual_explicit_concern'] for r in rs),'direct_effect_candidates_not_gold':sum(r['direct_effect_candidate_not_gold'] for r in rs),'strict_original_label_verified':0,'tiers':dict(collections.Counter(r['evidence_tier'] for r in rs)),'versions':{v:{'errors':sum(r[k] for r in rs),'correct':sum(not r[k] for r in rs),'error_rate':sum(r[k] for r in rs)/n if n else None,'flagged_errors':sum(r[k] and r['uniform_flag'] for r in rs)} for v,k in [('old','old_archived_wrong'),('new','new_wrong')]},'sample_ids':[r['sample_id'] for r in rs]}
groups={}
for k in range(4):
 rs=[r for r in rows if 'physical_'+str(k) in r['category_keys']]
 groups['physical_'+str(k)]={'all':describe(rs),'flagged':describe([r for r in rs if r['uniform_flag']]),'unflagged_not_certified':describe([r for r in rs if not r['uniform_flag']])}
pools={}
for k,wrongn,rightn in [(0,151,213),(1,64,357)]:
 rs=[r for r in rows if 'physical_'+str(k) in r['category_keys']]
 wrong=[r for r in rs if r['old_archived_wrong'] and r['new_wrong']]
 correct=[r for r in rs if not r['old_archived_wrong'] and not r['new_wrong']]
 assert len(wrong)==wrongn and len(correct)==rightn
 pools['physical_'+str(k)]={'both_streams_wrong':describe(wrong),'both_streams_correct':describe(correct)}
global_summary=describe(rows)
global_summary['flagged']=describe([r for r in rows if r['uniform_flag']])
global_summary['unflagged_not_certified']=describe([r for r in rows if not r['uniform_flag']])
save('all_sample_evidence_join.json',rows);save('joint_tables.json',{'global':global_summary,'physical_strata':groups,'common_error_control_pools':pools})

# Association only: statistical outcome is saved OOF error, not a new toxicity predictor.
frame=pd.DataFrame([{**{k:r[k] for k in ['label','fold','group_id','length','fragment','conflict','uniform_flag','unresolved_identity','difficult_cys']},'old_error':int(r['old_archived_wrong']),'new_error':int(r['new_wrong'])} for r in rows])
associations=[]
for label in [1,0]:
 data=frame[frame.label==label].copy()
 variables=[k for k in ['fragment','conflict','unresolved_identity'] if data[k].nunique()>1]
 for version in ['old','new']:
  for mode in ['unadjusted','flag_length_fold_identity_adjusted']:
   formula=version+'_error ~ difficult_cys'
   if mode!='unadjusted':formula+=' + '+' + '.join(variables+['bs(length, df=3)','C(fold)'])
   with warnings.catch_warnings(record=True) as warns:
    fit=smf.glm(formula,data=data,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':data.group_id})
   beta=float(fit.params['difficult_cys']);ci=fit.conf_int().loc['difficult_cys'].tolist()
   associations.append({'label':label,'version':version,'mode':mode,'n':len(data),'groups':data.group_id.nunique(),'events':int(data[version+'_error'].sum()),'formula':formula,'converged':bool(fit.converged),'odds_ratio_difficult_Cys':math.exp(beta),'cluster_sandwich_CI95':[math.exp(float(v)) for v in ci],'params':{k:float(v) for k,v in fit.params.items()},'warnings':[str(w.message) for w in warns],'limit':'Historical conditional association, NOT causal, source-adjusted or family-adjusted; group is sequence clustering unit, not full family.'})
save('conditional_associations.json',associations)

background={}
for label in [1,0]:
 rs=[r for r in rows if r['label']==label]
 for kind in ['group_id','curated_family','name_class']:
  buckets=collections.defaultdict(list)
  for r in rs:
   keys=[r[kind]] if kind=='group_id' else r[kind]
   for key in keys:buckets[key].append(r)
  supported={k:v for k,v in buckets.items() if len({r['difficult_cys'] for r in v})==2}
  informative={k:v for k,v in supported.items() if len({r['old_archived_wrong'] for r in v})==2}
  background[f'label{label}_{kind}']={'total_annotated_samples':len({r['sample_id'] for v in buckets.values() for r in v}),'background_bins':len(buckets),'bins_with_both_cys_strata':len(supported),'samples_in_both_cys_bins':len({r['sample_id'] for v in supported.values() for r in v}),'bins_also_varying_old_error':len(informative),'details':[{'background':k,'low_cys_n':sum(r['cys_count']<4 for r in v),'high_cys_n':sum(r['cys_count']>=4 for r in v),'old_errors':sum(r['old_archived_wrong'] for r in v),'new_errors':sum(r['new_wrong'] for r in v),'sample_ids':[r['sample_id'] for r in v]} for k,v in supported.items()]}
background['source_coverage']={'original_database_counts':dict(collections.Counter(r['original_database'] for r in rows)),'original_endpoint_counts':dict(collections.Counter(r['original_endpoint'] for r in rows)),'family_coverage':sum(bool(r['curated_family']) for r in rows),'name_proxy_coverage':sum(bool(r['name_class']) for r in rows),'source_family_adjustment_completed':False,'reason':'All original sources/endpoints unknown; family annotations sparse and prior review biased toward errors; name proxies not actual family labels.'}
save('background_overlap_audit.json',background)

# Existing controls only, no new pairing; deduplicate repeated category citations.
edges=read('analysis/ghxtox_shortfall_registry_20261027/existing_control_edges.json')
byid={r['sample_id']:r for r in rows}
dedup=list({(r['case_id'],r['control_id'],r['source_archive']):r for r in edges}.values())
out=[]
for e in dedup:
 x,y=byid[e['case_id']],byid[e['control_id']]
 out.append({**e,'case_uniform_flag':bool(x['uniform_flag']),'control_uniform_flag':bool(y['uniform_flag']),'case_evidence_tier':x['evidence_tier'],'control_evidence_tier':y['evidence_tier'],'case_cys':x['cys_count'],'control_cys':y['cys_count'],'same_cys_stratum':(x['cys_count']<4)==(y['cys_count']<4),'same_original_label':x['label']==y['label'],'case_new_wrong':x['new_wrong'],'control_new_wrong':y['new_wrong']})
save('reused_control_evidence.json',out)
save('input_manifest.json',{'inputs_sha256':inputs,'prediction_identity':'old archived p_fusion/operating point, new row-specific frozen threshold per shortfall registry; different inner protocols, not matched causal architecture comparison'})
print('JOINT',json.dumps({k:{j:(v['n'],v['uniform_flagged'],v['versions']) for j,v in z.items()} for k,z in groups.items()},ensure_ascii=False),flush=True)
print('POOLS',json.dumps(pools,ensure_ascii=False)[:3000],flush=True)
print('ASSOCIATIONS',[(x['label'],x['version'],x['mode'],x['odds_ratio_difficult_Cys'],x['cluster_sandwich_CI95']) for x in associations],flush=True)
print('BACKGROUNDS',{k:{a:b for a,b in v.items() if a!='details'} for k,v in background.items()},flush=True)
