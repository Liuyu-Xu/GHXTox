from pathlib import Path
import json,csv,hashlib,collections,math
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
manifest=read(P/'input_manifest.json')
for rel,sha in manifest['inputs_sha256'].items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==sha,rel
base=read(R/'analysis/ghxtox_shortfall_registry_20261027/all_sample_shortfall_index.json')
atlas=list(csv.DictReader((R/'analysis/expert_disagreement_20261025/training_expert_atlas.csv').open(encoding='utf-8-sig')))
joined=read(P/'all_sample_evidence_join.json');tables=read(P/'joint_tables.json')
assert len(joined)==len(base)==len(atlas)==6387
counter=collections.defaultdict(collections.Counter)
pools=collections.defaultdict(collections.Counter)
for b,a,j in zip(base,atlas,joined):
 assert b['sample_id']==a['sample_id']==j['sample_id'] and b['sequence']==a['sequence']==j['sequence']
 assert b['cys_count']==b['sequence'].count('C')==j['cys_count']
 flag=(a['any_fragment_match']=='True' or a['cross_source_label_disagreement']=='True')
 assert j['uniform_flag']==int(flag)
 assert j['old_archived_wrong']==((j['old_archived_probability']>=j['old_archived_threshold'])!=bool(j['label']))
 assert j['new_wrong']==((j['new_probability']>=j['new_threshold'])!=bool(j['label']))
 k=0 if b['label']==1 and b['cys_count']<4 else 1 if b['label']==0 and b['cys_count']>=4 else 2 if b['label']==1 else 3
 counter[(k,'all')]['n']+=1;counter[(k,'all')]['flag']+=flag
 state='flagged' if flag else 'unflagged_not_certified'
 for c in [(k,'all'),(k,state)]:
  counter[c]['old']+=b['old_archived_wrong'];counter[c]['new']+=b['new_wrong']
  if c[1]!='all':counter[c]['n']+=1;counter[c]['flag']+=flag
 if k in [0,1]:
  if b['old_archived_wrong'] and b['new_wrong']:state='both_streams_wrong'
  elif not b['old_archived_wrong'] and not b['new_wrong']:state='both_streams_correct'
  else:continue
  pools[(k,state)]['n']+=1;pools[(k,state)]['flag']+=flag
for (k,state),c in counter.items():
 out=tables['physical_strata']['physical_'+str(k)][state]
 assert c['n']==out['n'] and c['flag']==out['uniform_flagged']
 assert c['old']==out['versions']['old']['errors'] and c['new']==out['versions']['new']['errors']
for (k,state),c in pools.items():
 out=tables['common_error_control_pools']['physical_'+str(k)][state]
 assert c['n']==out['n'] and c['flag']==out['uniform_flagged']
fits=read(P/'conditional_associations.json')
assert len(fits)==8 and all(x['converged'] for x in fits)
assert all(math.isfinite(x['odds_ratio_difficult_Cys']) for x in fits)
assert not any(j['strict_original_label_verified'] for j in joined)
out={'passed':True,'samples':6387,'source_hashes_unchanged':len(manifest['inputs_sha256']),'physical_evidence_version_tables_rebuilt_independently':len(counter),'common_error_correct_pools':len(pools),'statistical_association_fits_converged':8,'strict_label_gold_records_certified_in_retained_metadata':0,'no_gold_does_not_mean_all_labels_wrong':True,'no_NN_training_or_weight_changes':True,'no_Test_recomputation':True,'limitations':['No complete original source/endpoint provenance','No adequate family common support','Conditional OR not causal or error risk ratio','Historical hypothesis-informed analysis, no new blind confirmation']}
(P/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
