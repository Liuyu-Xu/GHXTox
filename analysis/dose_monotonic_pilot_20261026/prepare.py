"""Freeze inherited dose candidates and whole-paper/sequence-purged folds."""
import csv,hashlib,json,datetime
from pathlib import Path
from collections import Counter
R=Path('D:/dachuang2026/GHXTox');O=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def table(n,rs):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
env=R/'analysis/environment_feasibility_20261025';audit=R/'analysis/hemolysis_dose_audit_20261026'
paths=[O/'protocol.json',env/'model_records.json',env/'sequence_clusters.json',audit/'confirmed_non_point_exclusion_panels.json',audit/'all_exact50_quarantine_panels.json',audit/'confirmed_non_point_records.json']
before={str(p):sha(p) for p in paths}
records={r['id']:r for r in read(env/'model_records.json')};clusters={r['sequence']:r['group'] for r in read(env/'sequence_clusters.json')}
datasets={}
for name,file in [('confirmed_overlay_primary','confirmed_non_point_exclusion_panels.json'),('all_exact50_quarantine_sensitivity','all_exact50_quarantine_panels.json')]:
 panels=[];points=[]
 for panel in read(audit/file):
  if panel['n_doses']<3:continue
  pp=sorted(panel['points'],key=lambda x:x['dose_uM']);assert len({p['dose_uM'] for p in pp})==len(pp)==panel['n_doses']
  pmid=str(panel['pmid']);sequence=panel['sequence'];group=clusters[sequence]
  panels.append(dict(panel_id=panel['panel_id'],sequence=sequence,chemical_key=panel['chemical_key'],pmid=pmid,species=panel['species'],pH=panel['pH'],edit_group=group,n_points=len(pp),measurement_scope='metadata_panel_not_gold_experimental_curve',response_span_pp=max(p['hemolysis_percent'] for p in pp)-min(p['hemolysis_percent'] for p in pp)))
  for p in pp:
   r=records[p['id']]
   assert r['sequence']==sequence and r['chemical_key']==panel['chemical_key'] and str(r['pmid'])==pmid and r['species']==panel['species']
   assert r['dose_uM']==p['dose_uM'] and r['hemolysis_percent']==p['hemolysis_percent']
   assert p['dose_uM']>0 and 0<=p['hemolysis_percent']<=100
   points.append(dict(**r,panel_id=panel['panel_id'],edit_group=group,panel_n_points=len(pp),measurement_status='literal_50_percent_potential_endpoint_derived' if p['hemolysis_percent']==50 else 'percent_literal_source_not_fully_verified'))
 datasets[name]=dict(panels=panels,points=points)
primary=datasets['confirmed_overlay_primary'];assert len(primary['panels'])==284 and len(primary['points'])==1305
secondary=datasets['all_exact50_quarantine_sensitivity'];assert len(secondary['panels'])==253 and len(secondary['points'])==1208
excluded={r['id'] for r in read(audit/'confirmed_non_point_records.json')}
assert not excluded&{r['id'] for data in datasets.values() for r in data['points']}
assert all(r['hemolysis_percent']!=50 for r in secondary['points'])
counts=Counter(p['pmid'] for p in primary['panels']);load=[0]*5;pmidfold={}
for pmid in sorted(counts,key=lambda p:(-counts[p],int(p))):
 f=min(range(5),key=lambda k:(load[k],k));pmidfold[pmid]=f;load[f]+=counts[pmid]
folds={};stats=[]
for name,data in datasets.items():
 for p in data['panels']:p['fold']=pmidfold[p['pmid']]
 for r in data['points']:r['fold']=pmidfold[str(r['pmid'])]
 folds[name]=[]
 for f in range(5):
  test=[r for r in data['points'] if r['fold']==f];testgroups={r['edit_group'] for r in test}
  beforepurge=[r for r in data['points'] if r['fold']!=f]
  train=[r for r in beforepurge if r['edit_group'] not in testgroups]
  assert not {r['edit_group'] for r in train}&testgroups
  assert not {str(r['pmid']) for r in train}&{str(r['pmid']) for r in test}
  assert not {r['sequence'] for r in train}&{r['sequence'] for r in test}
  assert len(train)>=80 and len(test)>=3
  entry=dict(fold=f,train_ids=[r['id'] for r in train],test_ids=[r['id'] for r in test],purged_train_ids=[r['id'] for r in beforepurge if r['edit_group'] in testgroups]);folds[name].append(entry)
  stats.append(dict(dataset=name,fold=f,train_points=len(train),test_points=len(test),train_panels=len({r['panel_id'] for r in train}),test_panels=len({r['panel_id'] for r in test}),purged_train_points=len(entry['purged_train_ids']),train_pmid_identifiers=len({str(r['pmid']) for r in train}),test_pmid_identifiers=len({str(r['pmid']) for r in test}),test_edit_groups=len(testgroups)))
 save(name+'_panels.json',data['panels']);table(name+'_point_manifest.csv',data['points'])
save('prepared_data.json',datasets);save('folds.json',folds);save('pmid_to_fold.json',pmidfold);table('split_summary.csv',stats)
after={str(p):sha(p) for p in paths};assert before==after
frozen=[O/'protocol.json',O/'prepared_data.json',O/'folds.json',O/'pmid_to_fold.json']
save('freeze_manifest.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='before_any_new_candidate_fit',input_hashes=before,frozen_hashes={str(p):sha(p) for p in frozen},dataset_counts={k:dict(panels=len(v['panels']),points=len(v['points']),sequences=len({r['sequence'] for r in v['points']}),pmid_identifiers=len({str(r['pmid']) for r in v['points']})) for k,v in datasets.items()},input_unchanged=True,all_test_papers_and_edit_groups_excluded_from_training=True))
print(json.dumps(stats,ensure_ascii=False,indent=2))
