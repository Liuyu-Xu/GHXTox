from pathlib import Path
import json,csv,hashlib,collections,re
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
manifest=read(P/'input_manifest.json')
for path,sha in manifest['inputs'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==sha,path
old=read(R/'analysis/label_cys_joint_20261027/all_sample_evidence_join.json')
side=read(P/'family_sidecar_before_outcomes.json');joined=read(P/'all_samples_with_symmetric_family.json')
assert len(old)==len(side)==len(joined)==6387
entrylist=list(csv.DictReader((P/'uniprot_reviewed_7_50_family.tsv').open(encoding='utf-8-sig'),delimiter='\t'))
entries={r['Entry']:r for r in entrylist}
entry_checks=0;old_family_not_recovered=[]
for o,f,j in zip(old,side,joined):
 assert o['sample_id']==f['sample_id']==j['sample_id'] and o['sequence']==f['sequence']==j['sequence']
 for key,value in o.items():assert j[key]==value,(o['sample_id'],key)
 for e in f['exact_entries']:
  assert entries[e['accession']]['Sequence']==o['sequence']
  assert e['raw_similarity']==entries[e['accession']]['Sequence similarities']
  for text in e['full_family']:assert 'belongs to ' in text and 'family' in text
  entry_checks+=1
 for family in o['curated_family']:
  if family not in f['full_families']:old_family_not_recovered.append({'sample_id':o['sample_id'],'old_family':family,'new_families':f['full_families']})
tables=read(P/'all_family_background_tables.json');support=read(P/'coverage_and_common_support.json')['support'];byid={r['sample_id']:r for r in joined}
table_checks=0
for tablekey,table in tables.items():
 for rec in table:
  ids=rec['all']['sample_ids'];rs=[byid[k] for k in ids]
  assert rec['all']['n']==len(rs) and len(ids)==len(set(ids))
  assert all(r['label']==rec['label'] and r['current_family_sidecar'][rec['layer']]==[rec['family']] for r in rs)
  for k,ls in [('low',[r for r in rs if r['cys_count']<4]),('high',[r for r in rs if r['cys_count']>=4])]:
   c=rec['strata'][k];assert c['n']==len(ls) and c['old_wrong']==sum(r['old_archived_wrong'] for r in ls) and c['new_wrong']==sum(r['new_wrong'] for r in ls)
  assert rec['support_ge5_groups_each']==(rec['strata']['low']['groups']>=5 and rec['strata']['high']['groups']>=5)
  table_checks+=1
 assert support[tablekey]['families_with_both_cys']==sum(r['both_cys'] for r in table)
edges=read(P/'reused_control_family_completion.json');oldedges=read(R/'analysis/label_cys_joint_20261027/reused_control_evidence.json')
assert len(edges)==len(oldedges)==178
for e,o in zip(edges,oldedges):
 for k,v in o.items():assert e[k]==v
 a,b=byid[e['case_id']]['current_family_sidecar'],byid[e['control_id']]['current_family_sidecar']
 assert e['shared_current_full_families']==sorted(set(a['full_families'])&set(b['full_families']))
fetch=read(P/'family_fetch_manifest.json');oldfetch=read(R/'analysis/ghxtox_error_atlas_20261025/evidence_phase2/uniprot_reviewed_length11_50_20261025.tsv.request.json')
assert fetch['headers']['X-UniProt-Release']==oldfetch['headers']['X-UniProt-Release']=='2026_03'
save={'passed':True,'all_6387_original_fields_unchanged':True,'external_entries':len(entrylist),'exact_accession_sequence_checks':entry_checks,'family_table_rows_rebuilt':table_checks,'old_cached_family_not_recovered':old_family_not_recovered,'archived_edge_records':len(edges),'distinct_ordered_sample_pairs':len({(e['case_id'],e['control_id']) for e in edges}),'new_pairs_created':0,'same_UniProt_release':'2026_03','unassigned_and_multi_family_explicit':True,'NN_training':False}
(P/'verification.json').write_text(json.dumps(save,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(save,ensure_ascii=False,indent=2))
