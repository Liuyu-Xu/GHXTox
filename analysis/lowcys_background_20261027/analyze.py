from pathlib import Path
import json,csv,re,collections,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
tsv=P/'uniprot_reviewed_7_50_family.tsv'
entries=list(csv.DictReader(tsv.open(encoding='utf-8-sig'),delimiter='\t'))
snapshot=collections.defaultdict(list)
for e in entries:snapshot[e['Sequence']].append(e)
old=read(R/'analysis/label_cys_joint_20261027/all_sample_evidence_join.json')
rules=read(R/'analysis/family_mechanism_comparison_20261026/protocol.json')['family_name_classes']
def parse(raw):
 clean=re.sub(r'\{[^}]*\}', '',raw)
 comments=[re.sub(r'\s+',' ',s).strip(' .;').lower() for s in re.split(r'SIMILARITY:\s*',clean,flags=re.I) if s.strip(' .;')]
 full=[s for s in comments if re.search(r'belongs to .*family',s)]
 parent=[s.split('.')[0].strip() for s in full]
 return sorted(set(full)),sorted(set(parent))

# Family fields are built solely from sequence and external records, before joining outcomes.
sidecar=[]
for s in old:
 matches=snapshot.get(s['sequence'],[]);full=set();parent=set();entryrows=[]
 for e in matches:
  fs,ps=parse(e['Sequence similarities']);full.update(fs);parent.update(ps)
  entryrows.append({'accession':e['Entry'],'entry_name':e['Entry Name'],'organism':e['Organism'],'protein_name':e['Protein names'],'raw_similarity':e['Sequence similarities'],'full_family':fs,'parent_family':ps})
 names='; '.join(sorted(set(e['Protein names'] for e in matches)))
 proxy=sorted(k for k,expr in rules.items() if re.search(expr,names,re.I) and not(k=='serine_protease' and re.search('inhibitor',names,re.I)))
 sidecar.append({'sample_id':s['sample_id'],'sequence':s['sequence'],'full_families':sorted(full),'parent_families':sorted(parent),'name_proxies':proxy,'exact_entries':entryrows,'match_status':'no_exact_entry' if not matches else 'matched_no_family_assignment' if not full else 'multiple_full_families' if len(full)>1 else 'single_full_family_among_annotated_matches','annotated_entry_count':sum(bool(x['full_family']) for x in entryrows),'unannotated_entry_count':sum(not x['full_family'] for x in entryrows)})
save('family_sidecar_before_outcomes.json',sidecar)
joined=[]
for s,f in zip(old,sidecar):
 assert s['sample_id']==f['sample_id'] and s['sequence']==f['sequence']
 joined.append({**s,'current_family_sidecar':f})
save('all_samples_with_symmetric_family.json',joined)
def counts(rs):
 return {'n':len(rs),'groups':len(set(r['group_id'] for r in rs)),'low_cys':sum(r['cys_count']<4 for r in rs),'high_cys':sum(r['cys_count']>=4 for r in rs),'flagged':sum(r['uniform_flag'] for r in rs),'old_wrong':sum(r['old_archived_wrong'] for r in rs),'new_wrong':sum(r['new_wrong'] for r in rs),'both_wrong':sum(r['old_archived_wrong'] and r['new_wrong'] for r in rs),'both_right':sum(not r['old_archived_wrong'] and not r['new_wrong'] for r in rs),'sample_ids':[r['sample_id'] for r in rs]}
def family_table(layer,label):
 buckets=collections.defaultdict(list)
 for r in joined:
  if r['label']!=label:continue
  values=r['current_family_sidecar'][layer]
  # Disjoint membership is required in the primary family tables.
  if len(values)==1:buckets[values[0]].append(r)
 out=[]
 for fam,rs in sorted(buckets.items()):
  low=[r for r in rs if r['cys_count']<4];high=[r for r in rs if r['cys_count']>=4]
  strata={'low':counts(low),'high':counts(high),'low_unflagged':counts([r for r in low if not r['uniform_flag']]),'high_unflagged':counts([r for r in high if not r['uniform_flag']])}
  out.append({'family':fam,'layer':layer,'label':label,'all':counts(rs),'strata':strata,'both_cys':bool(low and high),'support_ge5_groups_each':len(set(r['group_id'] for r in low))>=5 and len(set(r['group_id'] for r in high))>=5,'low_has_both_old_outcomes':any(r['old_archived_wrong'] for r in low) and any(not r['old_archived_wrong'] for r in low)})
 return out
tables={f'{layer}_label{label}':family_table(layer,label) for layer in ['full_families','parent_families','name_proxies'] for label in [1,0]}
save('all_family_background_tables.json',tables)
coverage={}
populations={'all':joined,'label1': [r for r in joined if r['label']==1],'low_cys_positive':[r for r in joined if 'physical_0' in r['category_keys']],'low_cys_shared_FN':[r for r in joined if 'physical_0' in r['category_keys'] and r['old_archived_wrong'] and r['new_wrong']],'low_cys_both_correct':[r for r in joined if 'physical_0' in r['category_keys'] and not r['old_archived_wrong'] and not r['new_wrong']]}
for key,rs in populations.items():
 coverage[key]={**counts(rs),'prior_curated_family':sum(bool(r['curated_family']) for r in rs),'current_any_full_family':sum(bool(r['current_family_sidecar']['full_families']) for r in rs),'current_single_full_family':sum(len(r['current_family_sidecar']['full_families'])==1 for r in rs),'current_single_parent_family':sum(len(r['current_family_sidecar']['parent_families'])==1 for r in rs),'status_counts':dict(collections.Counter(r['current_family_sidecar']['match_status'] for r in rs))}
support={}
for key,table in tables.items():
 both=[x for x in table if x['both_cys']];adequate=[x for x in table if x['support_ge5_groups_each']]
 support[key]={'families_total':len(table),'sample_n':sum(x['all']['n'] for x in table),'families_with_both_cys':len(both),'n_in_both_cys_families':sum(x['all']['n'] for x in both),'adequate_families':len(adequate),'adequate_families_names':[x['family'] for x in adequate],'both_cys_rows':both}
save('coverage_and_common_support.json',{'coverage':coverage,'support':support})

# Describe every positive low-Cys family, not only families with interesting outcomes.
lowtables=[]
for layer in ['full_families','parent_families']:
 for rec in tables[layer+'_label1']:
  if rec['strata']['low']['n']>0:lowtables.append(rec)
save('lowcys_positive_whole_family_panels.json',lowtables)

# Annotate archived178 edges only. Do not claim restored annotation establishes 178 homologous pairs.
oldedges=read(R/'analysis/label_cys_joint_20261027/reused_control_evidence.json')
meta={r['sample_id']:r for r in joined};edgeout=[]
for e in oldedges:
 a,b=meta[e['case_id']],meta[e['control_id']];af,bf=a['current_family_sidecar'],b['current_family_sidecar']
 edgeout.append({**e,'shared_current_full_families':sorted(set(af['full_families'])&set(bf['full_families'])),'shared_current_parent_families':sorted(set(af['parent_families'])&set(bf['parent_families'])),'case_current_match_status':af['match_status'],'control_current_match_status':bf['match_status'],'old_family_support_preserved':e['family_support']})
save('reused_control_family_completion.json',edgeout)

# Population-matched coverage prevents claiming the unannotated low-Cys group does not belong to families.
save('input_manifest.json',{'inputs':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [tsv,R/'analysis/label_cys_joint_20261027/all_sample_evidence_join.json',R/'analysis/family_mechanism_comparison_20261026/protocol.json',R/'analysis/label_cys_joint_20261027/reused_control_evidence.json']},'no_original_label_or_flags_changed':True,'strict_first_eligibility_preserved':True})
print('COVERAGE',json.dumps({k:{a:b for a,b in v.items() if a!='sample_ids'} for k,v in coverage.items()},ensure_ascii=False),flush=True)
print('SUPPORT',{k:{a:b for a,b in v.items() if a!='both_cys_rows'} for k,v in support.items()},flush=True)
print('BOTH_CYS_POSITIVE',json.dumps(support['full_families_label1']['both_cys_rows'],ensure_ascii=False)[:5000],flush=True)
