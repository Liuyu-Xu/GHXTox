"""Parent audit: source/endpoint integrity and cross-direction duplication only."""
import json,csv,hashlib
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
R=Path('D:/dachuang2026/GHXTox');O=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
base=R/'analysis/environment_feasibility_20261025';blocks_path=R/'analysis/label_ambiguity_map_20261025/public_terminal_pair_blocks.json'
protected=[base/'model_records.json',base/'sources/Hemolytik2_complete_data.csv',blocks_path]
before={str(p):sha(p) for p in protected}
blocks=read(blocks_path)
with (base/'sources/Hemolytik2_complete_data.csv').open(encoding='utf-8-sig',newline='') as f:raw={r['id']:r for r in csv.DictReader(f)}
block_checks=[];unique={}
for b in blocks:
 rs=b['records'];assert len(rs)==2
 a,c=rs
 assert a['sequence']==c['sequence']==b['sequence']
 assert str(a['pmid'])==str(c['pmid'])==str(b['pmid'])
 assert a['species']==c['species']==b['species'] and a['dose_uM']==c['dose_uM']==b['dose_uM']
 changed=int(a['c_terminal']!=c['c_terminal'])+int(a['n_terminal']!=c['n_terminal']);assert changed==1
 smiles=[raw[x['id']]['smiles'] for x in rs]
 assert all(x.strip() for x in smiles)
 check=dict(record_ids=[x['id'] for x in rs],sequence=b['sequence'],pmid=str(b['pmid']),species=b['species'],dose_uM=b['dose_uM'],same_nonempty_SMILES=smiles[0]==smiles[1],exactly_one_terminal_field_changed=True,reported_pH=b.get('pH'),pH_unknown_not_proof_of_matching=b.get('pH') is None)
 block_checks.append(check)
 key=(b['sequence'],str(b['pmid']),b['species'],tuple(sorted(x['chemical_key'] for x in rs)))
 unique.setdefault(key,[]).append(check)
chem=dict(database_blocks=len(blocks),distinct_backbones=len({b['sequence'] for b in blocks}),distinct_pmid_identifiers=len({str(b['pmid']) for b in blocks}),distinct_state_pairs=len(unique),all_same_nonempty_SMILES=sum(x['same_nonempty_SMILES'] for x in block_checks),single_terminal_field_difference_blocks=sum(x['exactly_one_terminal_field_changed'] for x in block_checks),scope='parent verified metadata only; no effect-direction ranking or molecular design; subagent was interrupted by safety review')
save('chemistry_metadata_parent_verification.json',dict(summary=chem,blocks=block_checks))
sel=R/'analysis/hemolysis_selectivity_audit_20261026';dose=R/'analysis/hemolysis_dose_audit_20261026'
ss=read(sel/'summary.json');ds=read(dose/'summary.json');records=read(dose/'confirmed_non_point_records.json')
xml=ET.parse(dose/'PMID34361810_fulltext.xml').getroot()
table=next(t for t in xml.iter('table-wrap') if t.get('id')=='molecules-26-04654-t004')
rows=[' '.join(' '.join(e.itertext()).split()) for e in table.iter('tr')]
SI=next(r for r in rows if r.startswith('SI '));HC=next(r for r in rows if r.startswith('HC '))
assert all(v in SI for v in ['16.77','33.26','0.99'])
assert all(v in HC for v in ['886.50','67.09','199.20','5.60'])
direct_rows={}
for tr in table.iter('tr'):
 cells=[' '.join(' '.join(cell.itertext()).split()) for cell in list(tr) if cell.tag in ['td','th']]
 if len(cells)!=5:continue
 try:
  dose_value=float(cells[0]);values=[float(x.replace('*','').strip()) for x in cells[1:]]
 except ValueError:continue
 direct_rows[dose_value]=values
checked_blocks=0
for b in blocks:
 if str(b['pmid'])!='34361810':continue
 assert b['dose_uM'] in direct_rows
 for r in b['records']:
  column=0 if r['c_terminal']=='acid' else 1
  assert r['hemolysis_percent']==direct_rows[b['dose_uM']][column]
 checked_blocks+=1
assert checked_blocks==9
strict={r['id'] for r in read(base/'model_records.json')}
assert sum(x['classification']=='confirmed wrong-column SI conversion' for x in records)==3
assert sum(x['id'] in strict for x in records)==3
assert sum(x['id'] in strict and x['actual_endpoint']=='SI' for x in records)==2
save('endpoint_table_parent_check.json',dict(source_xml=str(dose/'PMID34361810_fulltext.xml'),source_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC8348142/#molecules-26-04654-t004',SI_row=SI,HC50_row=HC,raw_database_wrong_SI_rows=3,wrong_SI_rows_in_existing3380=2,reported_HC50_derived_rows_in_existing3380=1,chemistry_direct_percentage_blocks_verified=checked_blocks,original_data_unchanged=True))
compounds=read(sel/'compounds.json');panels=read(dose/'confirmed_non_point_exclusion_panels.json')
dose3=[p for p in panels if p['n_doses']>=3]
sets={'selectivity22':{r['sequence'] for r in compounds},'dose269':{r['sequence'] for r in dose3},'chemistry7':{b['sequence'] for b in blocks}}
overlap={a+'__'+b:len(sets[a]&sets[b]) for i,a in enumerate(sets) for b in list(sets)[i+1:]}
oldseq={r['sequence'] for r in read(base/'model_records.json')}
overlap['selectivity_sequences_seen_in_old_environment_data']=len(sets['selectivity22']&oldseq)
overlap['chemistry_sequences_seen_in_old_environment_data']=len(sets['chemistry7']&oldseq)
save('cross_direction_overlap.json',dict(sequence_overlaps=overlap,warning='shared record sources and previous exploratory use prohibit treating these as independent blind validations; sequence overlap is not an effect comparison'))
summary=dict(selectivity=dict(original_studies=ss['original_articles_recovered'],named_series=3,sequences=ss['unique_compounds'],strain_association_rows=ss['paired_mic_strain_rows'],direct_HC50=ss['exact_HC50_compounds'],direct_threshold_MHC5=ss['threshold_MHC5_compounds'],not_ready_for_multistudy_training=True),
 dose_confirmed_overlay=ds['source_endpoint_audit']['sensitivity']['confirmed_non_point_exclusion'],dose_conservative_sensitivity=ds['source_endpoint_audit']['sensitivity']['all_exact50_quarantine'],
 chemistry_metadata=chem,subagent_status={'selectivity_audit':'completed','dose_response_audit':'completed','chemical_state_audit':'interrupted by system safety review; parent completed metadata checks only'},
 model_requests={'subagents':{'model':'gpt-6.1-sol','reasoning_effort':'medium'},'main':'main model/effort controlled by current conversation settings; no tool switch available'},
 decision='dose-response is the most data-ready bounded risk-prediction pilot after endpoint provenance audit; selectivity has a within-study diagnostic panel but insufficient coherent multi-study data; chemical-state panel is a small metadata/input-consistency test, not new-model training evidence')
save('summary.json',summary)
after={str(p):sha(p) for p in protected};assert before==after
for source in read(dose/'provenance.json')['inputs']:assert sha(source['path'])==source['sha256']
save('verification.json',dict(protected_input_hashes_before=before,protected_input_hashes_after=after,inputs_unchanged=True,dose_agent_input_hashes_verified=True,three_database_SI_errors_checked_against_original_table=True,terminal_pair_metadata_and_nonempty_SMILES_checked=True,no_model_training=True,no_test_predictions=True,chemistry_analysis_scope_limited_to_metadata=True))
print(json.dumps(dict(chemistry=chem,overlaps=overlap,primary_source_table_verified=True),ensure_ascii=False,indent=2))
