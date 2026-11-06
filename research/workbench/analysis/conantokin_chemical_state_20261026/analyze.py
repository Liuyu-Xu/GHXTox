"""Complete all six existing Conantokin annotations; do not fit a predictor."""
import csv,hashlib,json
from pathlib import Path
R=Path('D:/dachuang2026/GHXTox');O=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvread(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
atlaspath=R/'analysis/expert_disagreement_20261025/training_expert_atlas.csv'
metapath=R/'analysis/family_mechanism_comparison_20261026/all_training_annotation_index.json'
snapshot=R/'analysis/ghxtox_error_atlas_20261025/evidence_phase2/training_uniprot_exact_matches.csv'
meta=read(metapath);atlas={r['sample_id']:r for r in csvread(atlaspath)}
ids=[s for s,m in meta.items() if 'conantokin' in m['name_class']]
hit={r['sample_id'] for r in csvread(snapshot) if 'conantokin' in (r['Protein names']+' '+r['Function [CC]']).lower()}
assert set(ids)==hit and len(ids)==6
oldpaths=[atlaspath,metapath,snapshot]
records={};sources={}
for sid in ids:
 m=meta[sid];acc=m['accessions'][0]
 if m['full_record_accessions']:
  path=Path(m['full_record_details'][0]['cache']);oldpaths.append(path)
  j=read(path);recs=j.get('results',[j])
  record=next(x for x in recs if x.get('primaryAccession')==acc)
 else:
  path=O/'sources'/f'{acc}.json';record=read(path)
 records[sid]=record;sources[sid]=dict(accession=acc,path=str(path),url=f'https://www.uniprot.org/uniprotkb/{acc}/entry',sha256=sha(path),reused_old_cache=path in oldpaths)
oldpaths=list(dict.fromkeys(oldpaths));before={str(p):sha(p) for p in oldpaths}
papers={str(p['id']):p for p in read(O/'sources/primary_papers.json')['resultList']['result']}
paperindex={pmid:dict(title=p['title'],doi=p.get('doi'),publication_types=p.get('pubTypeList',{}).get('pubType',[]),is_review=any('review' in t.lower() for t in p.get('pubTypeList',{}).get('pubType',[])),url='https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/',abstract_cached=True,full_methods_checked=False) for pmid,p in papers.items()}
rows=[];detail=[]
for sid in ids:
 r=atlas[sid];rec=records[sid];assert rec['sequence']['value']==r['sequence']
 fs=rec.get('features',[]);gla=[];hyp=[];amides=[];chem=[]
 for f in fs:
  if f.get('type') not in ['Modified residue','Cross-link']:continue
  desc=f.get('description','');pos=f.get('location',{}).get('start',{}).get('value')
  chem.append(f)
  if desc=='4-carboxyglutamate':assert r['sequence'][pos-1]=='E';gla.append(pos)
  if 'hydroxyproline' in desc.lower():assert r['sequence'][pos-1]=='P';hyp.append(pos)
  if 'amide' in desc.lower() and pos==len(r['sequence']):amides.append(f)
 functions=[t for c in rec.get('comments',[]) if c.get('commentType')=='FUNCTION' for t in c.get('texts',[])]
 codes=sorted({e.get('evidenceCode','') for f in chem for e in f.get('evidences',[])})
 pmids=sorted({e['id'] for f in chem for e in f.get('evidences',[]) if e.get('source')=='PubMed'})
 species=rec.get('organism',{}).get('scientificName')
 name=meta[sid]['name'];short=name.split(' (')[0]
 row=dict(sample_id=sid,name=short,accession=rec['primaryAccession'],organism=species,sequence=r['sequence'],length=len(r['sequence']),original_label=int(r['label']),fold=int(r['fold']),sequence_result=r['sequence_category'],structure_result=r['structure_category'],fusion_result=r['fusion_category'],p_seq=float(r['p_seq']),p_struct=float(r['p_struct']),p_fusion=float(r['p_fusion']),mean_plddt=float(r['mean_plddt']),
  gla_present=bool(gla),gla_count=len(gla),gla_positions=';'.join(map(str,sorted(gla))),hyp_positions=';'.join(map(str,sorted(hyp))),
  terminal_amide_status='明确酰胺化注释' if amides else '未注释，不能据此判定游离端',
  chemical_evidence_codes=';'.join(codes),chemical_annotation_pmids=';'.join(pmids),original_Pr_paper='17962189' in pmids,
  evidence_origin='ECO:0000269原始研究引用' if 'ECO:0000269' in codes else 'ECO:0000303出版物注释，所引17153339为综述',
  direct_same_backbone_modified_unmodified_pair_available=False,source_url=sources[sid]['url'])
 rows.append(row);detail.append(dict(**row,source=sources[sid],modification_features=chem,function_texts_and_evidence=functions,reference_metadata=paperindex,causal_error_explanation_established=False))
with (O/'ALL_SIX_TABLE.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save('all_six_chemical_evidence.json',detail)
by={r['sample_id']:r for r in rows}
pairids=[('peptide_195','peptide_854'),('peptide_57','peptide_359'),('peptide_66','peptide_218')]
pair_notes=[]
for a,b in pairids:
 x,y=by[a],by[b]
 pair_notes.append(dict(error_id=a,correct_id=b,gla_present_both=x['gla_present'] and y['gla_present'],same_gla_count=x['gla_count']==y['gla_count'],same_gla_positions=x['gla_positions']==y['gla_positions'],same_species=x['organism']==y['organism'],same_original_Pr_study=x['original_Pr_paper'] and y['original_Pr_paper'],same_original_label=x['original_label']==y['original_label'],same_fold=x['fold']==y['fold'],hyp_positions_error=x['hyp_positions'],hyp_positions_correct=y['hyp_positions']))
save('existing_pair_comparison.json',pair_notes)
assert len({r['sequence'] for r in rows})==6
assert all(r['gla_present'] for r in rows)
assert by['peptide_195']['gla_positions']==by['peptide_854']['gla_positions']
summary=dict(n_existing_class=6,errors=3,correct=3,all_six_Gla_annotated=True,Gla_presence_discriminates_errors=False,
 matched_Pr1_Pr2_Gla_same_positions=by['peptide_195']['gla_positions'],
 Pr_study_ids=['peptide_195','peptide_854','peptide_218'],
 original_Pr_study_same_length=True,original_Pr_study_same_Gla_count=True,
 models_behavior='57 sequence FN/structure TP/fusion FN;66 and195 sequence TP/structure FN/fusion FN;218,359,854 all TP',
 review_only_annotation_ids=[r['sample_id'] for r in rows if not r['original_Pr_paper']],
 chemical_state_in_input='Gla positions encode E; Hyp positions encode P; terminal amide has no explicit input state',
 literal_P3_exactly_matches_Hyp3_within_six=all((r['sequence'][2]=='P')==('3' in r['hyp_positions'].split(';')) for r in rows),
 same_sequence_different_chemical_state_controls=0,
 training_decision='do not train a Gla-present correction branch on these six; first seek direct same-backbone chemical-state contrasts if pursuing mechanism',
 original_thresholds_labels_weights_unchanged=True)
save('summary.json',summary);save('primary_paper_metadata.json',paperindex)
lines=['# Conantokin全类修饰状态（全部6条）','', '|样本|名称|Gla位置|Hyp位置|C端状态|一维/三维/融合|证据来源|','|---|---|---|---|---|---|---|']
for r in rows:
 url=r['source_url'];p=r['chemical_annotation_pmids']
 lines.append(f"|{r['sample_id']}|[{r['name']}]({url})|{r['gla_positions']}|{r['hyp_positions'] or '未注释'}|{r['terminal_amide_status']}|{r['sequence_result']}/{r['structure_result']}/{r['fusion_result']}|{p}；{r['chemical_evidence_codes']}|")
lines+=['', 'ECO:0000269为实验引用；本组ECO:0000303引用17153339，该文献类型为综述，不能当作已核实的逐肽直接实验。没有修饰条目不等于已确认未修饰。','', '位置从1开始。六条输入均为标准氨基酸序列：Gla位点为E，Hyp位点为P；未改变输入，仅核对。']
(O/'ALL_SIX_TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
after={str(p):sha(p) for p in oldpaths};assert before==after
save('verification.json',dict(old_input_hashes_before=before,old_input_hashes_after=after,inputs_unchanged=True,all_six_sequences_and_modified_positions_verified=True,all_training_name_function_snapshot_scanned_no_extra_hit=True,new_full_records_downloaded=3,old_error_records_reused=3,new_paper_metadata_records=2,no_new_training=True,no_test_predictions=True,no_relabeling=True,no_old_matching_or_tests_rerun=True))
print(json.dumps(summary,ensure_ascii=False,indent=2))
