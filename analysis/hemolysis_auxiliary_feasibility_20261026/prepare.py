"""Published auxiliary labels, source coverage and leakage-safe sequence manifests.
No model fitting. Protected data read for sequence identity only.
"""
from pathlib import Path
import csv,json,re,hashlib,collections,time
from Bio.Align import PairwiseAligner
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
S=R/'analysis/dataset_comparison_20261025/sources'
paths=dict(cv=S/'raghavagps__HemoPI2/Dataset/cross_val_dataset.csv',independent=S/'raghavagps__HemoPI2/Dataset/independent_dataset.csv',raw=R/'analysis/environment_feasibility_20261025/sources/Hemolytik2_complete_data.csv',main=R/'analysis/error_evidence_linkage_20261025/all_training_linked.json',ann=R/'analysis/family_mechanism_comparison_20261026/all_training_annotation_index.json',toxipep=S/'GGCL7__ToxiPep/Dataset/Independent set/independent test set.txt',toxmamba=S/'Marscolono__ToxMamba/data/Toxi_data/independent_test.csv',test1=S/'jiaoshihu__CAPTP/data/test1.txt',test2=S/'jiaoshihu__CAPTP/data/test2.txt')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()}
def csvread(p,sep=','):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline=''),delimiter=sep))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def fasta_sequences(p):
 seq=[];out=[]
 for line in p.read_text(encoding='utf-8-sig').splitlines():
  if line.startswith('>'):
   if seq:out.append(''.join(seq));seq=[]
  elif line.strip():seq.append(line.strip())
 if seq:out.append(''.join(seq))
 return out
main=read(paths['main']);ann=read(paths['ann']);assert len(main)==6387
cv=csvread(paths['cv']);protected={}
for k in ['test1','test2','toxipep']:protected[k]=sorted(set(fasta_sequences(paths[k])))
# Only read independent sequence fields, never outcomes or model predictions.
protected['hemopi_independent']=sorted({r['SEQUENCE'] for r in csvread(paths['independent'])})
protected['toxmamba']=sorted({r['seq'] for r in csvread(paths['toxmamba'],'\t')})
raw=csvread(paths['raw']);rawseq=collections.defaultdict(list)
for r in raw:rawseq[r['seq']].append(r)
duplicates=collections.defaultdict(list)
for i,r in enumerate(cv):
 seq=r['SEQUENCE'];hc=float(r['μM']);label=int(r['label']);assert label==int(hc<=100) and hc>0
 duplicates[seq].append(dict(row_index=i,sequence=seq,hc50_uM=hc,label=label))
rejected=[];aux=[]
for seq,rs in duplicates.items():
 reason=None
 if not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{11,50}',seq):reason='outside_fixed_standard20_length11_50'
 elif len({r['label'] for r in rs})>1:reason='source_label_conflict_same_sequence'
 if reason:rejected.append(dict(sequence=seq,reason=reason,source_rows=rs));continue
 matches=rawseq.get(seq,[])
 # An identity link does not establish that any particular measurement is the CV HC50.
 pmids=sorted({r['pmid'].strip() for r in matches if r['pmid'].strip().isdigit()})
 chem=sorted({(r['cter'],r['nter'],r['ldmix'],r['lyn_cyc'],r['non_nat']) for r in matches})
 aux.append(dict(aux_id=f'HCV_{rs[0]["row_index"]+1:04d}',**rs[0],source_rows=[r['row_index'] for r in rs],raw_identity_record_ids=[r['id'] for r in matches],candidate_pmid_links=pmids,chemical_states_at_same_sequence=chem,raw_hc50_record_ids=[r['id'] for r in matches if re.search(r'HC\s*\(?50',r['activity'],re.I)],evidence_tier='published_source_HC50_label_not_assay_gold',negative_meaning='HC50_above100uM_not_no_toxicity',molecule_state_confirmed_for_source_row=False))
# Existing main sequence groups are retained and unioned when auxiliary data bridge them.
seqs=sorted({r['sequence'] for r in main}|{r['sequence'] for r in aux}|{s for v in protected.values() for s in v})
index={s:i for i,s in enumerate(seqs)};parent=list(range(len(seqs)))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(i,j):
 a,b=find(i),find(j)
 if a!=b:parent[max(a,b)]=min(a,b)
bygroup=collections.defaultdict(list)
for r in main:bygroup[r['group_id']].append(index[r['sequence']])
for ids in bygroup.values():
 for i in ids[1:]:union(ids[0],i)
aligner=PairwiseAligner(mode='global',match_score=0,mismatch_score=-1,open_gap_score=-1,extend_gap_score=-1)
assert -aligner.score('AAAA','AAAT')==1 and -aligner.score('AAAA','AAA')==1
edges=[];start=time.time()
for i,r in enumerate(aux):
 s=r['sequence'];si=index[s]
 for j,t in enumerate(seqs):
  if si==j:continue
  cutoff=int(.2*max(len(s),len(t))+1e-9)
  if abs(len(s)-len(t))>cutoff:continue
  distance=int(round(-aligner.score(s,t)))
  if distance<=cutoff:union(si,j);edges.append(dict(aux_id=r['aux_id'],other_sequence=t,distance=distance,limit=cutoff))
 if (i+1)%250==0:print(f'Checked auxiliary near-neighbours {i+1}/{len(aux)}',flush=True)
component={s:find(index[s]) for s in seqs}
protected_components={component[s] for v in protected.values() for s in v}
for r in aux:
 r['joint_component']=component[r['sequence']];r['reserved_component_overlap']=r['joint_component'] in protected_components
 for k,v in protected.items():r[k+'_exact_overlap']=r['sequence'] in set(v)
pmidmain={}
for r in main:
 a=ann[r['sample_id']]
 pmidmain[r['sample_id']]=sorted(set(a.get('pmids',[]))|set(re.findall(r'PubMed:(\d+)',' '.join(a.get('function_texts',[])))))
folds=[]
for f in range(5):
 test=[r for r in main if int(r['fold'])==f];testcomponents={component[r['sequence']] for r in test}
 # Available annotation citations are conservative purge links, not complete source provenance.
 knownpmids={p for r in test for p in pmidmain[r['sample_id']]}
 train=[r for r in main if int(r['fold'])!=f and component[r['sequence']] not in testcomponents]
 eligible=[r for r in aux if not r['reserved_component_overlap'] and r['joint_component'] not in testcomponents]
 paperfiltered=[r for r in eligible if not set(r['candidate_pmid_links'])&knownpmids]
 assert not {component[r['sequence']] for r in train}&testcomponents
 assert not {r['joint_component'] for r in paperfiltered}&testcomponents
 folds.append(dict(fold=f,test_main_ids=[r['sample_id'] for r in test],train_main_ids=[r['sample_id'] for r in train],aux_train_ids=[r['aux_id'] for r in paperfiltered],original_main_train_n=sum(int(r['fold'])!=f for r in main),main_train_n=len(train),main_test_n=len(test),aux_sequence_safe_n=len(eligible),aux_after_available_citation_purge_n=len(paperfiltered),aux_label_counts=dict(collections.Counter(r['label'] for r in paperfiltered)),aux_with_candidate_pmid_n=sum(bool(r['candidate_pmid_links']) for r in paperfiltered),known_citation_links_purged=len(eligible)-len(paperfiltered),unknown_primary_assay_sources_remain=True,paper_disjointness_certified=False))
summary=dict(source_cv_rows=len(cv),cv_unique_sequences=len(duplicates),standard_scope_aux=len(aux),label_counts=dict(collections.Counter(r['label'] for r in aux)),candidate_identity_linked_n=sum(bool(r['raw_identity_record_ids']) for r in aux),candidate_hc50_linked_n=sum(bool(r['raw_hc50_record_ids']) for r in aux),reserved_sequences={k:len(v) for k,v in protected.items()},reserved_component_overlap_n=sum(r['reserved_component_overlap'] for r in aux),sequence_safe_after_reserved_n=sum(not r['reserved_component_overlap'] for r in aux),joint_components=len(set(component.values())),alignment_seconds=time.time()-start,near_rule='unit-cost global edit distance <=floor(0.2*maxlen), with existing main groups unioned and transitive component purge; not standard CD-HIT identity',fold_counts=[{k:v for k,v in x.items() if not k.endswith('_ids')} for x in folds],no_auxiliary_independent_rows_used=True,strict_assay_gold=False)
for name,obj in [('auxiliary_source_records.json',aux),('rejected_scope_records.json',rejected),('protected_sequence_manifest.json',protected),('near_neighbor_edges.json',edges),('joint_sequence_components.json',component),('fold_manifests.json',folds),('summary.json',summary)]:save(name,obj)
assert all(sha(p)==before[k]['sha256'] for k,p in paths.items())
save('verification.json',dict(inputs=before,inputs_unchanged=True,source_threshold_checked_all1540=True,independent_outcomes_not_selected_or_scored=True,transitive_sequence_components_disjoint_per_fold=True,paper_disjointness_certified=False,model_fits=0))
print(json.dumps(summary,ensure_ascii=False,indent=2))
