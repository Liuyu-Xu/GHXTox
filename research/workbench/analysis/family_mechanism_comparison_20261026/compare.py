"""New family/sequence-matched comparison; reuse archived cohorts and features."""
import csv, hashlib, json, re, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
ROOT=Path('D:/dachuang2026/GHXTox'); OUT=Path(__file__).resolve().parent
P=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def save(n,x): (OUT/n).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(n,x):pd.DataFrame(x).to_csv(OUT/n,index=False,encoding='utf-8-sig')
atlaspath=ROOT/'analysis/expert_disagreement_20261025/training_expert_atlas.csv'
matchpath=ROOT/'analysis/ghxtox_error_atlas_20261025/evidence_phase2/training_uniprot_exact_matches.csv'
cachepath=Path('D:/dachuang2026/GHXTox-ziliao/train_chemical_sites_final_esm2_multisource_1_30_full2795.pt')
sourcepaths=[atlaspath,matchpath,OUT/'protocol.json',cachepath]
paths=list((ROOT/'analysis/ghxtox_error_atlas_20261025/evidence_phase2').glob('uniprot_error_records_batch*.json'))
paths += [ROOT/'analysis/expert_disagreement_20261025/case_annotations_new.json']
paths += list((ROOT/'analysis/assay_evidence_20261025/sources').glob('P*.json'))
paths=[p for p in paths if '.request.' not in p.name]
sourcepaths+=paths
paperpaths=[ROOT/'analysis/expert_disagreement_20261025/case_paper_abstracts_grouped.json',ROOT/'analysis/ghxtox_error_atlas_20261025/evidence_phase2/representative_paper_abstracts.json']
sourcepaths+=paperpaths
before={str(p):sha(p) for p in sourcepaths}
d=pd.read_csv(atlaspath,float_precision='round_trip'); assert len(d)==6387 and d.groupby('group_id').fold.nunique().max()==1
byid={r.sample_id:r for r in d.itertuples()}
matches=pd.read_csv(matchpath,keep_default_na=False); entries=defaultdict(list)
for _,r in matches.iterrows():
 if r['Entry']:
  assert r.sequence==byid[r.sample_id].sequence
  entries[r.sample_id].append(r.to_dict())
full={}; fullsources={}
for path in paths:
 j=read(path); records=j.get('results',[j]) if isinstance(j,dict) else j
 for rec in records:
  if isinstance(rec,dict) and rec.get('primaryAccession'):
   acc=rec['primaryAccession']
   if acc not in full or len(json.dumps(rec))>len(json.dumps(full[acc])):full[acc]=rec;fullsources[acc]=str(path)
papers={}
for path in paperpaths:
 for r in read(path).get('resultList',{}).get('result',[]):
  pmid=str(r.get('pmid',r.get('id','')))
  papers[pmid]=dict(pmid=pmid,title=r.get('title'),doi=r.get('doi'),abstract=r.get('abstractText'),cache=str(path),url='https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/')

def texts(rec,kind):return [t.get('value','') for c in rec.get('comments',[]) if c.get('commentType')==kind for t in c.get('texts',[])]
def location(f):return f.get('location',{}).get('start',{}).get('value'),f.get('location',{}).get('end',{}).get('value')
mentionrules={'membrane_or_lysis':r'hemol|haemol|cytol|pore.form|membrane.disrupt|membrane.permeabili',
 'ion_channel':r'ion.channel|potassium.channel|sodium.channel|calcium.channel|\bKir[0-9]|NMDA|nicotinic|GABA',
 'receptor_or_signaling':r'receptor|cGMP|cyclic GMP|growth.factor',
 'hemostasis_or_proteolysis':r'coagul|platelet|thrombin|proteol|fibrin|phospholipase',
 'antimicrobial':r'antibacter|antimicrob|antifungal|bactericid'}
meta={}
for r in d.itertuples():
 es=entries[r.sample_id]; names='; '.join(sorted(set(x['Protein names'] for x in es)))
 classes={k for k,expr in P['family_name_classes'].items() if re.search(expr,names,re.I) and not(k=='serine_protease' and re.search('inhibitor',names,re.I))}
 curated=[]; annotated=[]; pmids=set(); fs=[]; ownfull=[]; functiontexts=list(dict.fromkeys(x['Function [CC]'] for x in es if x['Function [CC]']))
 for x in es:
  rec=full.get(x['Entry'])
  if not rec:continue
  assert rec['sequence']['value']==r.sequence,(r.sample_id,x['Entry'])
  ownfull.append(x['Entry']);fs+=rec.get('features',[])
  for t in texts(rec,'SIMILARITY'):
   if re.search(r'belongs to .*family',t,re.I):curated.append(t.lower().strip().rstrip('.'))
  for ref in rec.get('references',[]):
   for cr in ref.get('citation',{}).get('citationCrossReferences',[]):
    if cr.get('database')=='PubMed':pmids.add(str(cr['id']))
  annotated.append(dict(accession=x['Entry'],cache=fullsources[x['Entry']],features=rec.get('features',[]),family=texts(rec,'SIMILARITY'),subunit=texts(rec,'SUBUNIT'),references=rec.get('references',[])))
 dis=[f for f in fs if f.get('type')=='Disulfide bond']; mods=[f for f in fs if f.get('type') in ['Modified residue','Cross-link']]
 intra=[]; inter=[]
 for f in dis:
  s,e=location(f)
  if re.search('interchain',f.get('description',''),re.I):inter.append(f);continue
  if isinstance(s,int) and isinstance(e,int) and 1<=s<e<=len(r.sequence) and r.sequence[s-1]==r.sequence[e-1]=='C':intra.append((s,e))
 intra=sorted(set(intra)); crossing=sum(s<u<e<v or u<s<v<e for i,(s,e) in enumerate(intra) for u,v in intra[i+1:])
 ftext=' '.join(functiontexts)
 mentions={k:[sentence.strip() for sentence in re.split(r'(?<=[.;])\s+',ftext) if re.search(expr,sentence,re.I)] for k,expr in mentionrules.items()}
 fragment=bool(r.any_fragment_match); boundary=any(f.get('type') in ['Chain','Peptide'] and any(f.get('location',{}).get(z,{}).get('modifier')=='OUTSIDE' for z in ['start','end']) for f in fs)
 meta[r.sample_id]=dict(sample_id=r.sample_id,accessions=[x['Entry'] for x in es],name=names,curated_family=sorted(set(curated)),name_class=sorted(classes),full_record_accessions=ownfull,
  fragment_reference_flag=fragment,chain_boundary_outside=boundary,subunit_name_lead=bool(re.search(r'subunit|\bchain\b',names,re.I)),chemical_feature_records=mods,
  annotated_intra_pairs=intra,annotated_interchain_records=inter,annotated_arc_crossings=crossing if intra else None,
  disulfide_evidence_status='explicit_annotation' if dis else 'no_feature_in_cached_full_record_unknown_actual_state' if ownfull else 'full_feature_record_unavailable',
  modification_evidence_status='explicit_annotation' if mods else 'no_feature_in_cached_full_record_unknown_actual_state' if ownfull else 'full_feature_record_unavailable',
  function_mentions=mentions,function_texts=functiontexts,annotation_negation_warning=bool(re.search(r'\bno\b|\bnot\b|lacks|without',ftext,re.I)),
  pmids=sorted(pmids),cached_papers=[papers[p] for p in sorted(pmids) if p in papers],full_record_details=annotated,
  source_urls=['https://www.uniprot.org/uniprotkb/'+x['Entry']+'/entry' for x in es])
save('all_training_annotation_index.json',meta)
if '--annotation-only' in __import__('sys').argv:
 print('Only annotation index rebuilt; no matching, features or tests rerun.',flush=True)
 raise SystemExit(0)
targets=d[d.expert_group=='structure_only_correct']; failures=targets[targets.fusion_error]; rescue=targets[~targets.fusion_error]
assert len(targets)==148 and len(failures)==50 and len(rescue)==98
aligner=PairwiseAligner(mode='global',match_score=2,mismatch_score=-1,open_gap_score=-3,extend_gap_score=-.5)
alignment_cache={}
def family_support(s,t):
 m,n=meta[s],meta[t];explicit=set(m['curated_family'])&set(n['curated_family']);proxy=set(m['name_class'])&set(n['name_class'])
 if explicit:return 'explicit_cached_family',sorted(explicit)
 if m['curated_family'] and n['curated_family']:return None,[]
 return ('name_class_proxy',sorted(proxy)) if proxy else (None,[])
def identity(s,t):
 key=tuple(sorted((s,t)))
 if key not in alignment_cache:
  a,b=byid[s].sequence,byid[t].sequence; al=aligner.align(a,b)[0]
  n=sum(sum(x==y for x,y in zip(a[i:j],b[k:l])) for (i,j),(k,l) in zip(*al.aligned))
  alignment_cache[key]=n/max(len(a),len(b))
 return alignment_cache[key]
pairs=[]; unmatched=[]; candidate_log=[]
for comp,pool in [('failure50_vs_rescue98',rescue),('failure50_vs_all_fusion_correct',d[~d.fusion_error])]:
 pools={}
 for r in failures.itertuples():
  same=pool[(pool.label==r.label)&(pool.fold==r.fold)&(pool.group_id!=r.group_id)]
  candidates=[]; family_n=0; length_n=0
  for c in same.itertuples():
   support,common=family_support(r.sample_id,c.sample_id)
   if not support:continue
   family_n+=1; lr=min(r.length,c.length)/max(r.length,c.length)
   if lr<.7:continue
   length_n+=1; ident=identity(r.sample_id,c.sample_id)
   if ident<.4:continue
   candidates.append(dict(comparison=comp,case_id=r.sample_id,control_id=c.sample_id,label=int(r.label),fold=int(r.fold),case_group=r.group_id,control_group=c.group_id,family_support=support,shared_family_or_class=';'.join(common),identity=ident,length_ratio=lr))
  pools[r.sample_id]=sorted(candidates,key=lambda x:(x['family_support']!='explicit_cached_family',-x['identity'],-x['length_ratio'],x['control_id']))
  candidate_log.append(dict(comparison=comp,sample_id=r.sample_id,same_label_fold_othergroup_n=len(same),family_compatible_n=family_n,length_compatible_n=length_n,sequence_compatible_n=len(candidates)))
 used=set()
 for sid in sorted(pools,key=lambda sid:(len(set(x['control_group'] for x in pools[sid])),sid)):
  r=byid[sid]; eligible=[x for x in pools[sid] if x['case_group'] not in used and x['control_group'] not in used]
  if eligible:
   q=eligible[0];pairs.append(q);used.update([q['case_group'],q['control_group']])
  else:unmatched.append(dict(comparison=comp,sample_id=sid,label=int(r.label),fold=int(r.fold),reason='no_family_sequence_comparable_control' if not pools[sid] else 'group_no_reuse_constraint'))
 print(comp,'matched',sum(x['comparison']==comp for x in pairs),flush=True)
csvout('matched_pairs.csv',pairs);csvout('unmatched_failures.csv',unmatched);csvout('matching_coverage.csv',candidate_log)
needed=set(targets.sample_id)|{x['control_id'] for x in pairs}

# Read only relevant SG coordinates from the same documented auxiliary structure cache.
import torch
payload=torch.load(cachepath,map_location='cpu',weights_only=False,mmap=True)
sg={}
for rec in payload['records']:
 sid=rec['sample_id']
 if sid not in needed:continue
 assert rec['sequence']==byid[sid].sequence and int(rec['label'])==byid[sid].label
 seq=rec['sequence'];coords=rec['chemical_site_coords'].tolist();mask=rec['chemical_site_mask'].tolist()
 ci=[i for i,x in enumerate(seq) if x=='C'];valid=[i for i in ci if mask[i][0]]
 distances=sorted((math.dist(coords[i][0],coords[j][0]),i+1,j+1) for k,i in enumerate(valid) for j in valid[k+1:])
 edges=[];used=set()
 for dist,i,j in distances:
  if dist<=2.5 and i not in used and j not in used:edges.append((i,j));used.update([i,j])
 sg[sid]=dict(cys_n=len(ci),sg_site_n=len(valid),sg_min_distance=distances[0][0] if distances else None,sg_close_pairs=edges,
  sg_close_edge_fraction=len(edges)/(len(ci)//2) if len(ci)>=2 and len(valid)==len(ci) else None,
  sg_proxy_arc_crossings=sum(i<k<j<l or k<i<l<j for n,(i,j) in enumerate(edges) for k,l in edges[n+1:]),
  sg_proxy_not_experimental_bond=True,auxiliary_cache_not_historical_OOF_checkpoint=True)
del payload
assert needed==set(sg)
save('sg_geometry_proxy.json',sg)

baseprotocol=read(ROOT/'analysis/ghxtox_error_atlas_20261025/protocol.json');w=baseprotocol['fusion_weight_sequence'];threshold=baseprotocol['fusion_threshold']
boundary=[]
for r in targets.itertuples():
 critical=(threshold-r.p_struct)/(r.p_seq-r.p_struct) if r.p_seq!=r.p_struct else None
 boundary.append(dict(sample_id=r.sample_id,label=int(r.label),fusion_error=bool(r.fusion_error),p_seq=r.p_seq,p_struct=r.p_struct,p_fusion=r.p_fusion,
  structure_correct_at_structure_threshold=True,structure_alone_correct_at_fusion_threshold=bool((r.p_struct>=threshold)==bool(r.label)),
  fusion_signed_margin=(2*r.label-1)*(r.p_fusion-threshold),weight_sequence_current=w,weight_boundary_algebra_only=critical))
 assert abs(r.p_fusion-(w*r.p_seq+(1-w)*r.p_struct))<1e-8
csvout('fusion_boundary_diagnostics.csv',boundary)

rng=np.random.default_rng(P['seed']);tests=[];pairfeatures=[]
features=P['paired_features']; other=['length','cys_count','net_charge_proxy','mean_hydropathy','mean_plddt','sequence_seed_std','p_seq','p_struct','p_fusion']
def value(sid,f):
 if f=='sg_close_edge_fraction':return sg[sid][f]
 v=getattr(byid[sid],f);return float(v) if pd.notna(v) else None
for pair in pairs:
 x=dict(pair)
 for f in features+other:x['case_'+f]=value(pair['case_id'],f);x['control_'+f]=value(pair['control_id'],f)
 pairfeatures.append(x)
csvout('matched_pair_features.csv',pairfeatures)
for comp in P['comparisons']:
 for label in [0,1]:
  subset=[x for x in pairfeatures if x['comparison']==comp and x['label']==label]
  for f in features:
   sub=[x for x in subset if x['case_'+f] is not None and x['control_'+f] is not None]
   diff=np.array([x['case_'+f]-x['control_'+f] for x in sub],dtype=float);n=len(diff)
   pval=1.;ci=[None,None]
   if n>=8:
    boot=diff[rng.integers(0,n,size=(2000,n))].mean(axis=1);ci=np.quantile(boot,[.025,.975]).tolist()
    perms=(rng.choice([-1,1],size=(5000,n))*diff).mean(axis=1);pval=(1+int((abs(perms)>=abs(diff.mean())-1e-14).sum()))/5001
   folds={fold:float(np.mean([x['case_'+f]-x['control_'+f] for x in sub if x['fold']==fold])) for fold in range(5) if any(x['fold']==fold for x in sub)}
   dd=[x['case_'+f]-x['control_'+f] for x in sub if x['fold'] in [0,1,2]];rr=[x['case_'+f]-x['control_'+f] for x in sub if x['fold'] in [3,4]]
   tests.append(dict(comparison=comp,label=label,feature=f,n_pairs=n,status='testable_exploratory' if n>=8 else 'insufficient_pairs',mean_case_minus_control=float(diff.mean()) if n else None,ci_low=ci[0],ci_high=ci[1],p=pval,fold_differences=folds,discovery_mean=float(np.mean(dd)) if dd else None,internal_check_mean=float(np.mean(rr)) if rr else None,same_split_direction=bool(np.mean(dd)*np.mean(rr)>0) if dd and rr else None))
order=sorted(range(len(tests)),key=lambda i:tests[i]['p']);last=1.
for rank in range(len(order),0,-1):
 ix=order[rank-1];last=min(last,tests[ix]['p']*len(tests)/rank);tests[ix]['q_BH32']=last
save('paired_feature_tests.json',tests);csvout('paired_feature_tests.csv',tests)

familyrows=[]
for group,frame in [('failure50',failures),('rescue98',rescue)]:
 for label in [0,1]:
  ids=list(frame[frame.label==label].sample_id)
  fams=sorted({c for sid in ids for c in meta[sid]['name_class']})+['unclassified_name']
  for family in fams:
   block=[s for s in ids if family in meta[s]['name_class'] or (family=='unclassified_name' and not meta[s]['name_class'])]
   if not block:continue
   familyrows.append(dict(outcome_group=group,label=label,name_class=family,n=len(block),sample_ids=';'.join(block),fragment_reference_n=sum(meta[s]['fragment_reference_flag'] for s in block),full_feature_record_n=sum(bool(meta[s]['full_record_accessions']) for s in block),explicit_disulfide_annotation_n=sum(meta[s]['disulfide_evidence_status']=='explicit_annotation' for s in block),explicit_chemical_annotation_n=sum(meta[s]['modification_evidence_status']=='explicit_annotation' for s in block),membrane_mention_n=sum(bool(meta[s]['function_mentions']['membrane_or_lysis']) for s in block),channel_mention_n=sum(bool(meta[s]['function_mentions']['ion_channel']) for s in block),cached_paper_n=len({p['pmid'] for s in block for p in meta[s]['cached_papers']})))
csvout('whole_class_annotation_summary.csv',familyrows)
save('target148_mechanism_evidence.json',[dict(**meta[sid],existing_label=int(byid[sid].label),existing_fold=int(byid[sid].fold),existing_fusion_error=bool(byid[sid].fusion_error),sg_geometry=sg[sid]) for sid in targets.sample_id])
save('matched_control_mechanism_evidence.json',[meta[s] for s in sorted({x['control_id'] for x in pairs})])
coverage={}
for group,frame in [('failure50',failures),('rescue98',rescue)]:
 ids=list(frame.sample_id);coverage[group]=dict(n=len(ids),identity_recovered_n=sum(bool(meta[s]['accessions']) for s in ids),full_feature_record_n=sum(bool(meta[s]['full_record_accessions']) for s in ids),explicit_family_n=sum(bool(meta[s]['curated_family']) for s in ids),name_class_n=sum(bool(meta[s]['name_class']) for s in ids),cached_abstract_sample_n=sum(bool(meta[s]['cached_papers']) for s in ids))
summary=dict(target148=148,failure50=50,rescue98=98,coverage=coverage,
 comparisons={comp:dict(pairs=sum(x['comparison']==comp for x in pairs),explicit_family_pairs=sum(x['comparison']==comp and x['family_support']=='explicit_cached_family' for x in pairs),name_proxy_pairs=sum(x['comparison']==comp and x['family_support']=='name_class_proxy' for x in pairs),identity_ge60_pairs=sum(x['comparison']==comp and x['identity']>=.6 for x in pairs),unmatched_failures=sum(x['comparison']==comp for x in unmatched)) for comp in P['comparisons']},
 testable_tests=sum(x['status']=='testable_exploratory' for x in tests),planned_tests=len(tests),q_lt05=[x for x in tests if x['q_BH32']<.05],
 failure_structure_not_correct_at_fusion_threshold=[r['sample_id'] for r in boundary if r['fusion_error'] and not r['structure_alone_correct_at_fusion_threshold']],
 limitations=P['restrictions'])
save('summary.json',summary)
for comp in P['comparisons']:
 pp=[x for x in pairs if x['comparison']==comp];groups=[v for x in pp for v in [x['case_group'],x['control_group']]];assert len(groups)==len(set(groups))
 for x in pp:
  assert byid[x['case_id']].label==byid[x['control_id']].label and byid[x['case_id']].fold==byid[x['control_id']].fold
after={str(p):sha(p) for p in sourcepaths};assert before==after
save('verification.json',dict(input_hashes_before=before,input_hashes_after=after,inputs_unchanged=True,old_classification_and_probabilities_reused=True,no_old_matching_rerun=True,matched_labels_folds_family_and_sequence_verified=True,no_group_reuse_within_contrast=True,SG_inputs_match_sequences=True,no_test_inputs=True,no_training=True,all148_retained=True))
print(json.dumps(clean(summary),ensure_ascii=False,indent=2),flush=True)
