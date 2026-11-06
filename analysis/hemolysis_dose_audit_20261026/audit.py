"""Audit existing strict fixed-dose observations; no fitting or prediction access.
Run: PYTHONIOENCODING=utf-8 D:/anaconda/envs/pytorch_env/python.exe analysis/hemolysis_dose_audit_20261026/audit.py
"""
from pathlib import Path
from collections import defaultdict, Counter
import json,csv,math,re,hashlib,statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'analysis/environment_feasibility_20261025'
def read(name): return json.loads((SRC/name).read_text(encoding='utf-8'))
def save(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
rows=read('model_records.json'); parsed=read('parsed_records.json')
clusters={r['sequence']:r['group'] for r in read('sequence_clusters.json')}
def key(r):return (r['sequence'],r['chemical_key'],r['pmid'],r['species'],r['pH'])
def dosekey(r):return key(r)+(r['dose_uM'],)
pb=defaultdict(list)
for r in parsed:pb[dosekey(r)].append(r)
repeats=[];conflicts=[]
for k,rr in pb.items():
 if len(rr)>1:
  v={'key':k,'ids':[r['id'] for r in rr],'responses':sorted(set(r['hemolysis_percent'] for r in rr)),'row_count':len(rr)}
  (repeats if len(v['responses'])==1 else conflicts).append(v)
conflict_keys={tuple(r['key'][:-1]) for r in conflicts}
b=defaultdict(list)
for r in rows:b[key(r)].append(r)
panels=[]
for i,(k,rr) in enumerate(sorted(b.items(),key=lambda kv:str(kv[0])),1):
 rr=sorted(rr,key=lambda r:r['dose_uM']); ds=[r['dose_uM'] for r in rr]; ys=[r['hemolysis_percent'] for r in rr]
 assert len(set(ds))==len(ds)
 drops=[{'low_dose_uM':ds[j],'high_dose_uM':ds[j+1],'drop_pp':ys[j]-ys[j+1]} for j in range(len(rr)-1) if ys[j]>ys[j+1]]
 pairs=[ys[a]-ys[c] for a in range(len(rr)) for c in range(a+1,len(rr)) if ys[a]>ys[c]]
 panels.append(dict(panel_id=f'P{i:04d}',sequence=k[0],chemical_key=k[1],pmid=k[2],species=k[3],pH=k[4],edit_cluster=clusters[k[0]],n_doses=len(ds),min_dose_uM=min(ds),max_dose_uM=max(ds),dose_fold_span=max(ds)/min(ds),log10_dose_span=math.log10(max(ds)/min(ds)),response_min=min(ys),response_max=max(ys),response_range_pp=max(ys)-min(ys),endpoint_change_pp=ys[-1]-ys[0],all_zero=all(y==0 for y in ys),all_hundred=all(y==100 for y in ys),constant_response=len(set(ys))==1,adjacent_drop_count=len(drops),max_adjacent_drop_pp=max([d['drop_pp'] for d in drops],default=0),max_pairwise_drop_pp=max(pairs,default=0),adjacent_drops=drops,source_has_excluded_same_dose_conflict=k in conflict_keys,points=[{a:r.get(a) for a in ['id','supporting_database_ids','dose_uM','hemolysis_percent','raw_activity']} for r in rr]))
def coverage(pp):return dict(panels=len(pp),rows=sum(p['n_doses'] for p in pp),unique_sequences=len({p['sequence'] for p in pp}),chemical_forms=len({p['chemical_key'] for p in pp}),pmids=len({p['pmid'] for p in pp}),edit_clusters=len({p['edit_cluster'] for p in pp}))
def quant(v):
 v=sorted(v)
 if not v:return None
 def q(f):
  a=(len(v)-1)*f;l=int(a);return v[l]+(v[min(l+1,len(v)-1)]-v[l])*(a-l)
 return dict(min=v[0],q25=q(.25),median=q(.5),q75=q(.75),max=v[-1])
pp=[p for p in panels if p['n_doses']>=3]
threshold={str(n):coverage([p for p in panels if p['n_doses']>=n]) for n in [2,3,4,5]}
metrics=dict(dose_fold_span=quant([p['dose_fold_span'] for p in pp]),response_range_pp=quant([p['response_range_pp'] for p in pp]),all_zero=sum(p['all_zero'] for p in pp),all_hundred=sum(p['all_hundred'] for p in pp),constant_response=sum(p['constant_response'] for p in pp),nonmonotone_any=sum(p['max_pairwise_drop_pp']>0 for p in pp),drop_gt5pp=sum(p['max_pairwise_drop_pp']>5 for p in pp),drop_gt10pp=sum(p['max_pairwise_drop_pp']>10 for p in pp),excluded_same_dose_conflict_panels=sum(p['source_has_excluded_same_dose_conflict'] for p in pp),reported_pH_panels=sum(p['pH'] is not None for p in pp),fold_span_ge10=sum(p['dose_fold_span']>=10 for p in pp),range_ge20pp=sum(p['response_range_pp']>=20 for p in pp),cross_50percent=sum(p['response_min']<50<p['response_max'] for p in pp))
# Context differences are preserved; do not pool across papers, RBC source, or pH.
bychem=defaultdict(list)
for p in panels:bychem[p['chemical_key']].append(p)
contexts=[]
for chem,cc in bychem.items():
 if len(cc)>1:contexts.append(dict(chemical_key=chem,panel_ids=[p['panel_id'] for p in cc],pmids=sorted({p['pmid'] for p in cc}),species=sorted({p['species'] for p in cc}),pH=sorted({str(p['pH']) for p in cc})))
# Describe only raw activity categories, never convert HC50 or censored values to points.
with (SRC/'sources/Hemolytik2_complete_data.csv').open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
raw_categories=Counter();raw_desc=[]
for r in raw:
 a=r['activity']; hc=bool(re.search(r'\b(?:HC|LC|HD|EC|IC)\s*50\b',a,re.I)); pct='%' in a;bound=bool(re.search('[<>≤≥~≈]',a))
 cat='HC50_or_other_50_endpoint' if hc else ('percent_with_bound_or_approx' if pct and bound else 'percent_other_expression' if pct else 'other')
 raw_categories[cat]+=1
 raw_desc.append(dict(id=r['id'],activity=a,category=cat,has_percent=pct,has_bound_or_approx=bound,has_50_endpoint=hc))
# Five folds by whole PMID, greedily balanced on panel count; purge every shared edit cluster.
# This is a descriptive data availability simulation, not CV/model fitting.
pmid_panels=defaultdict(list)
for p in pp:pmid_panels[p['pmid']].append(p)
folds=[[] for _ in range(5)]
for pm,pr in sorted(pmid_panels.items(),key=lambda kv:(-len(kv[1]),kv[0])):
 f=min(range(5),key=lambda j:sum(len(pmid_panels[x]) for x in folds[j]));folds[f].append(pm)
split=[]
for f,ids in enumerate(folds):
 te=[p for p in pp if p['pmid'] in ids];train0=[p for p in pp if p['pmid'] not in ids];seen={p['edit_cluster'] for p in te};tr=[p for p in train0 if p['edit_cluster'] not in seen]
 split.append(dict(fold=f,test_pmids=ids,test=coverage(te),train_before_purge=coverage(train0),train_after_purge=coverage(tr),purged_panels=len(train0)-len(tr)))
pmcounts=Counter(p['pmid'] for p in pp);clcounts=Counter(p['edit_cluster'] for p in pp)
def kish(c):return sum(c.values())**2/sum(v*v for v in c.values()) if c else 0
clean=[p for p in pp if p['dose_fold_span']>=10 and p['response_range_pp']>=20 and p['max_pairwise_drop_pp']<=5 and not p['source_has_excluded_same_dose_conflict']]
confirmed={'11995':('HC50','886.50','HC50 886.50 uM, not an independently measured 50-percent dose point'),'12005':('SI','16.77','SI 16.77 is dimensionless; true HC50 is 67.09 uM'),'12015':('SI','33.26','SI 33.26 is dimensionless; true HC50 is 199.20 uM; this D-peptide row is outside strict scope'),'12025':('SI','0.99','SI 0.99 is dimensionless; true HC50 is 5.60 uM')}
confirmed_table=[]
raw_by_id={r['id']:r for r in raw}
strict_ids={r['id'] for r in rows}
for ident,(endpoint,value,note) in confirmed.items():
 r=raw_by_id[ident];confirmed_table.append(dict(id=ident,pmid=r['pmid'],name=r['name'],sequence=r['seq'],MAP=r['MAP'],cter=r['cter'],nter=r['nter'],ldmix=r['ldmix'],raw_activity=r['activity'],actual_endpoint=endpoint,actual_value=value,actual_unit='dimensionless' if endpoint=='SI' else 'uM',classification='confirmed wrong-column SI conversion' if endpoint=='SI' else 'valid HC50 endpoint converted to 50-percent pseudo-point; not an independent observation',note=note,in_strict_3380=ident in strict_ids,source='https://pmc.ncbi.nlm.nih.gov/articles/PMC8348142/#molecules-26-04654-t004'))
def sensitivity(label,exclude):
 result=[]
 for p in panels:
  q=p.copy();pts=[t for t in p['points'] if not exclude(t)]
  if not pts:continue
  q['points']=pts;q['n_doses']=len(pts);ds=[t['dose_uM'] for t in pts];ys=[t['hemolysis_percent'] for t in pts]
  q['dose_fold_span']=max(ds)/min(ds);q['response_range_pp']=max(ys)-min(ys);q['max_pairwise_drop_pp']=max([ys[i]-ys[j] for i in range(len(ys)) for j in range(i+1,len(ys))],default=0)
  result.append(q)
 three=[p for p in result if p['n_doses']>=3]
 save(label+'_panels.json',[dict(panel_id=p['panel_id'],sequence=p['sequence'],chemical_key=p['chemical_key'],pmid=p['pmid'],species=p['species'],pH=p['pH'],n_doses=p['n_doses'],points=p['points']) for p in result])
 return dict(remaining_rows=sum(p['n_doses'] for p in result),thresholds={str(n):coverage([p for p in result if p['n_doses']>=n]) for n in [2,3,4,5]},three_plus_dose_fold_span=quant([p['dose_fold_span'] for p in three]),three_plus_response_range_pp=quant([p['response_range_pp'] for p in three]),three_plus_nonmonotone_any=sum(p['max_pairwise_drop_pp']>0 for p in three),three_plus_drop_gt5pp=sum(p['max_pairwise_drop_pp']>5 for p in three))
sensitivity_results=dict(confirmed_non_point_exclusion=sensitivity('confirmed_non_point_exclusion',lambda p:p['id'] in confirmed),all_exact50_quarantine=sensitivity('all_exact50_quarantine',lambda p:p['hemolysis_percent']==50))
save('confirmed_non_point_records.json',confirmed_table)
summary=dict(scope='existing strict 3380 model records only; raw activity lexical categories separately',baseline=coverage(panels),thresholds=threshold,n_doses_histogram=dict(sorted(Counter(p['n_doses'] for p in panels).items())),parsed_reconciliation=dict(parsed_rows=len(parsed),same_value_duplicate_groups=len(repeats),same_value_duplicate_extra_rows=sum(v['row_count']-1 for v in repeats),conflicting_groups=len(conflicts),conflicting_rows=sum(v['row_count'] for v in conflicts),strict_rows=len(rows)),three_plus=metrics,context_differences=dict(multi_context_forms=len(contexts),multiple_pmids=sum(len(x['pmids'])>1 for x in contexts),multiple_species=sum(len(x['species'])>1 for x in contexts),multiple_pH_including_unknown=sum(len(x['pH'])>1 for x in contexts)),raw_activity_lexical_only=dict(rows=len(raw),categories=dict(raw_categories),note='mutually exclusive lexical descriptors; not a new parsed eligible dataset; 50-endpoint category includes LC/HD/EC/IC50'),concentration_of_three_plus=dict(top_pmids=pmcounts.most_common(10),largest_paper_share=max(pmcounts.values(),default=0)/max(len(pp),1),paper_count_kish=kish(pmcounts),edit_cluster_count_kish=kish(clcounts)),descriptive_fivefold_pmid_cluster_purge=split,exploratory_quality_screen=dict(rule='>=3 doses, >=10fold span, >=20pp range, max all-pair decrease <=5pp, no excluded same-dose conflict; descriptive only, not prevalidated inclusion criteria',**coverage(clean)),limitations=['Missing pH is grouped as unknown, not evidence of identical pH','No incubation time/temperature/RBC density/buffer fields; panels are metadata groups, not established complete experimental curves','Sequence uniqueness is not biological independence; reused edit clusters are approximate relatedness groups, not validated families','No fit/training/GHXTox test predictions used'])
summary['source_endpoint_audit']=dict(confirmed_raw_records=len(confirmed_table),confirmed_raw_SI_column_errors=3,valid_raw_HC50_derived_points=1,confirmed_records_in_strict=sum(r['in_strict_3380'] for r in confirmed_table),strict_SI_column_errors=2,strict_HC50_derived_points=1,exact50_rows_in_strict=sum(r['hemolysis_percent']==50 for r in rows),exact50_scope='50% is a review flag, not proof of HC50; quarantine is intentionally conservative and may exclude real measurements',sensitivity=sensitivity_results,source='https://pmc.ncbi.nlm.nih.gov/articles/PMC8348142/#molecules-26-04654-t004',retrieval='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8348142/fullTextXML',cache='PMID34361810_fulltext.xml')
save('summary.json',summary);save('panels.json',panels);save('same_dose_duplicates.json',repeats);save('same_dose_conflicts.json',conflicts);save('different_contexts.json',contexts);save('raw_activity_categories.json',raw_desc)
fields=[k for k in panels[0] if k not in ['points','adjacent_drops']]
with (OUT/'panels.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:p[k] for k in fields} for p in panels)
inputs=['model_records.json','parsed_records.json','sequence_clusters.json','protocol.json','sources/Hemolytik2_complete_data.csv']
save('provenance.json',dict(inputs=[dict(path=str(SRC/n),sha256=hashlib.sha256((SRC/n).read_bytes()).hexdigest()) for n in inputs],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_run=False,predictions_read=False))
print(json.dumps(summary,ensure_ascii=False,indent=2))
