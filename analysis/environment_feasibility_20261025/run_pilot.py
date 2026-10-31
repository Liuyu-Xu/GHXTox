"""Pure-computational public-data pilot: fixed-dose hemolysis, no GHXTox retraining.
Run from any directory. Source-only feature engineering; group validation.
"""
from pathlib import Path
import csv,json,re,hashlib,time,collections,itertools
import numpy as np
from Bio.Align import PairwiseAligner
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,r2_score

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
RAW=OUT/'sources/Hemolytik2_complete_data.csv'
ATLAS=ROOT/'analysis/expert_disagreement_20261025/training_expert_atlas.csv'
before=hashlib.sha256(ATLAS.read_bytes()).hexdigest()
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def log(s):print(s,flush=True)
with RAW.open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
num=r'\d+(?:\.\d+)?'
pattern=re.compile(r'^\s*(?P<rel>[<>~≈≤≥]?)\s*(?P<y>'+num+r')(?:\s*±\s*'+num+r')?\s*%\s*H(?:a?emolysis|emolytic)\s+at\s*(?P<drel>[<>~≈≤≥]?)\s*(?P<dose>'+num+r')\s*(?P<unit>[µμu]M|mM)\s*(?:\((?P<suffix>[^)]*)\))?\s*$',re.I)
species_alias={'mouse':'mouse','mus musculus':'mouse','swiss mouse':'mouse','porcine':'pig','cattle':'bovine','ovine':'sheep','canine':'dog'}
ct_alias={'free':'acid','oh':'acid','amidation':'amide'};nt_alias={'free':'free','h':'free','acetylation':'acetyl'}
rejected=collections.Counter(); candidates=[]
for r in raw:
 m=pattern.fullmatch(r['activity'])
 if not m:rejected['not_strict_molar_percent_expression']+=1;continue
 if m['rel'] or m['drel']:rejected['inequality_or_approximate_point']+=1;continue
 if not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{5,50}',r['seq']):rejected['sequence_out_of_scope']+=1;continue
 if r['ldmix']!='L' or r['lyn_cyc']!='Linear' or r['non_nat']!='None':rejected['chemical_state_out_of_scope']+=1;continue
 # Avoid hidden cyclization or residue modification despite inconsistent coarse columns.
 tags=re.findall(r'\{([^}]+)\}',r['MAP'])
 if any(not t.startswith(('ct:','nt:')) for t in tags):rejected['other_MAP_modifications']+=1;continue
 if r['cter'].lower() not in ct_alias or r['nter'].lower() not in nt_alias:rejected['terminal_state_out_of_scope']+=1;continue
 y=float(m['y']);dose=float(m['dose'])*(1000 if m['unit']=='mM' else 1)
 if not (0<=y<=100 and dose>0):rejected['invalid_numeric_range']+=1;continue
 sp=r['source'].strip().lower();sp=species_alias.get(sp,sp)
 if sp in ['','na','unknown']:rejected['missing_assay_species']+=1;continue
 pmid=r['pmid'].strip()
 if not pmid.isdigit():rejected['missing_primary_identifier']+=1;continue
 phm=re.search(r'\bpH\s*('+num+')',m['suffix'] or '',re.I); ph=float(phm[1]) if phm else None
 if ph is not None and not 0<ph<14:rejected['invalid_pH']+=1;continue
 cterm=ct_alias[r['cter'].lower()];nterm=nt_alias[r['nter'].lower()]
 chem='|'.join([r['seq'],cterm,nterm,'linear','L'])
 candidates.append(dict(id=r['id'],pmid=pmid,sequence=r['seq'],name=r['name'],MAP=r['MAP'],chemical_key=chem,c_terminal=cterm,n_terminal=nterm,species=sp,dose_uM=dose,pH=ph,hemolysis_percent=y,raw_activity=r['activity'],origin=r['origin']))
save('parsed_records.json',candidates)
# Deduplicate identical reported measurements; exclude conflicting reported-condition groups.
condition=lambda r:(r['pmid'],r['chemical_key'],r['species'],r['dose_uM'],r['pH'])
buckets=collections.defaultdict(list)
for r in candidates:buckets[condition(r)].append(r)
model_rows=[];conflicts=[];duplicate_rows=0
for k,rs in buckets.items():
 ys={r['hemolysis_percent'] for r in rs}
 if len(ys)>1:conflicts.append(dict(key=k,ids=[r['id'] for r in rs],values=sorted(ys)));continue
 r=rs[0].copy();r['supporting_database_ids']=[v['id'] for v in rs];model_rows.append(r);duplicate_rows+=len(rs)-1
save('conflicting_recorded_conditions.json',conflicts);save('model_records.json',model_rows)
sequences=sorted({r['sequence'] for r in model_rows});index={s:i for i,s in enumerate(sequences)}
parent=list(range(len(sequences)))
def find(a):
 while a!=parent[a]:parent[a]=parent[parent[a]];a=parent[a]
 return a
def union(a,b):
 a,b=find(a),find(b)
 if a!=b:parent[b]=a
aligner=PairwiseAligner(mode='global',match_score=0,mismatch_score=-1,open_gap_score=-1,extend_gap_score=-1)
assert aligner.score('AAAA','AAAT')==-1 and aligner.score('AAAA','AAA')==-1
start=time.time();edges=0
for i,s in enumerate(sequences):
 for j in range(i):
  t=sequences[j];mx=max(len(s),len(t));limit=int(.2*mx+1e-9)
  if abs(len(s)-len(t))>limit:continue
  # Unit-cost global edit distance, normalized by max length; not a CD-HIT identity claim.
  if -aligner.score(s,t)<=limit+1e-9:union(i,j);edges+=1
groups=np.array([find(index[r['sequence']]) for r in model_rows])
save('sequence_clusters.json',[{'sequence':s,'group':find(i)} for i,s in enumerate(sequences)])
log(f'Prepared {len(model_rows)} records; {len(sequences)} sequences; {len(set(groups))} edit-distance groups; clustering {time.time()-start:.1f}s')

# Coverage and cross-context panels: no prediction-based selection.
chem_groups=collections.defaultdict(list)
for r in model_rows:chem_groups[r['chemical_key']].append(r)
cross_species=[(k,rs) for k,rs in chem_groups.items() if len({r['species'] for r in rs})>1]
matched_buckets=collections.defaultdict(list)
for r in model_rows:matched_buckets[(r['chemical_key'],r['pmid'],r['dose_uM'],r['pH'])].append(r)
paired=[{'chemical_key':k[0],'pmid':k[1],'dose_uM':k[2],'pH':k[3],'records':rs,'missing_controls':['time','temperature','RBC_density','buffer']} for k,rs in matched_buckets.items() if len({r['species'] for r in rs})>1]
save('same_paper_dose_cross_species.json',paired)
with ATLAS.open(encoding='utf-8-sig',newline='') as f: atlas={r['sequence']:r for r in csv.DictReader(f)}
known=collections.defaultdict(set)
with (ROOT/'analysis/dataset_comparison_20261025/all_records.csv').open(encoding='utf-8-sig',newline='') as f:
 for r in csv.DictReader(f):
  if r['dataset']=='GHXTox':known[r['sequence']].add(r['split'])
links=[]
for k,rs in chem_groups.items():
 s=rs[0]['sequence'];a=atlas.get(s)
 if a:links.append({'sample_id':a['sample_id'],'sequence':s,'chemical_key':k,'original_label':a['label'],'original_fusion_category':a['fusion_category'],'records':len(rs),'species':sorted({r['species'] for r in rs}),'pmids':sorted({r['pmid'] for r in rs}),'note':'原广义毒性标签；数据库分子形式与GHXTox输入仍需对应，不重新赋溶血TP/FN'})
save('ghxtox_training_links.json',links)
coverage={'raw_database_rows':len(raw),'raw_unique_backbones':len({r['seq'] for r in raw}),'available_columns':list(raw[0]),'missing_time_column':True,'missing_temperature_column':True,'parsed_in_scope_rows':len(candidates),'filter_exclusions':dict(rejected),'duplicate_rows_collapsed':duplicate_rows,'conflicting_recorded_condition_groups_excluded':len(conflicts),'model_rows':len(model_rows),'unique_backbones':len(sequences),'unique_chemical_forms':len(chem_groups),'paper_count':len({r['pmid'] for r in model_rows}),'edit_distance_groups':len(set(groups)),'species_counts':dict(collections.Counter(r['species'] for r in model_rows)),'pH_available_rows':sum(r['pH'] is not None for r in model_rows),'chemical_forms_cross_species':len(cross_species),'same_paper_same_dose_cross_species_blocks':len(paired),'same_paper_same_dose_cross_species_forms':len({p['chemical_key'] for p in paired}),'source_verified_against_primary_papers':False,'known_train_backbones':sum('train' in known[s] for s in sequences),'known_historical_test_backbones':sum(any(v!='train' for v in known[s]) for s in sequences),'chemical_family_labels':'未提供统一家族标签；origin自由文本不能当可靠家族数；序列群数另列','note':'数据库记录可用于探索，尚非逐条原文核实实验。跨物种块仍缺时间/温度等控制信息。'}
save('coverage.json',coverage)

# Fixed feature sets and hyperparameters; no tuning against held-out labels.
AA='ACDEFGHIKLMNPQRSTVWY';hydro=dict(zip(AA,[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
di=list(itertools.product(AA,repeat=2));sp_values=sorted({r['species'] for r in model_rows})
def peptide_features(r):
 s=r['sequence'];n=len(s);c=collections.Counter(s);d=collections.Counter(zip(s[:-1],s[1:]));h=np.array([hydro[a] for a in s]);theta=np.arange(n)*np.deg2rad(100)
 return [c[a]/n for a in AA]+[d[a]/max(1,n-1) for a in di]+[n,(c['K']+c['R']-c['D']-c['E'])/n,float(h.mean()),float(np.hypot((h*np.cos(theta)).sum(),(h*np.sin(theta)).sum())/n),r['c_terminal']=='amide',r['n_terminal']=='acetyl']
S=np.asarray([peptide_features(r) for r in model_rows],dtype=float);D=np.array([[np.log10(r['dose_uM'])] for r in model_rows]);E=np.array([[r['species']==sp for sp in sp_values] for r in model_rows],dtype=float)
features={'peptide':S,'peptide_dose':np.hstack([S,D]),'peptide_dose_species':np.hstack([S,D,E]),'dose_species_only':np.hstack([D,E])}
y=np.array([r['hemolysis_percent'] for r in model_rows]);pmids=np.array([r['pmid'] for r in model_rows]);chemkeys=np.array([r['chemical_key'] for r in model_rows])
def weights(ix):
 c=collections.Counter(chemkeys[ix]);return np.array([1/c[k] for k in chemkeys[ix]])
def metrics(ix,pred):return {'mae_percentage_points':float(mean_absolute_error(y[ix],pred,sample_weight=weights(ix))),'r2':float(r2_score(y[ix],pred,sample_weight=weights(ix)))}
results=[];predictions=[]
for design in ['sequence_edit_groups','paper_holdout_purged_sequence_groups']:
 splitter=GroupKFold(n_splits=5);splitgroups=groups if design=='sequence_edit_groups' else pmids
 for fold,(tr,te) in enumerate(splitter.split(S,y,splitgroups)):
  n_before=len(tr)
  if design.startswith('paper_'):tr=tr[~np.isin(groups[tr],np.unique(groups[te]))]
  assert not set(groups[tr]) & set(groups[te])
  if design.startswith('paper_'):assert not set(pmids[tr]) & set(pmids[te])
  if len(tr)<80:log(f'SKIP {design} fold {fold}: {len(tr)} rows');continue
  for name,X in features.items():
   model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=4,max_features=.8,n_jobs=2,random_state=2026)
   model.fit(X[tr],y[tr],sample_weight=weights(tr));pred=np.clip(model.predict(X[te]),0,100)
   results.append(dict(design=design,fold=fold,model=name,train_rows=len(tr),test_rows=len(te),train_rows_purged=n_before-len(tr),unseen_test_species=sorted(set(model_rows[i]['species'] for i in te)-set(model_rows[i]['species'] for i in tr)),**metrics(te,pred)))
   predictions.extend(dict(design=design,fold=fold,model=name,row_index=int(i),observed=float(y[i]),predicted=float(v),chemical_key=model_rows[i]['chemical_key']) for i,v in zip(te,pred))
  log(f'{design} fold {fold} done; train={len(tr)} test={len(te)} purge={n_before-len(tr)}')
  save('fold_metrics.json',results)
save('pilot_predictions.json',predictions)
summary=[]
for design in sorted({r['design'] for r in results}):
 for name in features:
  ps=[p for p in predictions if p['design']==design and p['model']==name];ix=np.array([p['row_index'] for p in ps]);pred=np.array([p['predicted'] for p in ps]);summary.append(dict(design=design,model=name,folds=len({p['fold'] for p in ps}),**metrics(ix,pred)))
save('pilot_summary.json',summary)
# Paired bootstrap of held-out errors by sequence cluster, fixed model comparisons.
rng=np.random.default_rng(2026);cis=[]
for design in sorted({r['design'] for r in results}):
 bymodel={name:{p['row_index']:p for p in predictions if p['design']==design and p['model']==name} for name in features}
 for base,aug in [('peptide','peptide_dose'),('peptide_dose','peptide_dose_species')]:
  ix=np.array(sorted(set(bymodel[base]) & set(bymodel[aug])));w=weights(ix)
  delta=np.array([abs(y[i]-bymodel[base][i]['predicted'])-abs(y[i]-bymodel[aug][i]['predicted']) for i in ix]);g=groups[ix];uniq=np.unique(g)
  num=np.array([np.sum(delta[g==v]*w[g==v]) for v in uniq]);den=np.array([np.sum(w[g==v]) for v in uniq]);boot=[]
  for _ in range(2000):b=rng.integers(len(uniq),size=len(uniq));boot.append(num[b].sum()/den[b].sum())
  cis.append({'design':design,'baseline':base,'augmented':aug,'MAE_reduction_percentage_points':float(np.sum(delta*w)/w.sum()),'paired_cluster_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),'scope':'冻结OOF预测的序列群重采样；不覆盖所有模型训练/原始实验不确定性'})
save('paired_comparisons.json',cis)
after=hashlib.sha256(ATLAS.read_bytes()).hexdigest();assert before==after
save('verification.json',dict(source_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),atlas_before=before,atlas_after=after,same_sequence_groups_disjoint=True,paper_holdout_sequence_purged=True,test_predictions_used=False,ghxtox_weights_changed=False,target='固定浓度下溶血百分比',HC50_not_input=True,censored_points_not_coerced=True,hyperparameter_search=False))
save('protocol.json',dict(target='hemolysis percentage at reported exact molar concentration',molecular_scope='5–50 standard-AA, L, linear, no other MAP tags, free/acetyl N, acid/amide C',environment_scope='dose and reported red-blood-cell species; no time/temperature model; pH collected but not fitted',model='ExtraTreesRegressor 160 trees min_samples_leaf4 max_features0.8 random_state2026',group_rule='transitive clusters with unit-cost global edit distance <= floor(0.2*max sequence length); not standard identity/CD-HIT',evaluation='5fold edit clusters; separately5fold paper holdout with all shared sequence clusters purged; per chemical-form weights',limitations=['database metadata may contain errors','missing time/temperature/buffer/RBC density','natural species proportions unbalanced','source PMID may reference reviews','no claim of independent GHXTox benchmark or mechanistic causality']))
log(json.dumps(summary,ensure_ascii=False));log(json.dumps(cis,ensure_ascii=False))
