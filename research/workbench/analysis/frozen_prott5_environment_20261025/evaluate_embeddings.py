"""Keep prior pilot records, folds, head settings and chemistry unchanged.
Replace handcrafted sequence features by frozen 1024d ProtT5 residue means.
"""
from pathlib import Path
import json,hashlib,collections,time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,r2_score

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BASE=ROOT/'analysis/environment_feasibility_20261025'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
input_paths=[BASE/n for n in ['model_records.json','sequence_clusters.json','pilot_predictions.json']];before={str(p):sha(p) for p in input_paths}
r=read(BASE/'model_records.json');cl={v['sequence']:v['group'] for v in read(BASE/'sequence_clusters.json')};previous=read(BASE/'pilot_predictions.json')
z=np.load(OUT/'prott5_mean_embeddings.npz');emb={s:v for s,v in zip(z['sequences'].tolist(),z['embeddings'])};assert len(emb)==1792
S=np.stack([np.concatenate([emb[a['sequence']],[a['c_terminal']=='amide',a['n_terminal']=='acetyl']]) for a in r]);D=np.array([[np.log10(a['dose_uM'])] for a in r]);species=sorted({a['species'] for a in r});E=np.array([[a['species']==s for s in species] for a in r],dtype=float)
features={'peptide':S,'peptide_dose':np.hstack([S,D]),'peptide_dose_species':np.hstack([S,D,E])};y=np.array([a['hemolysis_percent'] for a in r]);keys=np.array([a['chemical_key'] for a in r]);groups=np.array([cl[a['sequence']] for a in r]);papers=np.array([a['pmid'] for a in r])
def weights(ix):
 c=collections.Counter(keys[ix]);return np.array([1/c[k] for k in keys[ix]])
def metrics(ix,p):return {'mae_percentage_points':float(mean_absolute_error(y[ix],p,sample_weight=weights(ix))),'r2':float(r2_score(y[ix],p,sample_weight=weights(ix)))}
# Exactly identical condition-only model has already been evaluated; reuse its OOF outputs.
predictions=[v.copy() for v in previous if v['model']=='dose_species_only'];results=[];folds=[]
progress=OUT/'fold_metrics.json'
done=set()
cache=OUT/'partial_predictions.json'
if cache.exists():
 predictions=read(cache);results=read(progress);done={(v['design'],v['fold'],v['model']) for v in results}
for design in ['sequence_edit_groups','paper_holdout_purged_sequence_groups']:
 oldfold={(v['fold'],v['row_index']) for v in previous if v['design']==design and v['model']=='peptide'}
 for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(S,groups=groups if design=='sequence_edit_groups' else papers)):
  n_before=len(tr)
  if design.startswith('paper_'):tr=tr[~np.isin(groups[tr],np.unique(groups[te]))]
  assert not set(groups[tr]) & set(groups[te]);assert all((fold,int(i)) in oldfold for i in te)
  if design.startswith('paper_'):assert not set(papers[tr]) & set(papers[te])
  folds.append({'design':design,'fold':fold,'train_indices':tr.tolist(),'test_indices':te.tolist(),'purged_rows':n_before-len(tr)})
  for name,X in features.items():
   if (design,fold,name) in done:continue
   t=time.time();model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=4,max_features=.8,n_jobs=2,random_state=2026)
   model.fit(X[tr],y[tr],sample_weight=weights(tr));p=np.clip(model.predict(X[te]),0,100)
   results.append(dict(design=design,fold=fold,model=name,train_rows=len(tr),test_rows=len(te),purged_rows=n_before-len(tr),seconds=time.time()-t,**metrics(te,p)))
   predictions.extend(dict(design=design,fold=fold,model=name,row_index=int(i),observed=float(y[i]),predicted=float(v),chemical_key=r[i]['chemical_key']) for i,v in zip(te,p))
   save('fold_metrics.json',results);save('partial_predictions.json',predictions)
   print(f'{design} fold={fold} {name} MAE={results[-1]["mae_percentage_points"]:.3f} {time.time()-t:.1f}s',flush=True)
save('fold_indices.json',folds);save('pilot_predictions.json',predictions)
summary=[]
for design in sorted({v['design'] for v in predictions}):
 for name in list(features)+['dose_species_only']:
  ps=[p for p in predictions if p['design']==design and p['model']==name];ix=np.array([p['row_index'] for p in ps]);assert len(ix)==3380 and len(set(ix))==3380
  summary.append(dict(design=design,model=name,condition_only_reused=name=='dose_species_only',**metrics(ix,np.array([p['predicted'] for p in ps]))))
save('pilot_summary.json',summary)
rng=np.random.default_rng(2026);comparisons=[]
for design in sorted({v['design'] for v in predictions}):
 new={name:{p['row_index']:p['predicted'] for p in predictions if p['design']==design and p['model']==name} for name in features};old={p['row_index']:p['predicted'] for p in previous if p['design']==design and p['model']=='peptide'}
 for label,a,b in [('add_dose',new['peptide'],new['peptide_dose']),('add_species',new['peptide_dose'],new['peptide_dose_species']),('all_conditions',new['peptide'],new['peptide_dose_species']),('replace_handcrafted_by_ProtT5',old,new['peptide'])]:
  ix=np.array(sorted(a));w=weights(ix);delta=np.array([abs(y[i]-a[i])-abs(y[i]-b[i]) for i in ix]);g=groups[ix];ug=np.unique(g);num=np.array([np.sum(delta[g==v]*w[g==v]) for v in ug]);den=np.array([np.sum(w[g==v]) for v in ug]);boot=[]
  for _ in range(2000):bb=rng.integers(len(ug),size=len(ug));boot.append(float(num[bb].sum()/den[bb].sum()))
  comparisons.append(dict(design=design,comparison=label,MAE_reduction_percentage_points=float(np.average(delta,weights=w)),paired_cluster_bootstrap_95pct=np.quantile(boot,[.025,.975]).tolist()))
save('paired_comparisons.json',comparisons)
panel=read(BASE/'same_paper_dose_cross_species.json');panel_ids={a['id'] for b in panel for a in b['records']};ixset={i for i,a in enumerate(r) if a['id'] in panel_ids};pm=[]
for design in sorted({v['design'] for v in predictions}):
 for name in list(features)+['dose_species_only']:
  ps=[v for v in predictions if v['design']==design and v['model']==name and v['row_index'] in ixset];ix=np.array([v['row_index'] for v in ps]);pm.append(dict(design=design,model=name,records=len(ix),**metrics(ix,np.array([v['predicted'] for v in ps]))))
save('cross_species_panel_metrics.json',pm)
after={str(p):sha(p) for p in input_paths};assert before==after
save('verification.json',{'input_hashes_before':before,'input_hashes_after':after,'same_exact_folds_as_previous':True,'sequence_group_disjoint':True,'strict_paper_group_disjoint':True,'records_per_model_per_design':3380,'frozen_original_encoder':True,'head_fits':len(results),'condition_only_predictions_reused':True,'labels_or_GHXTox_weights_modified':False,'test_prediction_access':False})
save('protocol.json',{'encoder_manifest':'embedding_manifest.json','input_records_sha256':sha(BASE/'model_records.json'),'changed_component':'handcrafted sequence features replaced by 1024d frozen residue-mean ProtT5','unchanged':['records','target','terminal features','dose/species','sequence edit groups','PMID holdout and purging','head hyperparameters','chemical-form sample weighting'],'head':'ExtraTreesRegressor 160 trees min_samples_leaf4 max_features0.8 seed2026','tuning':False,'bootstrap':'2000 paired sequence-cluster resamples of fixed OOF errors; excludes model refitting and experimental uncertainty','note':'exploratory sequential representation comparison; no new blind validation'})
print(json.dumps(summary,ensure_ascii=False));print(json.dumps(comparisons,ensure_ascii=False))
