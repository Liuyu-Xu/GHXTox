"""Twenty fixed CPU fits; only new factor is monotonic dose constraint."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import csv,json,hashlib,time,itertools,collections
from pathlib import Path
import numpy as np
import sklearn,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(n,x):(O/n).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(n,rs):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
F=read(O/'freeze_manifest.json');P=read(O/'protocol.json')
for p,h in {**F['input_hashes'],**F['frozen_hashes']}.items():assert sha(p)==h,p
data=read(O/'prepared_data.json');splits=read(O/'folds.json')
AA='ACDEFGHIKLMNPQRSTVWY';DI=list(itertools.product(AA,repeat=2));KD=dict(zip(AA,[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
species=sorted({r['species'] for v in data.values() for r in v['points']})
names=['AA_'+a for a in AA]+['di_'+a+b for a,b in DI]+['length','KR_DE_density','mean_KD','ideal_alpha_hydrophobic_moment','C_amide','N_acetyl']+['RBC_'+s for s in species]+['log10_dose_uM']
doseindex=len(names)-1;assert doseindex==len(names)-1
save('feature_manifest.json',dict(names=names,dose_feature_index=doseindex,only_constrained_feature='log10_dose_uM',species_vocabulary=species,source_representation='same fixed composition/dipeptide/physical definitions as previous environment pilot, classifier and task are new'))
def features(r):
 s=r['sequence'];n=len(s);c=collections.Counter(s);di=collections.Counter(zip(s[:-1],s[1:]));h=np.array([KD[a] for a in s]);theta=np.arange(n)*np.deg2rad(100)
 return [c[a]/n for a in AA]+[di[a]/max(1,n-1) for a in DI]+[n,(c['K']+c['R']-c['D']-c['E'])/n,float(h.mean()),float(np.hypot((h*np.cos(theta)).sum(),(h*np.sin(theta)).sum())/n),float(r['c_terminal']=='amide'),float(r['n_terminal']=='acetyl')]+[float(r['species']==v) for v in species]+[float(np.log10(r['dose_uM']))]
def sample_weights(rows):
 counts=collections.Counter(r['panel_id'] for r in rows);w=np.array([1/counts[r['panel_id']] for r in rows]);return w/w.mean()
def median(y,w):
 order=np.argsort(y,kind='stable');return float(y[order][np.searchsorted(np.cumsum(w[order]),w.sum()/2)])
def panel_metrics(records,pred):
 groups=collections.defaultdict(list)
 for i,r in enumerate(records):groups[r['panel_id']].append(i)
 result=[]
 for panel,ix in groups.items():
  ix=sorted(ix,key=lambda i:records[i]['dose_uM']);y=np.array([records[i]['hemolysis_percent'] for i in ix]);p=np.array([pred[i] for i in ix]);r=records[ix[0]]
  result.append(dict(panel_id=panel,pmid=str(r['pmid']),edit_group=int(r['edit_group']),fold=int(r['fold']),n_points=len(ix),MAE_pp=float(abs(y-p).mean()),increment_MAE_pp=float(abs(np.diff(y)-np.diff(p)).mean()),predicted_decreasing_steps=int((np.diff(p)<-1e-7).sum()),adjacent_steps=len(ix)-1,observed_response_span_pp=float(np.ptp(y)),predicted_response_span_pp=float(np.ptp(p))))
 return result
def aggregate(panelrows):
 return dict(panels=len(panelrows),macro_MAE_pp=float(np.mean([r['MAE_pp'] for r in panelrows])),macro_increment_MAE_pp=float(np.mean([r['increment_MAE_pp'] for r in panelrows])),panels_with_predicted_decrease=sum(r['predicted_decreasing_steps']>0 for r in panelrows),predicted_decreasing_steps=sum(r['predicted_decreasing_steps'] for r in panelrows),adjacent_steps=sum(r['adjacent_steps'] for r in panelrows),mean_observed_panel_span_pp=float(np.mean([r['observed_response_span_pp'] for r in panelrows])),mean_predicted_panel_span_pp=float(np.mean([r['predicted_response_span_pp'] for r in panelrows])))
modelsdir=O/'fold_models';modelsdir.mkdir(exist_ok=True)
predictions=[];panelloss=[];foldmetrics=[];grid=[];timing=[]
start=time.perf_counter()
with threadpool_limits(limits=2):
 for dataset,v in data.items():
  rs=v['points'];index={r['id']:i for i,r in enumerate(rs)};X=np.array([features(r) for r in rs],dtype=np.float64);y=np.array([r['hemolysis_percent'] for r in rs],dtype=float)
  assert X.shape==(len(rs),len(names)) and np.isfinite(X).all()
  xhash=hashlib.sha256(X.tobytes()).hexdigest()
  for split in splits[dataset]:
   fold=split['fold'];tr=np.array([index[i] for i in split['train_ids']]);te=np.array([index[i] for i in split['test_ids']]);train=[rs[i] for i in tr];test=[rs[i] for i in te];weights=sample_weights(train)
   assert not {r['edit_group'] for r in train}&{r['edit_group'] for r in test}
   assert not {str(r['pmid']) for r in train}&{str(r['pmid']) for r in test}
   constant=np.full(len(te),median(y[tr],weights));pr=panel_metrics(test,constant)
   panelloss.extend(dict(dataset=dataset,model='training_weighted_median',**r) for r in pr)
   foldmetrics.append(dict(dataset=dataset,fold=fold,model='training_weighted_median',**aggregate(pr),point_MAE_pp=float(np.mean(abs(y[te]-constant)))))
   for modelname in ['unconstrained','monotonic_dose']:
    constraints=np.zeros(len(names),dtype=int)
    if modelname=='monotonic_dose':constraints[doseindex]=1
    model=HistGradientBoostingRegressor(**P['shared_parameters'],monotonic_cst=constraints)
    t=time.perf_counter();model.fit(X[tr],y[tr],sample_weight=weights);pred=np.clip(model.predict(X[te]),0,100);elapsed=time.perf_counter()-t
    joblib.dump(model,modelsdir/f'{dataset}_fold{fold}_{modelname}.joblib',compress=3)
    pr=panel_metrics(test,pred);panelloss.extend(dict(dataset=dataset,model=modelname,**r) for r in pr)
    foldmetrics.append(dict(dataset=dataset,fold=fold,model=modelname,**aggregate(pr),point_MAE_pp=float(np.mean(abs(y[te]-pred)))))
    for i,p in zip(te,pred):predictions.append(dict(dataset=dataset,model=modelname,record_id=rs[i]['id'],panel_id=rs[i]['panel_id'],pmid=str(rs[i]['pmid']),edit_group=int(rs[i]['edit_group']),fold=fold,dose_uM=float(rs[i]['dose_uM']),observed_percent=float(y[i]),predicted_percent=float(p)))
    buckets=collections.defaultdict(list)
    for i in te:buckets[rs[i]['panel_id']].append(i)
    GX=[];gridmeta=[]
    for panel,ix in buckets.items():
     doses=np.linspace(min(X[i,doseindex] for i in ix),max(X[i,doseindex] for i in ix),64)
     xx=np.repeat(X[ix[0]][None,:],64,axis=0);xx[:,doseindex]=doses;GX.append(xx);gridmeta.append(panel)
    gp=np.clip(model.predict(np.vstack(GX)),0,100).reshape(-1,64)
    violations=0
    for panel,pp in zip(gridmeta,gp):
     dd=np.diff(pp);n=int((dd<-1e-7).sum());violations+=n;grid.append(dict(dataset=dataset,fold=fold,model=modelname,panel_id=panel,grid_points=64,decreasing_grid_steps=n,max_downward_prediction_pp=float(max(0,-dd.min()))))
    if modelname=='monotonic_dose':assert violations==0
    assert xhash==hashlib.sha256(X.tobytes()).hexdigest()
    timing.append(dict(dataset=dataset,fold=fold,model=modelname,fit_predict_seconds=elapsed,train_points=len(tr),test_points=len(te)))
    print(f'{dataset} fold{fold} {modelname} MAE={aggregate(pr)["macro_MAE_pp"]:.3f} seconds={elapsed:.2f}',flush=True)
table('oof_predictions.csv',predictions);table('panel_losses.csv',panelloss);table('fold_metrics.csv',foldmetrics);table('grid_monotonicity.csv',grid);save('timing.json',timing)
summary={}
for dataset in data:
 summary[dataset]={model:aggregate([r for r in panelloss if r['dataset']==dataset and r['model']==model]) for model in ['training_weighted_median','unconstrained','monotonic_dose']}
rng=np.random.default_rng(P['seed']);comparisons=[]
for dataset in data:
 base={r['panel_id']:r for r in panelloss if r['dataset']==dataset and r['model']=='unconstrained'};mono={r['panel_id']:r for r in panelloss if r['dataset']==dataset and r['model']=='monotonic_dose'};assert set(base)==set(mono)
 delta=np.array([base[k]['MAE_pp']-mono[k]['MAE_pp'] for k in base]);keys=list(base)
 cis={}
 for unit in ['edit_group','pmid']:
  ids=sorted({base[k][unit] for k in keys},key=str);num=np.array([sum(delta[i] for i,k in enumerate(keys) if base[k][unit]==u) for u in ids]);den=np.array([sum(base[k][unit]==u for k in keys) for u in ids]);draw=rng.integers(0,len(ids),size=(2000,len(ids)));boot=num[draw].sum(axis=1)/den[draw].sum(axis=1);cis[unit]=np.quantile(boot,[.025,.975]).tolist()
 fd={f:float(np.mean([delta[i] for i,k in enumerate(keys) if base[k]['fold']==f])) for f in range(5)}
 inc=float(np.mean([base[k]['increment_MAE_pp']-mono[k]['increment_MAE_pp'] for k in keys]))
 comparisons.append(dict(dataset=dataset,MAE_reduction_pp=float(delta.mean()),paired_edit_group_bootstrap_CI95=cis['edit_group'],paired_pmid_bootstrap_CI95=cis['pmid'],fold_MAE_reduction_pp=fd,folds_improved=sum(v>0 for v in fd.values()),increment_MAE_reduction_pp=inc,scope='fixed exploratory OOF predictions; not retraining/measurement uncertainty'))
primary=next(x for x in comparisons if x['dataset']=='confirmed_overlay_primary');sensitivity=next(x for x in comparisons if x['dataset']=='all_exact50_quarantine_sensitivity')
checks=dict(primary_edit_group_CI_lower_positive=primary['paired_edit_group_bootstrap_CI95'][0]>0,primary_at_least_four_folds_improved=primary['folds_improved']>=4,sensitivity_MAE_reduction_positive=sensitivity['MAE_reduction_pp']>0,primary_increment_MAE_not_worse=primary['increment_MAE_reduction_pp']>=0)
decision=dict(checks=checks,continue_gate_passed=all(checks.values()),action='bounded_followup_only_no_deployment' if all(checks.values()) else 'do_not_upscale_or_tune_on_these_results')
save('metrics_summary.json',summary);save('paired_comparisons.json',comparisons);save('decision.json',decision)
for p,h in {**F['input_hashes'],**F['frozen_hashes']}.items():assert sha(p)==h,p
save('run_audit.json',dict(sklearn_version=sklearn.__version__,numpy_version=np.__version__,model_fits=len(timing),elapsed_seconds=time.perf_counter()-start,CPU_thread_limit=2,GPU_used=False,subagents_used=False,protocol_sha256=sha(O/'protocol.json'),fit_script_sha256=sha(Path(__file__)),all_frozen_and_original_inputs_unchanged=True,no_test_predictions=True,no_GHXTox_model_changes=True,models_not_for_deployment=True))
print(json.dumps(clean(dict(metrics=summary,comparisons=comparisons,decision=decision)),ensure_ascii=False,indent=2),flush=True)
