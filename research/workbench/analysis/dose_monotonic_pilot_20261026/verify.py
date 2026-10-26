"""Independent saved-artifact checks; no training or candidate reselection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import json,hashlib,collections,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
F=read(O/'freeze_manifest.json');P=read(O/'protocol.json');D=read(O/'prepared_data.json');splits=read(O/'folds.json');fm=read(O/'feature_manifest.json')
for path,h in {**F['input_hashes'],**F['frozen_hashes']}.items():assert sha(path)==h
audit=read(O/'run_audit.json');assert sha(O/'fit.py')==audit['fit_script_sha256'] and audit['model_fits']==20
pred=pd.read_csv(O/'oof_predictions.csv',dtype={'record_id':str,'pmid':str},float_precision='round_trip')
loss=pd.read_csv(O/'panel_losses.csv',dtype={'pmid':str},float_precision='round_trip')
metrics=read(O/'metrics_summary.json');names=fm['names'];dosecol=names.index('log10_dose_uM');assert dosecol==len(names)-1
aa='ACDEFGHIKLMNPQRSTVWY';kd=dict(zip(aa,[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
def vector(r):
 s=r['sequence'];n=len(s);c=collections.Counter(s);di=collections.Counter(zip(s[:-1],s[1:]));h=np.array([kd[a] for a in s]);theta=np.arange(n)*np.deg2rad(100)
 special={'length':n,'KR_DE_density':(c['K']+c['R']-c['D']-c['E'])/n,'mean_KD':h.mean(),'ideal_alpha_hydrophobic_moment':np.hypot((h*np.cos(theta)).sum(),(h*np.sin(theta)).sum())/n,'C_amide':r['c_terminal']=='amide','N_acetyl':r['n_terminal']=='acetyl','log10_dose_uM':np.log10(r['dose_uM'])}
 return [special[name] if name in special else c[name[3:]]/n if name.startswith('AA_') else di[tuple(name[3:])]/max(1,n-1) if name.startswith('di_') else r['species']==name[4:] for name in names]
largest_error=0.;total_mono_violations=0
with threadpool_limits(limits=2):
 for dataset,content in D.items():
  rows={r['id']:r for r in content['points']}
  for modelname in ['unconstrained','monotonic_dose']:
   ps=pred[(pred.dataset==dataset)&(pred.model==modelname)]
   assert ps.record_id.is_unique and set(ps.record_id)==set(rows)
   panelerrors=[]
   for panel,g in ps.groupby('panel_id'):
    g=g.sort_values('dose_uM');err=float(np.mean(abs(g.observed_percent-g.predicted_percent)));panelerrors.append(err)
    saved=loss[(loss.dataset==dataset)&(loss.model==modelname)&(loss.panel_id==panel)].iloc[0]
    assert abs(err-saved.MAE_pp)<1e-10
    inc=float(np.mean(abs(np.diff(g.observed_percent)-np.diff(g.predicted_percent))));assert abs(inc-saved.increment_MAE_pp)<1e-10
   assert abs(np.mean(panelerrors)-metrics[dataset][modelname]['macro_MAE_pp'])<1e-10
  for fold in splits[dataset]:
   f=fold['fold'];tr=[rows[x] for x in fold['train_ids']];te=[rows[x] for x in fold['test_ids']]
   assert not {r['edit_group'] for r in tr}&{r['edit_group'] for r in te}
   assert not {str(r['pmid']) for r in tr}&{str(r['pmid']) for r in te}
   X=np.asarray([vector(r) for r in te],dtype=float)
   models={name:joblib.load(O/'fold_models'/f'{dataset}_fold{f}_{name}.joblib') for name in ['unconstrained','monotonic_dose']}
   pa=models['unconstrained'].get_params();pb=models['monotonic_dose'].get_params();ca=np.asarray(pa.pop('monotonic_cst'));cb=np.asarray(pb.pop('monotonic_cst'));assert pa==pb
   assert not ca.any() and cb[dosecol]==1 and cb.sum()==1
   for name,model in models.items():
    fresh=np.clip(model.predict(X),0,100);lookup=pred[(pred.dataset==dataset)&(pred.model==name)&(pred.fold==f)].set_index('record_id')
    old=np.array([lookup.loc[r['id'],'predicted_percent'] for r in te]);e=float(max(abs(fresh-old)));largest_error=max(largest_error,e);assert e<1e-10
   # Every test panel must have identical non-dose features. Recheck full 64point grids.
   buckets=collections.defaultdict(list)
   for i,r in enumerate(te):buckets[r['panel_id']].append(i)
   grids=[]
   for panel,ix in buckets.items():
    other=[i for i in range(len(names)) if i!=dosecol]
    assert np.all(X[ix][:,other]==X[ix[0],other])
    z=np.repeat(X[ix[0]][None,:],64,axis=0);z[:,dosecol]=np.linspace(X[ix,dosecol].min(),X[ix,dosecol].max(),64);grids.append(z)
   gp=np.clip(models['monotonic_dose'].predict(np.vstack(grids)),0,100).reshape(-1,64)
   total_mono_violations+=int((np.diff(gp,axis=1)<-1e-7).sum())
assert total_mono_violations==0
decision=read(O/'decision.json');assert not decision['continue_gate_passed']
result=dict(all_frozen_original_inputs_unchanged=True,predictions_complete_and_unique=True,macro_MAE_and_adjacent_increment_errors_independently_recomputed=True,all_saved_model_point_predictions_reproduced=True,max_prediction_difference=largest_error,models_identical_except_dose_constraint=True,all_folds_paper_and_sequence_groups_disjoint=True,all_panels_nondose_features_identical=True,all_monotonic_64point_grids_verified=True,monotonic_grid_violation_count=total_mono_violations,no_new_fit_in_verification=True,subagents_used=False)
(O/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
