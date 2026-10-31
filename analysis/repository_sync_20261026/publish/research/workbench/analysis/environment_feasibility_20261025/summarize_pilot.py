from pathlib import Path
import json,collections
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_text(encoding='utf-8'))
def save(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
r=read('model_records.json');pred=read('pilot_predictions.json');clusters={v['sequence']:v['group'] for v in read('sequence_clusters.json')}
groups=np.array([clusters[v['sequence']] for v in r]);papers=np.array([v['pmid'] for v in r]);keys=np.array([v['chemical_key'] for v in r]);y=np.array([v['hemolysis_percent'] for v in r])
def w(ix):
 c=collections.Counter(keys[ix]);return np.array([1/c[k] for k in keys[ix]])
blocks=read('same_paper_dose_cross_species.json');eligible={}
for v in blocks:
 k=(v['chemical_key'],v['pmid'],v['dose_uM'],v['pH'])
 for i,a in enumerate(r):
  if (a['chemical_key'],a['pmid'],a['dose_uM'],a['pH'])==k:eligible[i]=True
panels=[]
for design in sorted({v['design'] for v in pred}):
 for name in sorted({v['model'] for v in pred}):
  ps=[p for p in pred if p['design']==design and p['model']==name and p['row_index'] in eligible];ix=np.array([p['row_index'] for p in ps]);v=np.array([p['predicted'] for p in ps]);panels.append(dict(design=design,model=name,records=len(ix),chemical_forms=len(set(keys[ix])),mae_percentage_points=float(mean_absolute_error(y[ix],v,sample_weight=w(ix)))))
save('cross_species_panel_metrics.json',panels)
blockranges=[max(a['hemolysis_percent'] for a in b['records'])-min(a['hemolysis_percent'] for a in b['records']) for b in blocks]
save('cross_species_panel_description.json',{'blocks':len(blocks),'records':len(eligible),'chemical_forms':len({b['chemical_key'] for b in blocks}),'blocks_observed_range_at_least_10pp':sum(v>=10 for v in blockranges),'median_observed_range_pp':float(np.median(blockranges)),'same_paper_same_dose_not_full_assay_matching':True,'confounders_unknown':['time','temperature','RBC_density','buffer'],'outcome_zero_rows':int(sum(y==0)),'outcome_hundred_rows':int(sum(y==100))})
# A train-only constant is important given low R2; it is not hyperparameter tuning.
const=[]
for design in ['sequence_edit_groups','paper_holdout_purged_sequence_groups']:
 out={name:[] for name in ['training_mean','training_median']}
 for tr,te in GroupKFold(n_splits=5).split(y,groups=groups if design=='sequence_edit_groups' else papers):
  if design.startswith('paper_'):tr=tr[~np.isin(groups[tr],np.unique(groups[te]))]
  tw=w(tr);order=np.argsort(y[tr]);ym=y[tr][order];ww=tw[order];med=float(ym[np.searchsorted(np.cumsum(ww),ww.sum()/2)])
  for name,val in [('training_mean',float(np.average(y[tr],weights=tw))),('training_median',med)]:out[name].extend((int(i),val) for i in te)
 for name,ps in out.items():
  ix=np.array([i for i,v in ps]);v=np.array([v for i,v in ps]);const.append(dict(design=design,model=name,mae_percentage_points=float(mean_absolute_error(y[ix],v,sample_weight=w(ix)))))
save('constant_baselines.json',const)
checks=read('verification.json');checks['all_fold_prediction_counts_per_model']=dict(collections.Counter(p['design']+'::'+p['model'] for p in pred));assert set(checks['all_fold_prediction_counts_per_model'].values())=={len(r)};save('verification.json',checks)
print(json.dumps({'panel':read('cross_species_panel_description.json'),'panel_metrics':panels,'constant_baselines':const},ensure_ascii=False))
