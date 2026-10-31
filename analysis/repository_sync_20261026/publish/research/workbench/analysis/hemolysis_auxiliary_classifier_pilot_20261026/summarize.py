from pathlib import Path
import json,collections
import numpy as np
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score,confusion_matrix
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
P=read(O/'protocol.json');pred=read(O/'oof_predictions.json');main=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json');components=read(R/'analysis/hemolysis_auxiliary_feasibility_20261026/joint_sequence_components.json')
ids=[r['sample_id'] for r in main];y=np.array([int(r['label']) for r in main]);fold=np.array([int(r['fold']) for r in main]);idx={s:i for i,s in enumerate(ids)}
g=np.array([components[r['sequence']] for r in main]);_,gi=np.unique(g,return_inverse=True);ng=len(set(g));mat={}
for seed in P['seeds']:
 for arm in P['arms']:
  records=[r for r in pred if r['seed']==seed and r['arm']==arm];assert len(records)==len({r['sample_id'] for r in records})==6387
  a=np.full(6387,np.nan)
  for r in records:assert y[idx[r['sample_id']]]==r['label'];a[idx[r['sample_id']]]=r['probability']
  assert np.isfinite(a).all();mat[(seed,arm)]=a
def metrics(mask,p):
 yy=y[mask];pp=p[mask];tn,fp,fn,tp=confusion_matrix(yy,pp>=.5,labels=[0,1]).ravel()
 return dict(n=int(mask.sum()),MCC=float(matthews_corrcoef(yy,pp>=.5)),AUPRC=float(average_precision_score(yy,pp)),AUROC=float(roc_auc_score(yy,pp)),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn))
allmask=np.ones(6387,dtype=bool);seedmetrics=[dict(seed=seed,arm=arm,**metrics(allmask,p)) for (seed,arm),p in mat.items()]
ensemble={arm:np.mean([mat[(seed,arm)] for seed in P['seeds']],axis=0) for arm in P['arms']}
ensmetrics=[dict(arm=arm,**metrics(allmask,p)) for arm,p in ensemble.items()]
foldmetrics=[dict(fold=f,arm=arm,**metrics(fold==f,p)) for f in range(5) for arm,p in ensemble.items()]
cm={}
for arm,p in ensemble.items():
 pr=p>=.5;cats=np.where(y==0,np.where(pr,1,0),np.where(pr,3,2));c=np.zeros((ng,4));np.add.at(c,(gi,cats),1);cm[arm]=c
def mcc(c):
 tn,fp,fn,tp=c;den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));return (tp*tn-fp*fn)/den if den else 0.
rng=np.random.default_rng(20261026);samples={arm:[] for arm in P['arms']}
for _ in range(P['bootstrap']):
 w=np.bincount(rng.integers(0,ng,ng),minlength=ng)
 for arm,c in cm.items():samples[arm].append(float(mcc(w@c)))
comparisons=[]
for base in ['toxicity_only','permuted_hemolysis_aux']:
 a=next(x for x in ensmetrics if x['arm']=='real_hemolysis_aux');b=next(x for x in ensmetrics if x['arm']==base);d=np.array(samples['real_hemolysis_aux'])-np.array(samples[base]);ci=np.quantile(d,[.025,.975]).tolist()
 comparisons.append(dict(candidate='real_hemolysis_aux',reference=base,delta_MCC=a['MCC']-b['MCC'],delta_AUPRC=a['AUPRC']-b['AUPRC'],delta_sensitivity=a['sensitivity']-b['sensitivity'],delta_specificity=a['specificity']-b['specificity'],delta_FN=a['FN']-b['FN'],delta_FP=a['FP']-b['FP'],MCC_paired_joint_component_bootstrap_ci95=ci,n_components=ng,scope='conditional on saved two-seed predictions, no retraining uncertainty, not a blind external test'))
pool=read(R/'analysis/classification_mechanism_map_20261026/membrane76_comparison_pool.json');memids={r['sample_id'] for r in pool['all76']};mask=np.array([s in memids for s in ids]);sub=[]
for arm,p in ensemble.items():
 sub.append(dict(arm=arm,membrane76_FN=int(np.sum(p[mask]<.5)),membrane76_TP=int(np.sum(p[mask]>=.5)),remaining6311=metrics(~mask,p),scope='old annotation/error-selected diagnostic pool, not mechanistic-positive gold'))
seed_deltas=[]
for seed in P['seeds']:
 a=next(x for x in seedmetrics if x['seed']==seed and x['arm']=='real_hemolysis_aux');b=next(x for x in seedmetrics if x['seed']==seed and x['arm']=='toxicity_only');seed_deltas.append(dict(seed=seed,delta_MCC=a['MCC']-b['MCC'],delta_AUPRC=a['AUPRC']-b['AUPRC']))
c=comparisons[0];ctrl=comparisons[1]
gate=dict(delta_MCC_at_least_005=c['delta_MCC']>=.005,bootstrap_lower_above_zero=c['MCC_paired_joint_component_bootstrap_ci95'][0]>0,both_seeds_nonnegative=all(x['delta_MCC']>=0 for x in seed_deltas),AUPRC_not_down_more_than_002=c['delta_AUPRC']>=-.002,real_above_permuted=ctrl['delta_MCC']>0)
decision=dict(criteria=gate,continue_gate_passed=all(gate.values()),action='replicate_with_independent_data_before_GHXTox_integration' if all(gate.values()) else 'do_not_tune_or_scale_this_fixed_auxiliary_configuration',scope='only this source-label frozen-mean small-head pilot, not a proof all hemolysis supervision cannot help')
for n,obj in [('seed_metrics.json',seedmetrics),('ensemble_metrics.json',ensmetrics),('fold_metrics.json',foldmetrics),('paired_comparisons.json',comparisons),('subgroup_metrics.json',sub),('seed_deltas.json',seed_deltas),('decision.json',decision)]: (O/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
np.savez_compressed(O/'ensemble_predictions.npz',sample_ids=np.array(ids),labels=y,**ensemble)
print(json.dumps(dict(metrics=ensmetrics,comparisons=comparisons,subgroups=sub,seed_deltas=seed_deltas,decision=decision),ensure_ascii=False,indent=2))
