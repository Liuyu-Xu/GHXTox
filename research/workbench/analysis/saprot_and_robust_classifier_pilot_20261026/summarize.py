from pathlib import Path
import json
import numpy as np
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score,brier_score_loss,confusion_matrix
from scipy.special import expit,log_expit,logsumexp
O=Path(__file__).resolve().parent;R=O.parents[1];read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
P=read(O/'protocol.json');z=np.load(O/'frozen_features.npz');ids=z['sample_ids'].tolist();y=z['labels'];fold=z['folds'];g=z['groups'];idx={s:i for i,s in enumerate(ids)};_,gi=np.unique(g,return_inverse=True);ng=len(set(g));pred=read(O/'oof_predictions.json');mat={}
archive=O/'rounded_probability_archive';archive.mkdir(exist_ok=True)
for name in ['ensemble_metrics.json','seed_metrics.json','paired_comparisons.json','decision.json']:
 if (O/name).exists() and not (archive/name).exists():(archive/name).write_bytes((O/name).read_bytes())
raw_logits=read(O/'oof_logits.json');lm={(r['seed'],r['arm'],r['sample_id']):r['logit'] for r in raw_logits};scores={}
for seed in P['seeds']:
 for arm in P['arms']:
  rows=[r for r in pred if r['seed']==seed and r['arm']==arm];assert len(rows)==len({r['sample_id'] for r in rows})==6387;a=np.full(6387,np.nan)
  for r in rows:
   i=idx[r['sample_id']];assert r['label']==int(y[i]) and r['fold']==int(fold[i]);a[i]=r['probability']
  assert np.isfinite(a).all();lo=np.array([lm[(seed,arm,s)] for s in ids],dtype=np.float64);precise=expit(lo);assert np.array_equal(precise>=.5,a>=.5);mat[(seed,arm)]=precise;scores[(seed,arm)]=lo
def metric(mask,p,score=None):
 yy=y[mask];pp=p[mask];tn,fp,fn,tp=map(int,confusion_matrix(yy,pp>=.5,labels=[0,1]).ravel());both=len(np.unique(yy))==2
 rank=pp if score is None else score[mask]
 return dict(n=len(yy),MCC=float(matthews_corrcoef(yy,pp>=.5)) if both else None,AUPRC=float(average_precision_score(yy,rank)) if both else None,AUROC=float(roc_auc_score(yy,rank)) if both else None,Brier=float(brier_score_loss(yy,pp)),sensitivity=tp/(tp+fn) if tp+fn else None,specificity=tn/(tn+fp) if tn+fp else None,TN=tn,FP=fp,FN=fn,TP=tp,original_label_disagreements=fn+fp)
allmask=np.ones(6387,dtype=bool);seedmetrics=[dict(seed=s,arm=a,**metric(allmask,p,scores[(s,a)])) for (s,a),p in mat.items()];ensembles={};ensemble_scores={}
for a in P['arms']:
 lo=np.array([scores[(s,a)] for s in P['seeds']]);rank=logsumexp(log_expit(lo),axis=0)-logsumexp(log_expit(-lo),axis=0);ensemble_scores[a]=rank;ensembles[a]=expit(rank)
 oldp=np.mean([np.array([r['probability'] for r in sorted([r for r in pred if r['arm']==a and r['seed']==s],key=lambda r:idx[r['sample_id']])]) for s in P['seeds']],axis=0)
 assert np.array_equal(oldp>=.5,ensembles[a]>=.5)
em=[dict(arm=a,**metric(allmask,p,ensemble_scores[a])) for a,p in ensembles.items()];fm=[dict(fold=f,arm=a,**metric(fold==f,p,ensemble_scores[a])) for f in range(5) for a,p in ensembles.items()]
cm={}
for a,p in ensembles.items():
 predclass=p>=.5;cat=np.where(y==0,np.where(predclass,1,0),np.where(predclass,3,2));c=np.zeros((ng,4));np.add.at(c,(gi,cat),1);cm[a]=c
def mcc(c):
 tn,fp,fn,tp=c;den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));return (tp*tn-fp*fn)/den if den else 0.
samples={a:[] for a in P['arms']};rng=np.random.default_rng(20261026)
for _ in range(2000):
 w=np.bincount(rng.integers(0,ng,ng),minlength=ng)
 for a,c in cm.items():samples[a].append(float(mcc(w@c)))
comparisons=[]
for a,b in [('plus_SaProt_3Di_BCE','baseline_BCE'),('plus_SaProt_3Di_BCE','plus_SaProt_masked_BCE'),('baseline_GCE_q07','baseline_BCE')]:
 ca=next(x for x in em if x['arm']==a);cb=next(x for x in em if x['arm']==b);d=np.array(samples[a])-np.array(samples[b]);seed_deltas=[]
 for s in P['seeds']:
  sa=next(x for x in seedmetrics if x['arm']==a and x['seed']==s);sb=next(x for x in seedmetrics if x['arm']==b and x['seed']==s);seed_deltas.append(dict(seed=s,delta_MCC=sa['MCC']-sb['MCC'],delta_AUPRC=sa['AUPRC']-sb['AUPRC']))
 comparisons.append(dict(candidate=a,reference=b,delta_MCC=ca['MCC']-cb['MCC'],delta_AUPRC=ca['AUPRC']-cb['AUPRC'],delta_sensitivity=ca['sensitivity']-cb['sensitivity'],delta_FN=ca['FN']-cb['FN'],delta_FP=ca['FP']-cb['FP'],CI95=np.quantile(d,[.025,.975]).tolist(),CI98_3333=np.quantile(d,[1/120,119/120]).tolist(),seed_deltas=seed_deltas,n_groups=ng,scope='paired group resampling conditional on predictions; not re-training uncertainty or independent final validation'))
original={r['sample_id']:r for r in read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json')};mem={r['sample_id'] for r in read(R/'analysis/classification_mechanism_map_20261026/membrane76_comparison_pool.json')['all76']};structrows={r['sample_id']:r for r in read(R/'analysis/saprot_input_audit_20261026/structure_sequences.json')}
flags={'membrane76':np.array([s in mem for s in ids]),'cross_source_label_conflict':np.array([original[s]['cross_source_label_disagreement']=='True' for s in ids]),'fragment_reference_flag':np.array([original[s]['any_fragment_match']=='True' for s in ids]),'positive_low_Cys_lt4':np.array([int(original[s]['cys_count'])<4 and int(original[s]['label'])==1 for s in ids]),'entirely_structure_masked':np.array([structrows[s]['unmasked_structure_n']==0 for s in ids])}
flags['no_simple_flags']=~(flags['cross_source_label_conflict']|flags['fragment_reference_flag']);sub=[dict(subgroup=k,arm=a,**metric(mask,p,ensemble_scores[a]),scope='original-label diagnostic; flags do not certify errors or safe negatives') for k,mask in flags.items() for a,p in ensembles.items()]
c,c2,c3=comparisons
sg=dict(delta_MCC_at_least005=c['delta_MCC']>=.005,correctedCIpositive_vsbase=c['CI98_3333'][0]>0,correctedCIpositive_vsmasked=c2['CI98_3333'][0]>0,bothseeds_nondecreasing_vsbase=all(x['delta_MCC']>=0 for x in c['seed_deltas']),AP_not_down_more002=c['delta_AUPRC']>=-.002)
rg=dict(delta_MCC_at_least005=c3['delta_MCC']>=.005,correctedCIpositive_vsbase=c3['CI98_3333'][0]>0,bothseeds_nondecreasing_vsbase=all(x['delta_MCC']>=0 for x in c3['seed_deltas']),AP_not_down_more002=c3['delta_AUPRC']>=-.002,sensitivity_not_down_more005=c3['delta_sensitivity']>=-.005)
dec=dict(structure_screen=dict(criteria=sg,passed=all(sg.values())),robust_loss_screen=dict(criteria=rg,passed=all(rg.values())),automatic_distillation_or_GHXTox_integration=False,scope='frozen35M approximatebackbone3Di featureprobe and fixedGCE; no universal rejection of structure or noise learning')
for n,obj in [('ensemble_metrics.json',em),('seed_metrics.json',seedmetrics),('fold_metrics.json',fm),('paired_comparisons.json',comparisons),('subgroup_metrics.json',sub),('decision.json',dec)]: (O/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
np.savez_compressed(O/'ensemble_predictions.npz',sample_ids=np.array(ids),labels=y,**ensembles)
np.savez_compressed(O/'ensemble_stable_ranking.npz',sample_ids=np.array(ids),**ensemble_scores)
(O/'precision_correction.json').write_text(json.dumps(dict(reason='float32 sigmoid saturation created artificial ties, especiallyGCE; allarms uniformly corrected',method='AP/AUROC use stable logodds of exactly same two-seed mean probability; classifier threshold unchanged',n_classifier_decisions_changed=0,not_a_new_prediction_rule=True,not_a_new_fit=True,rounded_original_metrics_preserved=True),indent=2),encoding='utf-8')
print(json.dumps(dict(metrics=em,comparisons=comparisons,decision=dec,membrane=[x for x in sub if x['subgroup']=='membrane76'],conflict=[x for x in sub if x['subgroup']=='cross_source_label_conflict']),ensure_ascii=False,indent=2))
