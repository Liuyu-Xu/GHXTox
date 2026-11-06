from pathlib import Path
import json
import numpy as np
from scipy.special import expit,log_expit,logsumexp
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score,confusion_matrix
O=Path(__file__).resolve().parent;R=O.parents[1];S=R/'analysis/saprot_and_robust_classifier_pilot_20261026';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
P=read(O/'protocol.json');z=np.load(S/'frozen_features.npz');ids=z['sample_ids'].tolist();idx={s:i for i,s in enumerate(ids)};y=z['labels'];fold=z['folds'];groups=z['groups'];_,gi=np.unique(groups,return_inverse=True);ng=len(set(groups));old=[r for r in read(S/'oof_logits.json') if r['arm']=='baseline_BCE'];new=read(O/'oof_predictions.json');logits={}
for seed in P['seeds']:
 for arm,source in [('baseline',old)]+[(a,new) for a in P['new_arms']]:
  rr=[r for r in source if r['seed']==seed and (r['arm']=='baseline_BCE' if arm=='baseline' else r['arm']==arm)];assert len(rr)==len({r['sample_id'] for r in rr})==6387;array=np.full(6387,np.nan)
  for r in rr:array[idx[r['sample_id']]]=r['logit']
  assert np.isfinite(array).all();logits[(seed,arm)]=array
def combine(arrays):
 l=np.stack(arrays);return logsumexp(log_expit(l),axis=0)-logsumexp(log_expit(-l),axis=0)
def metric(score,mask=None):
 if mask is None:mask=np.ones(6387,dtype=bool)
 yy=y[mask];ss=score[mask];tn,fp,fn,tp=map(int,confusion_matrix(yy,ss>=0,labels=[0,1]).ravel());both=len(np.unique(yy))==2
 return dict(n=len(yy),MCC=float(matthews_corrcoef(yy,ss>=0)) if both else None,AUPRC=float(average_precision_score(yy,ss)) if both else None,AUROC=float(roc_auc_score(yy,ss)) if both else None,sensitivity=tp/(tp+fn) if tp+fn else None,specificity=tn/(tn+fp) if tn+fp else None,TN=tn,FP=fp,FN=fn,TP=tp)
names=['baseline']+P['new_arms']+['fusion_masked','fusion_3Di'];scores={};seed_scores={}
for seed in P['seeds']:
 for a in ['baseline']+P['new_arms']:seed_scores[(seed,a)]=logits[(seed,a)]
 for kind in ['masked','3Di']:seed_scores[(seed,'fusion_'+kind)]=combine([logits[(seed,'baseline')],logits[(seed,'SaProt_only_'+kind)]])
for name in names:
 # Combining seed-wise fusion scores is mathematically equal to equal four-member averaging.
 scores[name]=combine([seed_scores[(seed,name)] for seed in P['seeds']])
em=[dict(arm=a,**metric(scores[a])) for a in names];sm=[dict(seed=s,arm=a,**metric(seed_scores[(s,a)])) for s in P['seeds'] for a in names];fm=[dict(fold=f,arm=a,**metric(scores[a],fold==f)) for f in range(5) for a in names]
prior=next(r for r in read(S/'ensemble_metrics.json') if r['arm']=='baseline_BCE');cur=em[0]
for k in ['MCC','AUPRC','AUROC']:assert abs(prior[k]-cur[k])<1e-12
complement=[]
for a in P['new_arms']:
 b_ok=(scores['baseline']>=0)==y;t_ok=(scores[a]>=0)==y
 complement.append(dict(teacher=a,both_correct=int((b_ok&t_ok).sum()),baseline_wrong_teacher_correct=int((~b_ok&t_ok).sum()),baseline_correct_teacher_wrong=int((b_ok&~t_ok).sum()),both_wrong=int((~b_ok&~t_ok).sum()),scope='oracle diagnostic requiring true labels; counts are not achievable routing improvement'))
cms={}
for a in names:
 pred=scores[a]>=0;cats=np.where(y==0,np.where(pred,1,0),np.where(pred,3,2));c=np.zeros((ng,4));np.add.at(c,(gi,cats),1);cms[a]=c
def mcc(c):
 tn,fp,fn,tp=c;d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));return (tp*tn-fp*fn)/d if d else 0.
rng=np.random.default_rng(20261026);draws={a:[] for a in names}
for _ in range(2000):
 w=np.bincount(rng.integers(0,ng,ng),minlength=ng)
 for a,c in cms.items():draws[a].append(float(mcc(w@c)))
comparisons=[]
for a,b,primary in [('SaProt_only_3Di','baseline',True),('fusion_3Di','baseline',True),('fusion_masked','baseline',True),('SaProt_only_masked','baseline',False),('SaProt_only_3Di','SaProt_only_masked',False),('fusion_3Di','fusion_masked',False)]:
 am=next(x for x in em if x['arm']==a);bm=next(x for x in em if x['arm']==b);d=np.array(draws[a])-np.array(draws[b]);sd=[]
 for s in P['seeds']:
  aa=next(x for x in sm if x['seed']==s and x['arm']==a);bb=next(x for x in sm if x['seed']==s and x['arm']==b);sd.append(dict(seed=s,delta_MCC=aa['MCC']-bb['MCC'],delta_AUPRC=aa['AUPRC']-bb['AUPRC']))
 comparisons.append(dict(candidate=a,reference=b,primary=primary,delta_MCC=am['MCC']-bm['MCC'],delta_AUPRC=am['AUPRC']-bm['AUPRC'],delta_FN=am['FN']-bm['FN'],delta_FP=am['FP']-bm['FP'],CI95=np.quantile(d,[.025,.975]).tolist(),CI98_3333=np.quantile(d,[1/120,119/120]).tolist() if primary else None,seed_deltas=sd,scope='paired originalgroup bootstrap conditional on saved fits; not independent final validation'))
decisions=[]
for a in ['fusion_masked','fusion_3Di']:
 c=next(x for x in comparisons if x['candidate']==a and x['reference']=='baseline');criteria=dict(MCC_atleast005=c['delta_MCC']>=.005,correctedCIpositive=c['CI98_3333'][0]>0,bothseeds_nondecreasing=all(x['delta_MCC']>=0 for x in c['seed_deltas']),AUPRC_not_down_more002=c['delta_AUPRC']>=-.002);decisions.append(dict(candidate=a,criteria=criteria,passed=all(criteria.values())))
pool=read(R/'analysis/classification_mechanism_map_20261026/membrane76_comparison_pool.json');mem={r['sample_id'] for r in pool['all76']};original={r['sample_id']:r for r in read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json')};masks={'membrane76':np.array([s in mem for s in ids]),'cross_source_conflict':np.array([original[s]['cross_source_label_disagreement']=='True' for s in ids]),'positive_low_Cys':np.array([int(original[s]['label'])==1 and int(original[s]['cys_count'])<4 for s in ids])};sub=[dict(subgroup=k,arm=a,**metric(score,mask),scope='original-label diagnostic,not experimentalgold') for k,mask in masks.items() for a,score in scores.items()]
decision=dict(fixed_fusion_screens=decisions,any_fusion_passed=any(x['passed'] for x in decisions),automatic_GHXTox_integration=False,distillation_performed=False,scope='only frozen35M meanfeatures and approximate3Di; no model/weight tuning from outcomes')
for n,obj in [('ensemble_metrics.json',em),('seed_metrics.json',sm),('fold_metrics.json',fm),('complementarity.json',complement),('paired_comparisons.json',comparisons),('subgroup_metrics.json',sub),('decision.json',decision)]: (O/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
np.savez_compressed(O/'ensemble_scores.npz',sample_ids=np.array(ids),labels=y,**scores)
print(json.dumps(dict(metrics=em,complementarity=complement,comparisons=comparisons,decision=decision),ensure_ascii=False,indent=2))
