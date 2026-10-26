from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp,log_expit
from sklearn.metrics import matthews_corrcoef,average_precision_score,confusion_matrix
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'analysis/saprot_and_robust_classifier_pilot_20261026';T=R/'analysis/saprot_standalone_20261026';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
z=np.load(B/'frozen_features.npz');ids=z['sample_ids'].tolist();idx={s:i for i,s in enumerate(ids)};y=z['labels'];_,gi=np.unique(z['groups'],return_inverse=True);ng=len(set(gi));old=[r for r in read(B/'oof_logits.json') if r['arm']=='baseline_BCE'];extra=read(O/'oof_logits.json');base={}
for seed in [20261026,20261027,20261028,20261029]:
 r=[x for x in old+extra if x['seed']==seed];assert len(r)==6387;a=np.zeros(6387)
 for q in r:a[idx[q['sample_id']]]=q['logit']
 base[seed]=a
combine=lambda a:logsumexp(log_expit(np.array(a)),axis=0)-logsumexp(log_expit(-np.array(a)),axis=0)
s=np.load(T/'ensemble_scores.npz');scores={'baseline2':s['baseline'],'baseline4':combine(list(base.values())),'fusion_masked4':s['fusion_masked'],'fusion_3Di4':s['fusion_3Di']}
def metric(score):
 p=score>=0;tn,fp,fn,tp=map(int,confusion_matrix(y,p,labels=[0,1]).ravel());return dict(MCC=float(matthews_corrcoef(y,p)),AUPRC=float(average_precision_score(y,score)),FN=fn,FP=fp,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp))
metrics=[dict(arm=a,**metric(sc)) for a,sc in scores.items()];cms={}
for a,sc in scores.items():
 p=sc>=0;cat=np.where(y==0,np.where(p,1,0),np.where(p,3,2));c=np.zeros((ng,4));np.add.at(c,(gi,cat),1);cms[a]=c
def mcc(c):
 tn,fp,fn,tp=c;d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));return (tp*tn-fp*fn)/d if d else 0.
rng=np.random.default_rng(20261026);draws={a:[] for a in scores}
for _ in range(2000):
 w=np.bincount(rng.integers(0,ng,ng),minlength=ng)
 for a,c in cms.items():draws[a].append(float(mcc(w@c)))
comparisons=[]
ref=next(x for x in metrics if x['arm']=='baseline4')
for arm in ['fusion_masked4','fusion_3Di4']:
 a=next(x for x in metrics if x['arm']==arm);d=np.array(draws[arm])-np.array(draws['baseline4']);ci=np.quantile(d,[.0125,.9875]).tolist();criteria=dict(MCC_atleast005=a['MCC']-ref['MCC']>=.005,correctedCIpositive=ci[0]>0,AP_not_down_more002=a['AUPRC']-ref['AUPRC']>=-.002);comparisons.append(dict(candidate=arm,reference='baseline4',delta_MCC=a['MCC']-ref['MCC'],delta_AUPRC=a['AUPRC']-ref['AUPRC'],CI95=np.quantile(d,[.025,.975]).tolist(),CI97_5=ci,criteria=criteria,passed=all(criteria.values()),scope='same4head count;conditional OOF screen,not final independent validation'))
for n,obj in [('ensemble_metrics.json',metrics),('paired_comparisons.json',comparisons),('decision.json',dict(same_size_validation_passed=any(x['passed'] for x in comparisons),candidates=[x['candidate'] for x in comparisons if x['passed']],automatic_integration=False))]: (O/n).write_text(json.dumps(obj,indent=2),encoding='utf-8')
np.savez_compressed(O/'ensemble_scores.npz',sample_ids=np.array(ids),labels=y,**scores)
print(json.dumps(dict(metrics=metrics,comparisons=comparisons),indent=2))
