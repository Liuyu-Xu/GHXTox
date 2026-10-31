"""Fixed, within-original-fold explanatory controls; no neural fitting."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
from scipy.special import expit,log_expit,logit
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score,r2_score
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(name,obj):(O/name).write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
P=read(O/'protocol.json');files=[O/'protocol.json',A/'saprot_contribution_20261026/sample_contributions.json',A/'saprot_ensemble_size_control_20261026/ensemble_scores.npz',A/'saprot_and_robust_classifier_pilot_20261026/frozen_features.npz',A/'classification_mechanism_map_20261026/membrane76_comparison_pool.json'];manifest={str(p):sha(p) for p in files};save('input_manifest.json',manifest)
rows=read(files[1]);z=np.load(files[3]);s=np.load(files[2]);ids=z['sample_ids'].tolist();idx={k:i for i,k in enumerate(ids)};y=z['labels'].astype(int);fold=z['folds'];groups=z['groups'];n=len(ids)
assert n==6387 and ids==s['sample_ids'].tolist() and np.array_equal(y,s['labels'])
byid={r['sample_id']:r for r in rows};rows=[byid[k] for k in ids]
for i,r in enumerate(rows):assert r['label']==y[i] and r['fold']==fold[i] and r['group_id']==groups[i] and r['sequence']==str(z['sequences'][i])
f=lambda k:np.array([r['features'][k] for r in rows],dtype=float)
base=s['baseline4'];bp=expit(base);length=f('sequence_length');cys=f('cys_count');span=f('cys_span_fraction');sequence=np.stack([length,cys,cys/length,span,base*(cys>=4)],axis=1)
assert np.isfinite(sequence).all()
tokens=[['curated:'+t for t in r['curated_family']]+['name:'+t for t in r['name_class']] for r in rows];missing=np.array([[not r['curated_family'],not r['name_class']] for r in rows],dtype=float)
half=np.full(n,-1,dtype=int)
for ff in range(5):
 gg=sorted(set(groups[fold==ff]),key=lambda g:hashlib.sha256((str(g)+'|saprot_explanation_v1').encode()).hexdigest());mapping={g:j%2 for j,g in enumerate(gg)}
 for i in np.where(fold==ff)[0]:half[i]=mapping[groups[i]]
assert np.all(half>=0)
scores={k:np.full(n,np.nan) for k in P['label_based_arms']};scores['baseline4_raw']=base.copy()
for kind in ['3Di','masked']:scores['fusion_'+kind+'4_raw']=s['fusion_'+kind+'4'].copy()
reconstruction={(kind,level):np.full(n,np.nan) for kind in ['3Di','masked'] for level in ['base','sequence','family']};logs=[];splitlogs=[];models=[]
for ff in range(5):
 for hh in range(2):
  tr=(fold==ff)&(half!=hh);te=(fold==ff)&(half==hh);assert not set(groups[tr])&set(groups[te]);assert len(set(np.unique(y[tr])))==2
  splitlogs.append(dict(fold=ff,evaluation_half=hh,train_ids=[ids[i] for i in np.where(tr)[0]],evaluation_ids=[ids[i] for i in np.where(te)[0]],train_n=int(tr.sum()),evaluation_n=int(te.sum()),train_positive=int(y[tr].sum()),same_original_fold=True,neural_training_excludes_entire_original_fold=True,meta_train_evaluation_groups_disjoint=True))
  counts=Counter(t for i in np.where(tr)[0] for t in set(tokens[i]));vocab=sorted(t for t,c in counts.items() if c>=5);toksets=[set(t) for t in tokens];fam=np.column_stack([missing]+[np.array([t in ts for ts in toksets],dtype=float) for t in vocab]);Xseq=np.column_stack([base,sequence]);Xfamily=np.column_stack([Xseq,fam])
  result=minimize_scalar(lambda b:float(np.mean(np.logaddexp(0,base[tr]+b)-y[tr]*(base[tr]+b))),bounds=(-12,12),method='bounded');assert result.success;scores['baseline4_bias_offset'][te]=base[te]+result.x;models.append(dict(arm='baseline4_bias_offset',fold=ff,evaluation_half=hh,intercept=float(result.x)))
  xs={'baseline4_Platt':base[:,None],'baseline4_CysLength':Xseq,'baseline4_CysLengthFamily':Xfamily,'fusion_3Di4_Platt':s['fusion_3Di4'][:,None],'fusion_masked4_Platt':s['fusion_masked4'][:,None]}
  for name,X in xs.items():
   sc=StandardScaler().fit(X[tr]);m=LogisticRegression(C=1.,solver='lbfgs',max_iter=3000).fit(sc.transform(X[tr]),y[tr]);assert m.n_iter_.max()<3000;scores[name][te]=m.decision_function(sc.transform(X[te]));models.append(dict(arm=name,fold=ff,evaluation_half=hh,scaler_mean=sc.mean_,scaler_scale=sc.scale_,coefficient=m.coef_,intercept=m.intercept_,family_vocab=vocab if name.endswith('Family') else None,iterations=m.n_iter_))
  Xbase=np.column_stack([base,bp,bp*(1-bp)]);probe={'base':Xbase,'sequence':np.column_stack([Xbase,sequence]),'family':np.column_stack([Xbase,sequence,fam])}
  for kind in ['3Di','masked']:
   delta=expit(s['fusion_'+kind+'4'])-bp
   for level,X in probe.items():
    sc=StandardScaler().fit(X[tr]);m=Ridge(alpha=1.).fit(sc.transform(X[tr]),delta[tr]);reconstruction[kind,level][te]=m.predict(sc.transform(X[te]));models.append(dict(arm='shift_'+kind+'_'+level,fold=ff,evaluation_half=hh,scaler_mean=sc.mean_,scaler_scale=sc.scale_,coefficient=m.coef_,intercept=m.intercept_,family_vocab=vocab if level=='family' else None,toxicity_labels_used=False))
  print(f'fold{ff} half{hh}: calibrated/probed train{tr.sum()} eval{te.sum()} family_features{len(vocab)}',flush=True)
for a,v in scores.items():assert np.isfinite(v).all(),a
for a,v in reconstruction.items():assert np.isfinite(v).all(),a
def metric(score,mask=None):
 mask=np.ones(n,dtype=bool) if mask is None else mask;yy=y[mask];ss=score[mask];pp=expit(ss);pred=ss>=0;tn=int(((yy==0)&~pred).sum());fp=int(((yy==0)&pred).sum());fn=int(((yy==1)&~pred).sum());tp=int(((yy==1)&pred).sum());both=len(set(yy))==2
 return dict(n=len(yy),MCC=float(matthews_corrcoef(yy,pred)) if both else None,AUPRC=float(average_precision_score(yy,ss)) if both else None,AUROC=float(roc_auc_score(yy,ss)) if both else None,Brier=float(np.mean((pp-yy)**2)),NLL=float(np.mean(np.logaddexp(0,ss)-yy*ss)),FN=fn,FP=fp,TP=tp,TN=tn)
metrics=[dict(arm=a,**metric(v)) for a,v in scores.items()];foldmetrics=[dict(arm=a,fold=ff,**metric(v,fold==ff)) for a,v in scores.items() for ff in range(5)]
_,gi=np.unique(groups,return_inverse=True);ng=gi.max()+1;gs=np.bincount(gi,minlength=ng);rng=np.random.default_rng(20261026);W=np.array([np.bincount(rng.integers(0,ng,ng),minlength=ng) for _ in range(2000)],dtype=float);den=W@gs
boot={}
for a,v in scores.items():
 p=v>=0;cat=np.where(y==0,np.where(p,1,0),np.where(p,3,2));cm=np.zeros((ng,4));np.add.at(cm,(gi,cat),1);cc=W@cm;tn,fp,fn,tp=cc.T;dd=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));mcc=np.divide(tp*tn-fp*fn,dd,out=np.zeros(len(dd)),where=dd!=0);br=np.bincount(gi,weights=(expit(v)-y)**2,minlength=ng);nl=np.bincount(gi,weights=np.logaddexp(0,v)-y*v,minlength=ng);boot[a]=dict(MCC=mcc,Brier=W@br/den,NLL=W@nl/den)
lookup={r['arm']:r for r in metrics};comparisons=[]
for a,b in P['primary_comparisons']:
 d=boot[a]['MCC']-boot[b]['MCC'];fd=[next(r for r in foldmetrics if r['arm']==a and r['fold']==ff)['MCC']-next(r for r in foldmetrics if r['arm']==b and r['fold']==ff)['MCC'] for ff in range(5)]
 comparisons.append(dict(candidate=a,reference=b,delta_MCC=lookup[a]['MCC']-lookup[b]['MCC'],CI95=np.quantile(d,[.025,.975]),CI99_1667=np.quantile(d,[.05/12,1-.05/12]),delta_AUPRC=lookup[a]['AUPRC']-lookup[b]['AUPRC'],delta_Brier=lookup[a]['Brier']-lookup[b]['Brier'],Brier_CI95=np.quantile(boot[a]['Brier']-boot[b]['Brier'],[.025,.975]),delta_NLL=lookup[a]['NLL']-lookup[b]['NLL'],NLL_CI95=np.quantile(boot[a]['NLL']-boot[b]['NLL'],[.025,.975]),fold_delta_MCC=fd))
memids={r['sample_id'] for r in read(files[4])['all76']};strata={'all':np.ones(n,dtype=bool),'negative':y==0,'positive':y==1,'lowCys_negative':(y==0)&(cys<4),'lowCys_positive':(y==1)&(cys<4),'highCys_positive':(y==1)&(cys>=4),'membrane76':np.array([k in memids for k in ids])};baseok=(base>=0)==y;sub=[]
for a,v in scores.items():
 ok=(v>=0)==y
 for key,mask in strata.items():sub.append(dict(arm=a,subgroup=key,**metric(v,mask),corrected=int((~baseok&ok&mask).sum()),harmed=int((baseok&~ok&mask).sum())))
probes=[]
for (kind,level),pred in reconstruction.items():
 target=expit(s['fusion_'+kind+'4'])-bp;ap=np.clip(bp+pred,1e-12,1-1e-12);ascore=logit(ap);tok=(s['fusion_'+kind+'4']>=0)==y;aok=(ascore>=0)==y;rescued=~baseok&tok;harmed=baseok&~tok
 probes.append(dict(kind=kind,level=level,R2_shift=float(r2_score(target,pred)),MAE_shift=float(np.mean(np.abs(target-pred))),true_shift_std=float(np.std(target)),fold_R2=[float(r2_score(target[fold==ff],pred[fold==ff])) for ff in range(5)],actual_teacher_rescues=int(rescued.sum()),rescues_reproduced=int((rescued&aok).sum()),actual_teacher_harms=int(harmed.sum()),harms_reproduced=int((harmed&~aok).sum()),approximation_vs_baseline_rescues=int((~baseok&aok).sum()),approximation_vs_baseline_harms=int((baseok&~aok).sum()),approximate_classifier=metric(ascore),scope='toxicity-label-free fit to frozen teacher behavior; predictability not causal explained fraction'))
save('metrics.json',metrics);save('fold_metrics.json',foldmetrics);save('paired_comparisons.json',comparisons);save('subgroup_metrics.json',sub);save('teacher_shift_reconstruction.json',probes);save('fitted_controls.json',models);save('splits.json',splitlogs)
np.savez_compressed(O/'crossfit_scores.npz',sample_ids=np.array(ids),labels=y,folds=fold,groups=groups,halves=half,**scores,**{'shift_'+k+'_'+l:v for (k,l),v in reconstruction.items()})
save('sample_predictions.json',[dict(sample_id=k,label=int(y[i]),fold=int(fold[i]),evaluation_half=int(half[i]),scores={a:float(v[i]) for a,v in scores.items()},probabilities={a:float(expit(v[i])) for a,v in scores.items()}) for i,k in enumerate(ids)])
assert all(sha(Path(p))==h for p,h in manifest.items());save('verification.json',dict(samples=n,original_folds=5,subfolds=10,source_hashes_unchanged=True,neural_models_trained=0,new_label_control_fits=60,label_free_ridge_fits=60,all_meta_train_evaluation_groups_disjoint=True,all_meta_train_and_evaluation_same_original_neural_heldout_fold=True,all_evaluations_have_one_crossfit_prediction=True,no_external_tests_used=True,interpretation='exploratory conditional comparison, not independent confirmation'))
print(json.dumps(clean(dict(metrics=metrics,comparisons=comparisons,probes=probes)),ensure_ascii=False,indent=2))
