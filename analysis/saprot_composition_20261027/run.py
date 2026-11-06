"""Fixed composition/order probes of frozen teacher behavior; no label-based fitting."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy.linalg import solve
from scipy.special import expit,log_expit
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,matthews_corrcoef,average_precision_score,roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(n,x):(O/n).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def log(s):
 print(s,flush=True)
 with (O/'run.log').open('a',encoding='utf-8') as out:out.write(s+'\n')
P=read(O/'protocol.json');paths=[O/'protocol.json',A/'saprot_and_robust_classifier_pilot_20261026/frozen_features.npz',A/'saprot_standalone_20261026/ensemble_scores.npz',A/'saprot_ensemble_size_control_20261026/ensemble_scores.npz',A/'saprot_explanation_20261026/crossfit_scores.npz',A/'classification_mechanism_map_20261026/membrane76_comparison_pool.json'];sourcehashes={str(p):sha(p) for p in paths};save('input_manifest.json',sourcehashes)
start=time.time();z=np.load(paths[1]);ids=z['sample_ids'].tolist();seqs=z['sequences'].tolist();fold=z['folds'];groups=z['groups'];labels=z['labels'];n=len(ids);old=np.load(paths[2]);four=np.load(paths[3]);split=np.load(paths[4]);half=split['halves'];assert ids==old['sample_ids'].tolist()==four['sample_ids'].tolist()==split['sample_ids'].tolist();assert np.array_equal(fold,split['folds']) and np.array_equal(groups,split['groups']) and np.array_equal(labels,split['labels'])
alphabet='ACDEFGHIKLMNPQRSTVWY';aa={a:i for i,a in enumerate(alphabet)};X=np.zeros((n,21));D=np.zeros((n,400));counts=[]
for i,s in enumerate(seqs):
 assert len(s)>1 and set(s)<=set(alphabet);c=np.array([s.count(a) for a in alphabet],dtype=float);X[i,:20]=c/len(s);X[i,20]=len(s);observed=np.zeros((20,20))
 for a,b in zip(s,s[1:]):observed[aa[a],aa[b]]+=1
 expected=(np.outer(c,c)-np.diag(c))/(len(s)*(len(s)-1));D[i]=(observed/(len(s)-1)-expected).ravel();counts.append(c.tolist())
assert np.allclose(X[:,:20].sum(axis=1),1) and np.isfinite(D).all();Q=PolynomialFeatures(degree=2,include_bias=False).fit_transform(X);assert Q.shape==(n,252)
sap=z['saprot_masked'].astype(float);base=z['base'];prot=base[:,:1024].astype(float);teacherprob=expit(old['SaProt_only_masked']);Y=np.column_stack([teacherprob,sap,prot]);base2=old['baseline'];base4=four['baseline4'];actual=four['fusion_masked4'];actualrescues=((base4>=0)!=labels)&((actual>=0)==labels);actualharms=((base4>=0)==labels)&((actual>=0)!=labels);assert actualrescues.sum()==96 and actualharms.sum()==62
arms=P['arms'];predp={a:np.full(n,np.nan) for a in arms};errors={a:{k:np.full(n,np.nan) for k in ['saprot_SSE','protT5_SSE','saprot_centered_cosine','protT5_centered_cosine']} for a in arms};reference={'saprot':np.full(n,np.nan),'protT5':np.full(n,np.nan)};models=[];splitlogs=[]
for ff in range(5):
 for hh in range(2):
  tr=(fold==ff)&(half!=hh);te=(fold==ff)&(half==hh);assert set(groups[tr]).isdisjoint(groups[te]);tridx=np.where(tr)[0];teidx=np.where(te)[0];meanY=Y[tr].mean(axis=0);center=Y[tr]-meanY
  cs=StandardScaler().fit(X[tr]);Ctrain=cs.transform(X[tr]);Ctest=cs.transform(X[te]);Kc=rbf_kernel(Ctrain,gamma=1/21);Kct=rbf_kernel(Ctest,Ctrain,gamma=1/21);ds=StandardScaler().fit(D[tr]);Dt=ds.transform(D[tr]);Dv=ds.transform(D[te]);norm=max(float(np.mean(np.sum(Dt**2,axis=1))),1e-12);Korder=Dt@Dt.T/norm;Kot=Dv@Dt.T/norm
  for key,sl in [('saprot',slice(1,481)),('protT5',slice(481,1505))]:reference[key][te]=np.sum((Y[te,sl]-meanY[sl])**2,axis=1)
  splitlogs.append(dict(original_fold=ff,evaluation_half=hh,train_n=int(tr.sum()),eval_n=int(te.sum()),train_ids=[ids[i] for i in tridx],evaluation_ids=[ids[i] for i in teidx],neural_and_probe_parent_groups_disjoint=True))
  for arm in arms:
   details=dict(arm=arm,fold=ff,evaluation_half=hh,teacher_target_mean=float(meanY[0]),teacher_label_used=False)
   if arm=='constant_teacher':prediction=np.broadcast_to(meanY,(te.sum(),Y.shape[1])).copy();details['kind']='constant'
   elif arm in ['composition_RBF','composition_order_kernel']:
    K,KT=(Kc,Kct) if arm=='composition_RBF' else ((Kc+Korder)/2,(Kct+Kot)/2);coef=solve(K+np.eye(len(K)),center,assume_a='pos');prediction=KT@coef+meanY;details.update(kind='kernel',teacher_coefficient=coef[:,0],composition_mean=cs.mean_,composition_scale=cs.scale_,order_mean=ds.mean_ if arm.endswith('kernel') else None,order_scale=ds.scale_ if arm.endswith('kernel') else None,order_normalizer=norm if arm.endswith('kernel') else None,train_indices=tridx)
   else:
    design=X if arm=='composition_linear' else Q if arm=='composition_quadratic' else np.column_stack([X,D]);sc=StandardScaler().fit(design[tr]);m=Ridge(alpha=1.,fit_intercept=False,solver='cholesky').fit(sc.transform(design[tr]),center);prediction=m.predict(sc.transform(design[te]))+meanY;details.update(kind='linear',teacher_coefficient=m.coef_[0],design_mean=sc.mean_,design_scale=sc.scale_)
   assert np.isfinite(prediction).all();predp[arm][te]=prediction[:,0]
   for key,sl in [('saprot',slice(1,481)),('protT5',slice(481,1505))]:
    errors[arm][key+'_SSE'][te]=np.sum((prediction[:,sl]-Y[te,sl])**2,axis=1);pcenter=prediction[:,sl]-meanY[sl];truecenter=Y[te,sl]-meanY[sl];den=np.linalg.norm(pcenter,axis=1)*np.linalg.norm(truecenter,axis=1);errors[arm][key+'_centered_cosine'][te]=np.divide(np.sum(pcenter*truecenter,axis=1),den,out=np.zeros(te.sum()),where=den>1e-14)
   models.append(details)
  log(f'fold{ff} half{hh}: 6fixed probes, train{tr.sum()} eval{te.sum()}, {time.time()-start:.1f}s')
for a in arms:assert np.isfinite(predp[a]).all()
def fusion_with_teacher(prob):
 p=np.clip(prob,0,1)
 with np.errstate(divide='ignore'):
  return np.logaddexp(log_expit(base2),np.log(p))-np.logaddexp(log_expit(-base2),np.log1p(-p))
def metric(score,mask=None):
 mask=np.ones(n,dtype=bool) if mask is None else mask;y=labels[mask];p=score[mask]>=0;both=len(set(y))==2
 return dict(n=int(mask.sum()),MCC=float(matthews_corrcoef(y,p)) if both else None,AUPRC=float(average_precision_score(y,score[mask])) if both else None,AUROC=float(roc_auc_score(y,score[mask])) if both else None,FN=int(((y==1)&~p).sum()),FP=int(((y==0)&p).sum()))
scores={'baseline4':base4,'actual_masked_fusion':actual};metrics=[];subgroups=[];memids={r['sample_id'] for r in read(paths[5])['all76']};cys=X[:,aa['C']]*X[:,20];masks={'lowCys_positive':(labels==1)&(np.rint(cys)<4),'membrane76':np.array([k in memids for k in ids])}
for arm in arms:
 pp=predp[arm];clipped=np.clip(pp,0,1);ss=fusion_with_teacher(clipped);scores[arm]=ss;ok=(ss>=0)==labels;rec=int((actualrescues&ok).sum());h=int((actualharms&~ok).sum());em=metric(ss);teacher_agree=float(np.mean((pp>=.5)==(old['SaProt_only_masked']>=0)));adequate=float(r2_score(teacherprob,pp))>=.8 and teacher_agree>=.95 and rec/96>=.8 and (em['FN']+em['FP'])<=int(((actual>=0)!=labels).sum())+.005*n
 metrics.append(dict(arm=arm,teacher_probability_R2=float(r2_score(teacherprob,pp)),teacher_probability_MAE=float(np.mean(abs(pp-teacherprob))),clipped_teacher_MAE=float(np.mean(abs(clipped-teacherprob))),teacher_decision_agreement=teacher_agree,unbounded_predictions_outside01=int(((pp<0)|(pp>1)).sum()),saprot_representation_skill=float(1-errors[arm]['saprot_SSE'].sum()/reference['saprot'].sum()),protT5_representation_skill=float(1-errors[arm]['protT5_SSE'].sum()/reference['protT5'].sum()),saprot_centered_cosine_mean=float(errors[arm]['saprot_centered_cosine'].mean()),protT5_centered_cosine_mean=float(errors[arm]['protT5_centered_cosine'].mean()),actual_fusion_rescues_reproduced=rec,actual_fusion_harms_reproduced=h,approx_fusion=em,new_rescues_vs_baseline4=int((~((base4>=0)==labels)&ok).sum()),new_harms_vs_baseline4=int((((base4>=0)==labels)&~ok).sum()),practical_screen_passed=bool(adequate),fold_teacher_R2=[float(r2_score(teacherprob[fold==ff],pp[fold==ff])) for ff in range(5)]))
for arm,ss in scores.items():
 for name,mask in masks.items():subgroups.append(dict(arm=arm,subgroup=name,**metric(ss,mask)))
# Conditional paired group uncertainty, not repeated neural/probe refitting.
_,gi=np.unique(groups,return_inverse=True);ng=gi.max()+1;rng=np.random.default_rng(20261027);W=np.array([np.bincount(rng.integers(0,ng,ng),minlength=ng) for _ in range(2000)],dtype=float);gn=np.bincount(gi,minlength=ng);sY=np.bincount(gi,weights=teacherprob,minlength=ng);ssY=np.bincount(gi,weights=teacherprob**2,minlength=ng);nn=W@gn;sst=W@ssY-(W@sY)**2/nn;ds={};reprden={k:W@np.bincount(gi,weights=v,minlength=ng) for k,v in reference.items()}
for arm in arms:
 scalarSSE=np.bincount(gi,weights=(predp[arm]-teacherprob)**2,minlength=ng);ds[arm]=dict(probability_R2=1-(W@scalarSSE)/sst)
 for key in reference:ds[arm][key+'_skill']=1-W@np.bincount(gi,weights=errors[arm][key+'_SSE'],minlength=ng)/reprden[key]
comparisons=[]
for a,b in P['primary_contrasts']:
 for measure in ['probability_R2','saprot_skill','protT5_skill']:
  d=ds[a][measure]-ds[b][measure];av=next(r for r in metrics if r['arm']==a);bv=next(r for r in metrics if r['arm']==b);field={'probability_R2':'teacher_probability_R2','saprot_skill':'saprot_representation_skill','protT5_skill':'protT5_representation_skill'}[measure];comparisons.append(dict(candidate=a,reference=b,measure=measure,delta=av[field]-bv[field],CI95=np.quantile(d,[.025,.975]),CI99=np.quantile(d,[.005,.995])))
save('metrics.json',metrics);save('paired_contrasts.json',comparisons);save('subgroup_metrics.json',subgroups);save('reference_metrics.json',[dict(arm=a,**metric(scores[a])) for a in ['baseline4','actual_masked_fusion']]);save('teacher_probability_models.json',models);save('splits.json',splitlogs)
np.savez_compressed(O/'probe_predictions.npz',sample_ids=np.array(ids),labels=labels,folds=fold,halves=half,groups=groups,true_teacher_probability=teacherprob,**{'teacher_'+k:v for k,v in predp.items()},**{'fusion_'+k:v for k,v in scores.items()},**{a+'_'+k:v for a,r in errors.items() for k,v in r.items()},**{'constant_'+k+'_SSE':v for k,v in reference.items()})
save('sample_reconstructions.json',[dict(sample_id=k,sequence=seqs[i],original_label_metadata_only=int(labels[i]),fold=int(fold[i]),half=int(half[i]),true_teacher_probability=float(teacherprob[i]),actual_masked_rescue=bool(actualrescues[i]),actual_masked_harm=bool(actualharms[i]),teacher_estimates={a:float(predp[a][i]) for a in arms},approx_fusion_probabilities={a:float(expit(scores[a][i])) for a in arms}) for i,k in enumerate(ids)])
assert all(sha(Path(path))==h for path,h in sourcehashes.items());save('verification.json',dict(n=n,sequence_only_features=True,neural_models_loaded_or_trained=0,new_GPU_embeddings=0,teacher_target_labels_used=False,all_meta_groups_disjoint=True,all_eval_and_metatrain_same_original_neural_heldout_fold=True,source_hashes_unchanged=True,probe_fits=50,constant_controls=10,seconds=time.time()-start,practical_screen_note='descriptive behavior reconstruction, not equivalence/causal attribution/model promotion'))
log('Finished '+str(len(arms))+' fixed arms; '+str(round(time.time()-start,1))+'s');print(json.dumps(clean(metrics),ensure_ascii=False,indent=2))
