"""Nested role-isolated three Cys experts against historical GHXTox OOF."""
import os,sys,json,hashlib,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,matthews_corrcoef,average_precision_score,roc_auc_score
from collections import Counter
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def group(c):return 'low' if c<=1 else 'middle' if c<=5 else 'high'
def main():
 start=time.time();meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');bg=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json');roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json');cache=torch.load(R/'runs/prottox_tsk_v1/features.pt',map_location='cpu',weights_only=False);records=cache['records'];assert len(meta)==len(records)==len(bg)==6387
 for m,b,c in zip(meta,bg,records):assert m['sequence']==c['sequence']==b['sequence'] and m['label']==int(c['label'])==int(b['label']) and m['sample_id']==b['sample_id']
 X=np.stack([x['tristat'].numpy().astype('float64') for x in records]);assert X.shape==(6387,3072) and np.isfinite(X).all()
 out=[];models=[];fold_audit=[]
 for fold in range(5):
  subsets={k:[x['source_index'] for x in roles if x['outer_fold']==fold and x['role']==k] for k in ['train','validation','calibration','test']};groups={k:{meta[i]['group_id'] for i in v} for k,v in subsets.items()};assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b)
  foldmap={};saved={}
  for g in ['low','middle','high']:
   ix={k:[i for i in v if group(meta[i]['cys_count'])==g] for k,v in subsets.items()};y=lambda ix:np.array([meta[i]['label'] for i in ix],dtype=int)
   for role in ix:assert len(set(y(ix[role])))==2,(fold,g,role)
   scaler=StandardScaler().fit(X[ix['train']]);xt=scaler.transform(X[ix['train']]);xv=scaler.transform(X[ix['validation']]);best=None
   for C in [0.001,0.01,0.1]:
    clf=LogisticRegression(C=C,class_weight='balanced',solver='liblinear',max_iter=500,random_state=42).fit(xt,y(ix['train']));loss=log_loss(y(ix['validation']),clf.predict_proba(xv)[:,1],labels=[0,1]);key=(loss,C)
    if best is None or key<best[0]:best=(key,clf)
   (_,chosen),clf=best;cal_logits=clf.decision_function(scaler.transform(X[ix['calibration']])).reshape(-1,1);cal=LogisticRegression(C=1,solver='lbfgs',max_iter=500).fit(cal_logits,y(ix['calibration']));pt=cal.predict_proba(clf.decision_function(scaler.transform(X[ix['test']])).reshape(-1,1))[:,1]
   for i,p in zip(ix['test'],pt):foldmap[i]=float(p)
   saved[g]={'scaler':scaler,'classifier':clf,'calibrator':cal,'C':chosen}
   print('fold',fold,'group',g,'counts',*[len(ix[k]) for k in ['train','validation','calibration','test']],'chosen_C',chosen,flush=True)
  assert set(foldmap)==set(subsets['test'])
  for i in subsets['test']:
   m=meta[i];b=bg[i];assert int(b['fold'])==fold
   A=float(b['p_fusion']);seq=float(b['p_seq']);struct=float(b['p_struct']);assert abs(A-(.72*seq+.28*struct))<2e-7
   B=foldmap[i];out.append({'sample_id':m['sample_id'],'sequence':m['sequence'],'group_id':m['group_id'],'fold':fold,'label':m['label'],'cys_count':m['cys_count'],'expert_group':group(m['cys_count']),'flag':bool(m['uniform_flag']),'A_probability':A,'B_probability':B,'C_probability':.5*A+.5*B})
  joblib.dump(saved,P/f'fold{fold}_experts.joblib',compress=3)
  fold_audit.append({'fold':fold,'train_validation_cal_test_sequence_group_disjoint':True,'selected_C':{g:saved[g]['C'] for g in saved},'n_test':len(subsets['test'])});save(P/'status.json',{'state':'running','completed_folds':fold+1,'elapsed_seconds':time.time()-start})
 assert len(out)==6387 and len({x['sample_id'] for x in out})==6387
 save(P/'predictions.json',out)
 def metric(rows,arm):
  y=np.array([x['label'] for x in rows]);p=np.array([x[arm+'_probability'] for x in rows]);t=.5 if arm=='B' else .47022256531284556;d=(p>=t).astype(int);tp=int(((y==1)&(d==1)).sum());tn=int(((y==0)&(d==0)).sum());fp=int(((y==0)&(d==1)).sum());fn=int(((y==1)&(d==0)).sum());two=len(set(y))==2
  return {'n':len(y),'TP':tp,'TN':tn,'FP':fp,'FN':fn,'errors':fp+fn,'MCC':float(matthews_corrcoef(y,d)) if two else None,'AUPRC':float(average_precision_score(y,p)) if two else None,'AUROC':float(roc_auc_score(y,p)) if two else None,'Brier':float(np.mean((p-y)**2))}
 panels={'all':lambda x:True,'low':lambda x:x['expert_group']=='low','middle':lambda x:x['expert_group']=='middle','high':lambda x:x['expert_group']=='high','low_positive':lambda x:x['expert_group']=='low' and x['label']==1,'high_negative':lambda x:x['expert_group']=='high' and x['label']==0,'flagged':lambda x:x['flag'],'unflagged_not_gold':lambda x:not x['flag']}
 stats={};comparisons={}
 for name,fn in panels.items():
  rows=[x for x in out if fn(x)];stats[name]={a:metric(rows,a) for a in 'ABC'};comparisons[name]={}
  for a in 'BC':
   ca=[int((x['A_probability']>=.47022256531284556)==x['label']) for x in rows];cb=[int((x[a+'_probability']>=(.5 if a=='B' else .47022256531284556))==x['label']) for x in rows]
   comparisons[name][a]={'corrected':sum(not i and j for i,j in zip(ca,cb)),'harmed':sum(i and not j for i,j in zip(ca,cb))}
 save(P/'metrics.json',stats);save(P/'comparisons.json',comparisons)
 foldmetrics={str(f):{a:metric([x for x in out if x['fold']==f],a) for a in 'ABC'} for f in range(5)};save(P/'fold_metrics.json',foldmetrics)
 rng=np.random.default_rng(20261028);buckets={}
 for row in out:buckets.setdefault(row['fold'],{}).setdefault(row['group_id'],[]).append(row)
 def cm(rr,a):
  x=np.zeros(4);t=.5 if a=='B' else .47022256531284556
  for z in rr:x[2*z['label']+int(z[a+'_probability']>=t)]+=1
  return x
 matrices={f:np.array([[cm(v,a) for a in 'ABC'] for v in g.values()]) for f,g in buckets.items()}
 boot=np.zeros((2000,3,4))
 for f,arr in matrices.items():boot+=(rng.multinomial(len(arr),np.full(len(arr),1/len(arr)),size=2000)@arr.reshape(len(arr),12)).reshape(2000,3,4)
 def mcc(a):
  tn,fp,fn,tp=[a[...,i] for i in range(4)];den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));return np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den),where=den>0)
 d=mcc(boot);intervals={k:np.quantile(d[:,j]-d[:,0],[.025,.975]).tolist() for j,k in [(1,'B_minus_A'),(2,'C_minus_A')]};save(P/'paired_bootstrap.json',intervals)
 save(P/'verification.json',{'passed':True,'n':len(out),'folds':fold_audit,'same_sequence_label_all_sources':True,'history_fusion_reconstructed':True,'historical_A_3seed_and_3D':True,'B_frozen_PLM_features_1d_only':True,'C_fixed_equal_probability':True,'no_Test1_Test2':True,'historical_global_calibration_not_nested':True,'not_external_blind':True,'different_thresholds':True})
 save(P/'status.json',{'state':'completed','elapsed_seconds':time.time()-start,'n':len(out)})
 print('ALL DONE',json.dumps({'all':stats['all'],'low_positive':stats['low_positive'],'high_negative':stats['high_negative'],'comparisons':comparisons['all'],'CI':intervals}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:save(P/'status.json',{'state':'failed','error':repr(e)});raise
