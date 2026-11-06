import json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
read=lambda p:json.loads(p.read_text(encoding='utf8'));meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');bg=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json');roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json'); rows={a:[] for a in 'ABCD'}; metas=[];ver=[]
for f in range(5):
 root=(R/f'analysis/autodl_cys_fold{f}_20261029/runs/fold{f}') if f<3 else P/f'runs/fold{f}';ids=[x['source_index'] for x in roles if x['outer_fold']==f and x['role']=='test'];res={a:read(root/a/'result.json') for a in 'ABCD'};assert len(set(res[a]['initial_head_sha256'] for a in 'ABCD'))==1
 for a in 'ABCD':
  z=read(root/a/'predictions.json');assert len(z)==len(ids) and all(x['sample_id']==meta[i]['sample_id'] and x['label']==meta[i]['label'] and x['group_id']==meta[i]['group_id'] and abs(x['fusion_probability']-(.72*x['sequence_probability']+.28*float(bg[i]['p_struct'])))<1e-9 for x,i in zip(z,ids));rows[a]+=z
 metas += [meta[i] for i in ids];ver.append({'fold':f,'n':len(ids),'init':res['A']['initial_head_sha256'],'epochs':{a:len(res[a]['history']) for a in 'ABCD'},'seed':{a:res[a]['seed'] for a in 'ABCD'}})
assert len(metas)==6387 and len({m['sample_id'] for m in metas})==6387
y=np.array([m['label'] for m in metas]);c=np.array([m['cys_count'] for m in metas]);flags=np.array([m['uniform_flag'] for m in metas],dtype=bool);g=np.array([str(m['fold'])+'|'+m['group_id'] for m in metas]);t=.47022256531284556;pred={a:np.array([r['fusion_probability'] for r in rows[a]]) for a in rows};d={a:pred[a]>=t for a in rows}
panels={'all':np.ones(len(y),dtype=bool),'low_pos':(c<=1)&(y==1),'low_neg':(c<=1)&(y==0),'high_pos':(c>=6)&(y==1),'high_neg':(c>=6)&(y==0),'mid':(c>=2)&(c<=5),'flagged':flags,'unflagged':~flags};metrics={}
for name,mask in panels.items():
 yy=y[mask];metrics[name]={}
 for a in rows:
  dd=d[a][mask];pp=pred[a][mask];metrics[name][a]={'n':int(mask.sum()),'errors':int((dd!=yy).sum()),'FN':int(((yy==1)&~dd).sum()),'FP':int(((yy==0)&dd).sum()),'MCC':float(matthews_corrcoef(yy,dd)) if len(np.unique(yy))==2 else None,'AUPRC':float(average_precision_score(yy,pp)) if len(np.unique(yy))==2 else None,'AUROC':float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None}
# resample group within fold, conditional on fixed predictions
rng=np.random.default_rng(20261029);bs=np.zeros((2000,4,4))
for f in range(5):
 ids=np.array([int(m['fold'])==f for m in metas]);uni,inv=np.unique(g[ids],return_inverse=True);ct=np.zeros((len(uni),4,4));yy=y[ids]
 for j,a in enumerate('ABCD'):
  for i,(v,dd) in enumerate(zip(yy,d[a][ids])):ct[inv[i],j,2*v+int(dd)]+=1
 bs+=(rng.multinomial(len(uni),np.full(len(uni),1/len(uni)),size=2000)@ct.reshape(len(uni),-1)).reshape(2000,4,4)
tn,fp,fn,tp=[bs[:,:,i] for i in range(4)];den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));m=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den),where=den>0);comp={}
for x,z in [('A','B'),('B','C'),('C','D'),('A','C'),('A','D')]:
 q=m[:,ord(z)-65]-m[:,ord(x)-65];comp[x+'->'+z]={'delta_MCC':metrics['all'][z]['MCC']-metrics['all'][x]['MCC'],'CI95':np.quantile(q,[.025,.975]).tolist(),'corrected':int(((d[x]!=y)&(d[z]==y)).sum()),'harmed':int(((d[x]==y)&(d[z]!=y)).sum())}
out={'metrics':metrics,'comparisons':comp,'verification':ver};(P/'pooled_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False,indent=2))
