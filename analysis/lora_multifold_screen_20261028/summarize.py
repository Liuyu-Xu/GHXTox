"""Audit and summarize four prespecified folds plus historical fold0."""
import json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'analysis/repository_sync_20261026/validation_numpy1_clean'))
import numpy as np
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
manifest=read(P/'manifest.json');assert manifest['source_sha256']==hashlib.sha256((R/'src/ghxtox/lora_mamba_1d.py').read_bytes()).hexdigest();assert manifest['role_contract_sha256']==hashlib.sha256((R/'analysis/group_weight_trial_20261027/role_weight_contract.json').read_bytes()).hexdigest();assert manifest['protocol_sha256']==hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest()
roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json');meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json')
allrows=[];fstats=[];audit=[]
for f in range(5):
 q=R/'analysis/lora_training_validation_20261027' if f==0 else P/f'fold{f}'
 stat=read(q/'status.json');assert stat['state']=='completed'
 arms=['no_lora','lora_lr1e5','lora_lr3e5'];dev={a:read(q/'runs'/a/'result.json') for a in arms}
 assert stat['selected_lora_candidate_by_validation_only']==min(arms[1:],key=lambda a:dev[a]['validation_BCE']);sel=stat['selected_lora_candidate_by_validation_only']
 assert len({d['initial_head_sha256'] for d in dev.values()})==1
 for arm,d in dev.items():
  assert d['source_sha256']==manifest['source_sha256'];assert (q/'runs'/arm/'best_model.pt').exists()
  assert d['selected_epoch']==min(d['history'],key=lambda h:h['validation_unweighted_BCE'])['epoch']
  for h in d['history']:
   assert h['AMP_skipped_steps']==0 and h['successful_optimizer_steps']>0
   assert h['LoRA_B_tensors_updated']==(0 if arm=='no_lora' else 144)
   assert all(not len(dev[z]['history'])>=h['epoch'] or dev[z]['history'][h['epoch']-1]['order_sha256']==h['order_sha256'] for z in arms)
 pred={a:read(q/('predictions_'+a+'.json')) for a in arms};ids=[z['sample_id'] for z in pred['no_lora']]
 assert all([z['sample_id'] for z in pred[a]]==ids for a in arms)
 expected=[meta[x['source_index']]['sample_id'] for x in roles if x['outer_fold']==f and x['role']=='test'];assert ids==expected and len(set(ids))==len(ids)
 for i in range(len(ids)):
  x=pred['no_lora'][i];z=pred[sel][i];assert x['label']==z['label'] and x['group_id']==z['group_id']
  allrows.append({'fold':f,'sample_id':ids[i],'group_id':x['group_id'],'label':x['label'],'selected':sel,'seq_base':x['sequence_probability'],'seq_lora':z['sequence_probability'],'fusion_base':x['fusion_probability'],'fusion_lora':z['fusion_probability']})
 fstats.append({'fold':f,'sample_count':len(ids),'selected':sel,'validation_BCE':{a:dev[a]['validation_BCE'] for a in arms},'metrics':read(q/'evaluation_metrics.json')['metrics'],'comparison':read(q/'evaluation_metrics.json')['comparisons']})
 audit.append({'fold':f,'selected':sel,'head_initialization_same':True,'batch_orders_same':True,'nonzero_lora_B_per_epoch':144,'AMP_skips':0,'n':len(ids),'epochs':{a:len(dev[a]['history']) for a in arms}})
assert len(allrows)==6387 and len({x['sample_id'] for x in allrows})==6387
save(P/'all_fold_predictions.json',allrows);save(P/'fold_metrics.json',fstats)
def calc(rows,mode):
 y=np.array([x['label'] for x in rows]);threshold=.5 if mode=='seq' else .47022256531284556
 pb=np.array([x[mode+'_base'] for x in rows]);pt=np.array([x[mode+'_lora'] for x in rows]);b=(pb>=threshold).astype(int);t=(pt>=threshold).astype(int)
 def info(p,d):return {'n':len(y),'MCC':float(matthews_corrcoef(y,d)),'AUPRC':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),'errors':int((y!=d).sum()),'FP':int(((y==0)&(d==1)).sum()),'FN':int(((y==1)&(d==0)).sum())}
 g={}
 for i,row in enumerate(rows):g.setdefault((row['fold'],row['group_id']),[]).append(i)
 counts={}
 for (fold,group),ix in g.items():
  arr=np.zeros((2,4),dtype=np.int64)
  for arm,d in enumerate([b,t]):np.add.at(arr[arm],2*y[ix]+d[ix],1)
  counts.setdefault(fold,[]).append(arr)
 def mcc(x):
  tn,fp,fn,tp=[x[...,i] for i in range(4)];den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));return np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den,dtype=float),where=den>0)
 rng=np.random.default_rng(20261028);boot=np.zeros((2000,2,4))
 for v in counts.values():
  v=np.array(v);w=rng.multinomial(len(v),np.full(len(v),1/len(v)),size=2000);boot+=(w@v.reshape(len(v),8)).reshape(2000,2,4)
 delta=mcc(boot[:,1])-mcc(boot[:,0]);bb=info(pb,b);tt=info(pt,t)
 return {'base':bb,'lora':tt,'delta_MCC':tt['MCC']-bb['MCC'],'delta_95CI':np.quantile(delta,[.025,.975]).tolist(),'corrected':int(((b!=y)&(t==y)).sum()),'harmed':int(((b==y)&(t!=y)).sum())}
results={}
for name,rows in [('folds1to4',[x for x in allrows if x['fold']!=0]),('fold0',[x for x in allrows if x['fold']==0]),('all5',[x for x in allrows])]:results[name]={m:calc(rows,m) for m in ['seq','fusion']}
save(P/'pooled_metrics.json',results);save(P/'verification.json',{'passed':True,'fold_audit':audit,'total_unique_samples':6387,'primary_folds':[1,2,3,4],'first_fold_heldout_previously_inspected':True,'no_Test1_Test2':True,'CI_scope':'fixed predictions, paired sequence groups stratified by fold; excludes training/seed/selection uncertainty'})
lines=['# LoRA多折单种子初筛结果','','训练划分按序列组隔离；每折分别训练无LoRA及两档LoRA，按该折内部验证误差选择LoRA学习率。分组校准及留出评估随后进行。第一折先前已看过，故首要汇总其余四折；所有数据之前用于项目研究，不是全新外部盲测。','','|范围|输出|无LoRA MCC|选中的LoRA MCC|MCC差及95%区间|无LoRA错误|LoRA错误|纠正/添错|','|---|---|---:|---:|---|---:|---:|---|']
for name in ['folds1to4','fold0','all5']:
 for mode in ['seq','fusion']:
  z=results[name][mode];ci=z['delta_95CI'];lines.append(f"|{name}|{mode}|{z['base']['MCC']:.4f}|{z['lora']['MCC']:.4f}|{z['delta_MCC']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]|{z['base']['errors']}|{z['lora']['errors']}|{z['corrected']}/{z['harmed']}|")
lines+=['','条件融合使用本次单种子一维结果与历史固定三维OOF，不能当完整三种子模型或全流程独立融合。区间只反映这批固定预测按序列组重采样的变化，不包含多随机种子/训练波动。未使用Test1/Test2。','','每折结果与审计详见 fold_metrics.json、pooled_metrics.json、verification.json。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('COMPLETE SUMMARY',json.dumps(results['folds1to4'],ensure_ascii=False),flush=True)
