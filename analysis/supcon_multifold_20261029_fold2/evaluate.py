"""Post-training exploratory one-fold paired evaluation; no Test1/Test2."""
import os,sys,json,gc
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path[:0]=[str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'),str(R/'src'),str(R/'analysis/group_weight_trial_20261027')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='2'
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
from ghxtox.lora_mamba_1d import remove_inactive_lora,LoRALinear
from run_sequence_trial import load_model
from run_trial import sample_weight
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json');bg=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json')
ix={k:[x['source_index'] for x in roles if x['outer_fold']==2 and x['role']==k] for k in ['train','validation','calibration','test']}
assert all(not {meta[i]['group_id'] for i in ix[a]}&{meta[i]['group_id'] for i in ix[b]} for a in ix for b in ix if a!=b)
assert all(m['sample_id']==b['sample_id'] and m['sequence']==b['sequence'] and m['label']==int(b['label']) for m,b in zip(meta,bg))
status=read(P/'status.json');assert status['state']=='completed' and not status['smoke_only']
prev=R/'analysis/lora_multifold_screen_20261028/fold2';base=read(prev/'predictions_no_lora.json');assert [m['sample_id'] for m in base]==[meta[i]['sample_id'] for i in ix['test']]
runs={'A':base}
for arm,key in [('supcon_all','B'),('supcon_unflagged','C')]:
 model,tok,cfg=load_model(42);remove_inactive_lora(model.plm.encoder);model.plm.gradient_checkpointing_disable()
 cp=torch.load(P/'runs'/arm/'best_model.pt',map_location='cpu',weights_only=True);state=cp['model_state'];assert set(state)=={n for n,p in model.named_parameters() if p.requires_grad};model.load_state_dict(state,strict=False);model.eval();assert not any(isinstance(m,LoRALinear) for m in model.modules())
 def predict(ids):
  out=[]
  with torch.inference_mode():
   for k in range(0,len(ids),16):
    z=ids[k:k+16];enc=tok([' '.join(meta[i]['sequence']) for i in z],padding=True,add_special_tokens=True,return_tensors='pt');tokens=enc['input_ids'].cuda();mask=enc['attention_mask'].cuda();lens=torch.tensor([len(meta[i]['sequence']) for i in z],device='cuda')
    with torch.autocast('cuda',dtype=torch.float16):logits=model(tokens,mask,lens)
    out.extend(logits.float().cpu().tolist())
  return np.array(out,dtype=float)
 cal=predict(ix['calibration']);cy=np.array([meta[i]['label'] for i in ix['calibration']]);platt=LogisticRegression(C=1,max_iter=1000).fit(cal[:,None],cy)
 test=predict(ix['test']);prob=platt.predict_proba(test[:,None])[:,1];out=[]
 for j,i in enumerate(ix['test']):
  m=meta[i];out.append({'sample_id':m['sample_id'],'group_id':m['group_id'],'label':m['label'],'sequence_probability':float(prob[j]),'fusion_probability':float(.72*prob[j]+.28*float(bg[i]['p_struct']))})
 runs[key]=out;save(P/('predictions_'+key+'.json'),out);print('evaluated',key,flush=True)
 del model,tok,cp,state;gc.collect();torch.cuda.empty_cache()
y=np.array([meta[i]['label'] for i in ix['test']]);groups=np.array([meta[i]['group_id'] for i in ix['test']]);ids=[meta[i]['sample_id'] for i in ix['test']]
assert all([x['sample_id'] for x in v]==ids for v in runs.values())
parents=['belongs to the mcd family','belongs to the non-disulfide-bridged peptide (ndbp) superfamily'];family=lambda m:sample_weight(m,'C')==2
panels={'all':list(range(len(y))),'target':[j for j,i in enumerate(ix['test']) if family(meta[i])],'lowcys_positive':[j for j,i in enumerate(ix['test']) if meta[i]['label']==1 and meta[i]['cys_count']<4],'flagged':[j for j,i in enumerate(ix['test']) if meta[i]['uniform_flag']],'unflagged_not_gold':[j for j,i in enumerate(ix['test']) if not meta[i]['uniform_flag']],'negative':[j for j,i in enumerate(ix['test']) if meta[i]['label']==0],'other_positive':[j for j,i in enumerate(ix['test']) if meta[i]['label']==1 and not family(meta[i])]}
for family_name,title in [('belongs to the cationic peptide 04 (cupiennin) family','Cupiennin'),('belongs to the msdin fungal toxin family','MSDIN')]:panels[title]=[j for j,i in enumerate(ix['test']) if meta[i]['label']==1 and family_name in meta[i]['current_family_sidecar']['parent_families']]
metrics={};comparison={};decisions={}
for mode,t in [('sequence',.5),('fusion',.47022256531284556)]:
 metrics[mode]={};decisions[mode]={}
 for arm,out in runs.items():
  p=np.array([x[mode+'_probability'] for x in out]);d=(p>=t).astype(int);decisions[mode][arm]=d;metrics[mode][arm]={}
  for label,ii in panels.items():
   yy=y[ii];dd=d[ii];pp=p[ii];both=len(set(yy.tolist()))==2
   metrics[mode][arm][label]={'n':len(ii),'groups':len(set(groups[ii].tolist())),'errors':int((yy!=dd).sum()),'FN':int(((yy==1)&(dd==0)).sum()),'FP':int(((yy==0)&(dd==1)).sum()),'MCC':float(matthews_corrcoef(yy,dd)) if both else None,'AUPRC':float(average_precision_score(yy,pp)) if both else None,'AUROC':float(roc_auc_score(yy,pp)) if both else None}
 comparison[mode]=[]
 unique=list(dict.fromkeys(groups.tolist()));indices={g:np.flatnonzero(groups==g) for g in unique}
 for a,b in [('A','B'),('A','C'),('B','C')]:
  aa=decisions[mode][a];bb=decisions[mode][b];rng=np.random.default_rng(20261028);delta=[]
  for _ in range(2000):
   ss=rng.integers(0,len(unique),len(unique));ii=np.concatenate([indices[unique[k]] for k in ss]);delta.append(matthews_corrcoef(y[ii],bb[ii])-matthews_corrcoef(y[ii],aa[ii]))
  comparison[mode].append({'comparison':a+'->'+b,'delta_MCC':metrics[mode][b]['all']['MCC']-metrics[mode][a]['all']['MCC'],'CI95_group_bootstrap':np.quantile(delta,[.025,.975]).tolist(),'corrected':int(((aa!=y)&(bb==y)).sum()),'harmed':int(((aa==y)&(bb!=y)).sum())})
save(P/'metrics.json',metrics);save(P/'comparisons.json',comparison)
ids_target=[ids[j] for j in panels['target']];samples=[]
for j in panels['target']:
 m=meta[ix['test'][j]];samples.append({'sample_id':ids[j],'group_id':m['group_id'],'parent_family':m['current_family_sidecar']['parent_families'],'uniform_flag':m['uniform_flag'],'probabilities':{a:{mode:runs[a][j][mode+'_probability'] for mode in ['sequence','fusion']} for a in runs}})
save(P/'target_samples.json',samples)
base_result=read(prev/'runs/no_lora/result.json');new={a:read(P/'runs'/name/'result.json') for a,name in [('B','supcon_all'),('C','supcon_unflagged')]}
for arm,v in new.items():
 assert v['initial_head_sha256']==base_result['initial_head_sha256']
 for h in v['history']:
  assert h['AMP_skipped_steps']==0 and h['LoRA_B_tensors_updated']==0
  if h['epoch']<=len(base_result['history']):assert h['order_sha256']==base_result['history'][h['epoch']-1]['order_sha256']
verification={'passed':True,'fold':2,'seed':42,'same_initial_head_and_batch_orders':True,'n_test':len(y),'target_count':len(ids_target),'groups_disjoint':True,'no_lora_all_arms':True,'no_amp_skips':True,'fixed_historical_3D_conditional':True,'no_Test1_Test2':True,'fold_and_targets_chosen_after_previous_error_observation':True,'not_blind_not_gold':True}
save(P/'verification.json',verification)
lines=['# 监督对比学习：首折探索结果','','复用fold2无LoRA基线A，B为分类加监督对比，C为相同对比但对比项排除旧疑点样本（分类项保留全部）。按原标签构对、训练批次内不同序列组，lambda0.1、温度0.1预先固定；非盲评、仅单种子。无旧标记不代表可靠金标准。','','|输出|方案|MCC|AUPRC|总错误|目标家族漏判/总数|其他原正类错误|原负类误报|','|---|---|---:|---:|---:|---:|---:|---:|']
for mode in ['sequence','fusion']:
 for arm in ['A','B','C']:
  z=metrics[mode][arm];lines.append(f"|{mode}|{arm}|{z['all']['MCC']:.4f}|{z['all']['AUPRC']:.4f}|{z['all']['errors']}|{z['target']['FN']}/{z['target']['n']}|{z['other_positive']['errors']}|{z['negative']['FP']}|")
lines+=['','配对比较：']
for mode in ['sequence','fusion']:
 for c in comparison[mode]:
  ci=c['CI95_group_bootstrap'];lines.append(f"- {mode} {c['comparison']}：MCC差{c['delta_MCC']:+.4f}，95%区间[{ci[0]:+.4f}, {ci[1]:+.4f}]，纠{c['corrected']}条、添{c['harmed']}条。")
lines+=['','区间只反映该折固定预测的按组重采样，不包含训练/种子/选择不确定性。目标只有4条，不能据此确认家族整体收益或生物学机制；已看过的fold2只适合探索。融合是单个一维新模型加历史固定三维OOF，不是论文三种子完整模型。未使用Test1/Test2。','', '分层与每条目标见 metrics.json、target_samples.json、verification.json。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('COMPLETE',json.dumps({'metrics':{m:{a:x['all'] for a,x in v.items()} for m,v in metrics.items()},'comparisons':comparison},ensure_ascii=False),flush=True)
