"""Independent matched-fold audit and paired OOF analysis."""
import json,hashlib,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
import numpy as np
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
manifest=read(P/'manifest.json');assert manifest['source_sha256']==hashlib.sha256((R/'src/ghxtox/lora_mamba_1d.py').read_bytes()).hexdigest();assert manifest['role_sha256']==hashlib.sha256((R/'analysis/group_weight_trial_20261027/role_weight_contract.json').read_bytes()).hexdigest();assert manifest['protocol_sha256']==hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest()
meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json');rows=[];audit=[]
for fold in range(5):
 base=R/('analysis/lora_training_validation_20261027' if fold==0 else f'analysis/lora_multifold_screen_20261028/fold{fold}');f=R/'analysis/family_weight_no_lora_20261028' if fold==3 else P/f'fold{fold}'
 status=read(f/'status.json');assert status['state']=='completed' and not status['smoke_only'];base_result=read(base/'runs/no_lora/result.json');weighted=read(f/'runs/low_cys/result.json')
 assert weighted['initial_head_sha256']==base_result['initial_head_sha256'];assert weighted['source_sha256']==manifest['source_sha256']
 assert weighted['selected_epoch']==min(weighted['history'],key=lambda x:x['validation_unweighted_BCE'])['epoch']
 for h in weighted['history']:
  assert h['AMP_skipped_steps']==0 and h['successful_optimizer_steps']>0 and h['LoRA_B_tensors_updated']==0
  if h['epoch']<=len(base_result['history']):assert h['order_sha256']==base_result['history'][h['epoch']-1]['order_sha256']
 ids={k:[x['source_index'] for x in roles if x['outer_fold']==fold and x['role']==k] for k in ['train','validation','calibration','test']}
 assert all(not {meta[i]['group_id'] for i in ids[a]}&{meta[i]['group_id'] for i in ids[b]} for a in ids for b in ids if a!=b)
 A=read(base/'predictions_no_lora.json');B=read(f/'predictions_B.json');expected=[meta[i]['sample_id'] for i in ids['test']];assert [x['sample_id'] for x in A]==expected==[x['sample_id'] for x in B]
 for i,a,b in zip(ids['test'],A,B):
  m=meta[i];assert a['label']==b['label']==m['label'] and a['group_id']==b['group_id']==m['group_id']
  parents=m['current_family_sidecar']['parent_families'];target=len(parents)==1 and parents[0] in ['belongs to the mcd family','belongs to the non-disulfide-bridged peptide (ndbp) superfamily'] and m['label']==1 and m['cys_count']<4
  rows.append({'fold':fold,'sample_id':m['sample_id'],'group_id':m['group_id'],'label':m['label'],'cys_count':m['cys_count'],'uniform_flag':bool(m['uniform_flag']),'target':target,'sequence_A':a['sequence_probability'],'sequence_B':b['sequence_probability'],'fusion_A':a['fusion_probability'],'fusion_B':b['fusion_probability']})
 train=[meta[i] for i in ids['train'] if meta[i]['label']==1 and meta[i]['cys_count']<4]
 audit.append({'fold':fold,'test_n':len(expected),'same_initial_head_and_order':True,'no_lora':True,'amp_skips':0,'weighted_train_n':len(train),'weighted_train_flagged':sum(bool(x['uniform_flag']) for x in train),'weighted_epochs':len(weighted['history']),'base_epochs':len(base_result['history'])})
assert len(rows)==6387 and len({x['sample_id'] for x in rows})==6387
save(P/'all_oof_predictions.json',rows)
def stats(items,mode):
 y=np.array([r['label'] for r in items]);p=np.array([r[mode] for r in items]);t=.5 if mode.startswith('sequence') else .47022256531284556;d=(p>=t).astype(int);both=len(np.unique(y))==2
 tp=int(((y==1)&(d==1)).sum());tn=int(((y==0)&(d==0)).sum());fp=int(((y==0)&(d==1)).sum());fn=int(((y==1)&(d==0)).sum())
 return {'n':len(y),'groups':len({(r['fold'],r['group_id']) for r in items}),'TP':tp,'TN':tn,'FP':fp,'FN':fn,'errors':fp+fn,'sensitivity':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'MCC':float(matthews_corrcoef(y,d)) if both else None,'AUPRC':float(average_precision_score(y,p)) if both else None,'AUROC':float(roc_auc_score(y,p)) if both else None}
def compare(items,kind,bootstrap=False):
 A=stats(items,kind+'_A');B=stats(items,kind+'_B');y=np.array([r['label'] for r in items]);pa=np.array([r[kind+'_A'] for r in items]);pb=np.array([r[kind+'_B'] for r in items]);t=.5 if kind=='sequence' else .47022256531284556;da=(pa>=t).astype(int);db=(pb>=t).astype(int)
 out={'A':A,'B':B,'delta_MCC':B['MCC']-A['MCC'] if A['MCC'] is not None else None,'delta_AUPRC':B['AUPRC']-A['AUPRC'] if A['AUPRC'] is not None else None,'corrected':int(((da!=y)&(db==y)).sum()),'harmed':int(((da==y)&(db!=y)).sum())}
 if bootstrap:
  buckets={}
  for i,r in enumerate(items):buckets.setdefault(r['fold'],{}).setdefault(r['group_id'],[]).append(i)
  rng=np.random.default_rng(20261028);dm=[];dp=[]
  for _ in range(2000):
   ii=[]
   for group in buckets.values():
    v=list(group.values());ii.extend(j for k in rng.integers(0,len(v),len(v)) for j in v[k])
   yy=y[ii]
   if len(np.unique(yy))<2:continue
   dm.append(matthews_corrcoef(yy,db[ii])-matthews_corrcoef(yy,da[ii]));dp.append(average_precision_score(yy,pb[ii])-average_precision_score(yy,pa[ii]))
  out['MCC_difference_95CI']=np.quantile(dm,[.025,.975]).tolist();out['AUPRC_difference_95CI']=np.quantile(dp,[.025,.975]).tolist()
 return out
panels={'all':lambda x:True,'lowcys_positive':lambda x:x['label']==1 and x['cys_count']<4,'highcys_positive':lambda x:x['label']==1 and x['cys_count']>=4,'lowcys_negative':lambda x:x['label']==0 and x['cys_count']<4,'highcys_negative':lambda x:x['label']==0 and x['cys_count']>=4,'flagged':lambda x:x['uniform_flag'],'unflagged_not_gold':lambda x:not x['uniform_flag'],'MCD_NDBP':lambda x:x['target']}
scopes={'four_followup_folds':[x for x in rows if x['fold']!=3],'prior_selected_fold3':[x for x in rows if x['fold']==3],'all_five_folds':rows,**{f'fold{f}':[x for x in rows if x['fold']==f] for f in range(5)}}
result={}
for name,data in scopes.items():
 result[name]={}
 for panel,fn in panels.items():
  subset=[r for r in data if fn(r)]
  if subset:result[name][panel]={mode:compare(subset,mode,bootstrap=(panel=='all' and name in ['four_followup_folds','all_five_folds'])) for mode in ['sequence','fusion']}
save(P/'metrics.json',result);save(P/'verification.json',{'passed':True,'folds':audit,'unique_oof_samples':len(rows),'primary_followup_folds':[0,1,2,4],'previously_inspected_fold3_reported_separately':True,'not_blind_external':True,'three_seed_full_ensemble_not_trained':True,'Test1_Test2_unused':True,'baseline_source_hash_diff':'previous baseline before LoRA removal compatibility change; same initial head and epoch order, same no-LoRA train path, settings and roles verified','confidence_interval_scope':'paired group bootstrap fixed predictions, grouped within folds; excludes training and selection uncertainty'})
labels={'four_followup_folds':'其余4折','prior_selected_fold3':'已看过的第3折','all_five_folds':'5折合计'}
lines=['# 低Cys加权多折验证','','无LoRA；每折A为复用的无额外加权模型，B为低Cys原正类1.5倍加权模型；各折同分组角色、初始化、批次顺序、优化器和早停规则。首要汇总此前单折之外的0/1/2/4折；第3折单独列出。','','|范围|输出|样本|A MCC|B MCC|A错|B错|A漏判|B漏判|AUPRC A/B|纠错/添错|','|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
for name in labels:
 for mode in ['sequence','fusion']:
  q=result[name]['all'][mode];a=q['A'];b=q['B'];lines.append(f"|{labels[name]}|{mode}|{a['n']}|{a['MCC']:.4f}|{b['MCC']:.4f}|{a['errors']}|{b['errors']}|{a['FN']}|{b['FN']}|{a['AUPRC']:.4f}/{b['AUPRC']:.4f}|{q['corrected']}/{q['harmed']}|")
lines+=['','首要4折配对区间：']
for mode in ['sequence','fusion']:
 q=result['four_followup_folds']['all'][mode];c=q['MCC_difference_95CI'];a=q['AUPRC_difference_95CI'];lines.append(f"- {mode}: ΔMCC={q['delta_MCC']:+.4f}, 95%区间[{c[0]:+.4f}, {c[1]:+.4f}]；ΔAUPRC={q['delta_AUPRC']:+.4f}, 95%区间[{a[0]:+.4f}, {a[1]:+.4f}]。")
lines+=['','带疑点和未标记样本分组、低/高Cys原正负类及各折结果见metrics.json；训练中带疑点的加权样本数见verification.json。未标记不等于标签可靠，疑点不等于错误。','','各折原数据被项目长期使用，不是新外部盲测；区间为固定预测按序列组重采样，不包含训练种子/模型选择波动。仅一维单种子；条件融合使用固定历史三维OOF，不是完整三种子模型。未使用Test1/Test2。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('SUMMARY DONE',json.dumps(result['four_followup_folds']['all'],ensure_ascii=False),flush=True)
