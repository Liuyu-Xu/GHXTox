"""Exploratory single-fold evaluation of all frozen development checkpoints.
Selection is already fixed solely from validation BCE. No external Test1/Test2.
"""
import sys,os,json,hashlib,gc
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[2]
sys.path[:0]=[str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'),str(R/'src'),str(R/'analysis/group_weight_trial_20261027')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='2'
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
from ghxtox.lora_mamba_1d import LoRALinear
from run_sequence_trial import load_model
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
 status=read(P/'status.json');assert status['state']=='completed' and not status['smoke_only']
 selected=status['selected_lora_candidate_by_validation_only'];arms=['no_lora','lora_lr1e5','lora_lr3e5']
 meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');role=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json');bg=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json')
 ix={k:[x['source_index'] for x in role if x['outer_fold']==3 and x['role']==k] for k in ['train','validation','calibration','test']}
 assert all(not {meta[i]['group_id'] for i in ix[a]}&{meta[i]['group_id'] for i in ix[b]} for a in ix for b in ix if a!=b)
 assert all(m['sample_id']==b['sample_id'] and m['sequence']==b['sequence'] and m['label']==int(b['label']) for m,b in zip(meta,bg))
 save(P/'evaluation_plan.json',{'status':'fixed before evaluation','selected_by_validation_only':selected,'arms_evaluated':arms,'calibration':'Platt on calibration only','sequence_threshold':0.5,'fusion':'0.72 sequence + 0.28 historical frozen 3D OOF','fusion_threshold':0.47022256531284556,'evaluation':'outerfold3 evaluation, dataset previously studied; not independent external validation','no_Test1_Test2':True,'metrics':'MCC,AUPRC,AUROC,FN,FP,corrected and harmed','bootstrap':'2000 group paired conditional fixed predictions, no model selection uncertainty','selection_rule':'do not select or tune using evaluation'})
 runs={}
 for arm in arms:
  model,tok,cfg=load_model(42)
  if arm=='no_lora':
   for n,m in list(model.plm.named_modules()):
    if isinstance(m,LoRALinear):
     parent=model.plm;parts=n.split('.')
     for part in parts[:-1]:parent=getattr(parent,part)
     setattr(parent,parts[-1],m.base)
   model.plm.gradient_checkpointing_disable()
  cp=torch.load(P/'runs'/arm/'best_model.pt',map_location='cpu',weights_only=True);state=cp['model_state']
  expected={n for n,p in model.named_parameters() if p.requires_grad};assert set(state)==expected
  model.load_state_dict(state,strict=False);model.eval()
  def predict(indices):
   arr=[]
   with torch.inference_mode():
    for k in range(0,len(indices),16):
     chosen=indices[k:k+16];enc=tok([' '.join(meta[i]['sequence']) for i in chosen],padding=True,add_special_tokens=True,return_tensors='pt');ids=enc['input_ids'].cuda();mask=enc['attention_mask'].cuda();lengths=torch.tensor([len(meta[i]['sequence']) for i in chosen],device='cuda')
     with torch.autocast('cuda',dtype=torch.float16):logits=model(ids,mask,lengths)
     arr.extend(logits.float().cpu().tolist())
   return np.array(arr,dtype=float)
  cal=predict(ix['calibration']);labels=np.array([meta[i]['label'] for i in ix['calibration']]);platt=LogisticRegression(C=1,max_iter=1000).fit(cal[:,None],labels)
  test=predict(ix['test']);prob=platt.predict_proba(test[:,None])[:,1];three=np.array([bg[i]['p_struct'] for i in ix['test']],dtype=float)
  y=np.array([meta[i]['label'] for i in ix['test']]);out=[]
  for j,i in enumerate(ix['test']):out.append({'sample_id':meta[i]['sample_id'],'group_id':meta[i]['group_id'],'label':int(y[j]),'sequence_probability':float(prob[j]),'fusion_probability':float(.72*prob[j]+.28*three[j])})
  runs[arm]=out;save(P/('predictions_'+arm+'.json'),out);print('evaluated',arm,flush=True)
  del model,tok,cp,state;gc.collect();torch.cuda.empty_cache()
 metrics={};decisions={}
 for mode,threshold in [('sequence',.5),('fusion',.47022256531284556)]:
  metrics[mode]={};decisions[mode]={}
  for arm,out in runs.items():
   p=np.array([r[mode+'_probability'] for r in out]);d=(p>=threshold).astype(int);decisions[mode][arm]=d
   metrics[mode][arm]={'n':len(y),'MCC':float(matthews_corrcoef(y,d)),'AUPRC':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),'errors':int((y!=d).sum()),'FP':int(((y==0)&(d==1)).sum()),'FN':int(((y==1)&(d==0)).sum())}
 groups=[x['group_id'] for x in runs['no_lora']];unique=list(dict.fromkeys(groups));indices={g:np.array([i for i,x in enumerate(groups) if x==g]) for g in unique};rng=np.random.default_rng(20261027);comparisons={}
 for mode in metrics:
  base=decisions[mode]['no_lora'];treat=decisions[mode][selected];boot=[]
  for _ in range(2000):
   choices=rng.integers(0,len(unique),size=len(unique));i=np.concatenate([indices[unique[j]] for j in choices]);boot.append(matthews_corrcoef(y[i],treat[i])-matthews_corrcoef(y[i],base[i]))
  comparisons[mode]={'comparison':'selected LoRA vs no LoRA','selected':selected,'delta_MCC':metrics[mode][selected]['MCC']-metrics[mode]['no_lora']['MCC'],'bootstrap_95CI':np.quantile(boot,[.025,.975]).tolist(),'corrected':int(((base!=y)&(treat==y)).sum()),'harmed':int(((base==y)&(treat!=y)).sum())}
 save(P/'evaluation_metrics.json',{'selected_by_validation_only':selected,'metrics':metrics,'comparisons':comparisons})
 print(json.dumps({'selected':selected,'metrics':metrics,'comparisons':comparisons},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
