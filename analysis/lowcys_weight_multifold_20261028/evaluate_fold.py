"""Evaluate one frozen B model on disjoint fold calibration and test roles."""
import sys,os,json,gc,argparse
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path[:0]=[str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'),str(R/'src'),str(R/'analysis/group_weight_trial_20261027')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='2'
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from ghxtox.lora_mamba_1d import remove_inactive_lora,LoRALinear
from run_sequence_trial import load_model
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def main(fold):
 f=P/f'fold{fold}';status=read(f/'status.json');assert status['state']=='completed' and not status['smoke_only']
 meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json');bg=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json')
 ix={k:[x['source_index'] for x in roles if x['outer_fold']==fold and x['role']==k] for k in ['train','validation','calibration','test']}
 assert all(not {meta[i]['group_id'] for i in ix[a]}&{meta[i]['group_id'] for i in ix[b]} for a in ix for b in ix if a!=b)
 assert all(meta[i]['sample_id']==bg[i]['sample_id'] and meta[i]['sequence']==bg[i]['sequence'] and meta[i]['label']==int(bg[i]['label']) and int(bg[i]['fold'])==fold for i in ix['test'])
 model,tok,cfg=load_model(42);remove_inactive_lora(model.plm.encoder);model.plm.gradient_checkpointing_disable()
 cp=torch.load(f/'runs/low_cys/best_model.pt',map_location='cpu',weights_only=True);state=cp['model_state'];assert cp['outer_fold']==fold and set(state)=={n for n,p in model.named_parameters() if p.requires_grad};model.load_state_dict(state,strict=False);model.eval();assert not any(isinstance(m,LoRALinear) for m in model.modules())
 def predict(index):
  out=[]
  with torch.inference_mode():
   for k in range(0,len(index),16):
    ids=index[k:k+16];enc=tok([' '.join(meta[i]['sequence']) for i in ids],padding=True,add_special_tokens=True,return_tensors='pt');tokens=enc['input_ids'].cuda();mask=enc['attention_mask'].cuda();lens=torch.tensor([len(meta[i]['sequence']) for i in ids],device='cuda')
    with torch.autocast('cuda',dtype=torch.float16):logits=model(tokens,mask,lens)
    out.extend(logits.float().cpu().tolist())
  return np.array(out,dtype=float)
 cal=predict(ix['calibration']);labels=np.array([meta[i]['label'] for i in ix['calibration']]);platt=LogisticRegression(C=1,max_iter=1000).fit(cal[:,None],labels)
 pred=predict(ix['test']);prob=platt.predict_proba(pred[:,None])[:,1];out=[]
 for j,i in enumerate(ix['test']):
  m=meta[i];out.append({'sample_id':m['sample_id'],'group_id':m['group_id'],'label':m['label'],'sequence_probability':float(prob[j]),'fusion_probability':float(.72*prob[j]+.28*float(bg[i]['p_struct']))})
 save(f/'predictions_B.json',out);save(f/'evaluation_status.json',{'completed':True,'fold':fold,'n_calibration':len(cal),'n_test':len(pred),'platt_slope':float(platt.coef_[0,0]),'platt_intercept':float(platt.intercept_[0]),'no_Test1_Test2':True,'no_outer_labels_used_for_choice':True})
 print('EVALUATED fold',fold,'n',len(out),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fold',required=True,type=int,choices=[0,1,2,4]);main(ap.parse_args().fold)
