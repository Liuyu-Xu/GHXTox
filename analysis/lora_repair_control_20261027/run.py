"""One fixed-fold actual1D repair-control A/B/C. No MCD/NDBP extra boost."""
import os,sys,pathlib,json,time,hashlib,gc,argparse,csv
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'));sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'analysis/group_weight_trial_20261027'))
os.environ['OMP_NUM_THREADS']='2';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from ghxtox.utils import set_seed
from run_sequence_trial import load_model
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(state):
 h=hashlib.sha256()
 for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def main(args):
 meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json')
 roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json')
 bg=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json')
 rr={role:[r['source_index'] for r in roles if r['outer_fold']==0 and r['role']==role] for role in ['train','validation','calibration','test']}
 groups={role:{meta[i]['group_id'] for i in ix} for role,ix in rr.items()}
 assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b)
 assert all(m['sample_id']==b['sample_id'] and m['sequence']==b['sequence'] and m['label']==int(b['label']) for m,b in zip(meta,bg))
 model,tok,cfg=load_model(42)
 trainable={n:p for n,p in model.named_parameters() if p.requires_grad};initial={n:p.detach().cpu().clone() for n,p in trainable.items()};initial_sha=digest(initial)
 pos=sum(meta[i]['label'] for i in rr['train']);pw=torch.tensor((len(rr['train'])-pos)/pos,device='cuda')
 save(P/'run_manifest.json',{'seed':42,'outer_fold':0,'config':cfg,'role_ids':{k:[meta[i]['sample_id'] for i in ix] for k,ix in rr.items()},'role_groups_disjoint':True,'initial_trainable_sha256':initial_sha,'trainable_parameter_count':sum(p.numel() for p in trainable.values()),'frozen_parameter_count':sum(p.numel() for p in model.parameters() if not p.requires_grad),'class_pos_weight':float(pw),'class_pos_weight_same_for_all_arms':True,'torch':torch.__version__,'cuda':torch.cuda.get_device_name(0),'source_sha256':hashlib.sha256((R/'src/ghxtox/lora_mamba_1d.py').read_bytes()).hexdigest(),'cold_start_from_pretrained_not_full_checkpoint':True})
 def batch(ix):
  enc=tok([' '.join(meta[i]['sequence']) for i in ix],padding=True,add_special_tokens=True,return_tensors='pt')
  assert enc['input_ids'].shape[1]==max(len(meta[i]['sequence']) for i in ix)+1
  return enc['input_ids'].cuda(),enc['attention_mask'].cuda(),torch.tensor([len(meta[i]['sequence']) for i in ix],device='cuda'),torch.tensor([meta[i]['label'] for i in ix],device='cuda',dtype=torch.float32)
 def predict(ix):
  model.eval();out=[]
  with torch.inference_mode():
   for k in range(0,len(ix),16):
    chosen=ix[k:k+16];ids,mask,lens,labels=batch(chosen)
    with torch.autocast(device_type='cuda',dtype=torch.float16):logits=model(ids,mask,lens)
    out += [{'source_index':i,'logit':float(x)} for i,x in zip(chosen,logits.cpu().tolist())]
  return out
 for arm in args.arms.split(','):
  assert arm in ['A','B','C'];folder=P/'runs'/arm
  if (folder/'result.json').exists():print('REUSE COMPLETE',arm,flush=True);continue
  model.load_state_dict(initial,strict=False);set_seed(42)
  model.plm.gradient_checkpointing_disable()
  model.plm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':arm=='A'})
  optimizer=torch.optim.AdamW(list(trainable.values()),lr=.0003,weight_decay=.01)
  scaler=torch.cuda.amp.GradScaler(init_scale=128.0)
  assert digest({n:p for n,p in trainable.items()})==initial_sha
  best=float('inf');best_state=None;best_epoch=None;history=[];started=time.time();torch.cuda.reset_peak_memory_stats()
  for epoch in range(1,16):
   gen=torch.Generator().manual_seed(42+epoch*10000)
   order=[rr['train'][i] for i in torch.randperm(len(rr['train']),generator=gen).tolist()]
   order_hash=hashlib.sha256(np.array(order,dtype=np.int64).tobytes()).hexdigest()
   model.train();loss_sum=0.0;first_gradient=None;steps=0
   for offset in range(0,len(order),16):
    ix=order[offset:offset+16];ids,mask,lens,labels=batch(ix)
    w=torch.tensor([1.5 if arm=='C' and meta[i]['label']==1 and meta[i]['cys_count']<4 else 1.0 for i in ix],device='cuda')
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda',dtype=torch.float16):
     logits=model(ids,mask,lens)
     per=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,pos_weight=pw,reduction='none');loss=(w*per).sum()/w.sum()
    assert torch.isfinite(loss),('nonfinite loss',arm,epoch,offset)
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    if offset==0:
     bs=[p for n,p in trainable.items() if '.lora_B' in n];present=sum(p.grad is not None for p in bs)
     assert present==(0 if arm=='A' else 144),(arm,present)
     assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
     first_gradient={'LoRA_B_tensors':len(bs),'LoRA_B_grad_present':present,'head_grad_present':model.classifier.weight.grad is not None}
    torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0)
    scaler.step(optimizer);scaler.update();loss_sum+=float(loss.item())*len(ix);steps+=1
   b_nonzero=sum(bool(p.count_nonzero()) for n,p in trainable.items() if '.lora_B' in n)
   assert b_nonzero==(0 if arm=='A' else 144),(arm,b_nonzero)
   val=predict(rr['validation']);vl=float(np.mean([np.logaddexp(0,r['logit'])-meta[r['source_index']]['label']*r['logit'] for r in val]))
   history.append({'epoch':epoch,'train_weighted_loss':loss_sum/len(order),'validation_unweighted_BCE':vl,'steps':steps,'order_sha256':order_hash,'first_batch_gradient':first_gradient,'LoRA_B_nonzero_at_epoch_end':b_nonzero,'elapsed_seconds':time.time()-started})
   if vl<best:best=vl;best_epoch=epoch;best_state={n:p.detach().cpu().clone() for n,p in trainable.items()}
   save(folder/'progress.json',{'arm':arm,'history':history,'selected_epoch_so_far':best_epoch})
   print(f'REPAIR_CONTROL arm={arm} epoch={epoch}/15 train={loss_sum/len(order):.4f} val={vl:.4f} seconds={time.time()-started:.1f}',flush=True)
  model.load_state_dict(best_state,strict=False)
  cal=predict(rr['calibration']);platt=LogisticRegression(C=1,max_iter=1000).fit(np.array([[x['logit']] for x in cal]),np.array([meta[x['source_index']]['label'] for x in cal]))
  test=predict(rr['test']);prob=platt.predict_proba(np.array([[x['logit']] for x in test]))[:,1]
  for r,p in zip(test,prob):
   i=r['source_index'];sp=float(bg[i]['p_struct']);fusion=.72*float(p)+.28*sp
   r.update(sample_id=meta[i]['sample_id'],label=meta[i]['label'],sequence_probability=float(p),sequence_prediction=int(p>=.5),frozen_structure_probability=sp,fusion_probability=fusion,fusion_prediction=int(fusion>=.47022256531284556))
  result={'arm':arm,'outer_fold':0,'seed':42,'initial_trainable_sha256':initial_sha,'history':history,'selected_epoch':best_epoch,'seconds':time.time()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'calibration_records':cal,'platt':{'slope':float(platt.coef_[0,0]),'intercept':float(platt.intercept_[0])},'predictions':test,'native_class_pos_weight':float(pw),'scope':'one-fold one-seed actual1D matched controls; frozen historical3D fusion is conditional, not public3seed ensemble/blind complete pipeline'}
  save(folder/'result.json',result)
  torch.save({'artifact_type':'research_lora_repair_control','arm':arm,'config':cfg,'model_state':best_state,'seed':42,'outer_fold':0,'training_ids':[meta[i]['sample_id'] for i in rr['train']]},folder/'best_model.pt')
  print('COMPLETE ARM',arm,'seconds',time.time()-started,flush=True)
  del optimizer,scaler,best_state;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arms',default='A,B');main(ap.parse_args())
