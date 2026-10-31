"""Development-only LoRA optimization, no held-out/test predictions."""
import os,sys,json,time,hashlib,gc,argparse
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path[:0]=[str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'),str(R/'src'),str(R/'analysis/group_weight_trial_20261027')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='2'
import numpy as np
import torch
from ghxtox.utils import set_seed
from ghxtox.lora_mamba_1d import LoRALinear,make_training_optimizer,check_lora_gradients
from run_sequence_trial import load_model
def supcon(features,labels,groups,eligible,temp):
 z=torch.nn.functional.normalize(features.float(),dim=1);logits=z@z.T/temp
 n=len(labels);different=torch.tensor([[groups[i]!=groups[j] for j in range(n)] for i in range(n)],device=labels.device,dtype=torch.bool)
 valid=different & eligible[:,None] & eligible[None,:]
 positive=valid & (labels[:,None]==labels[None,:]);negative=valid & (labels[:,None]!=labels[None,:])
 anchors=(positive.sum(1)>0)&(negative.sum(1)>0)
 if not anchors.any():return features.sum()*0,0
 logits=logits.masked_fill(~valid,float('-inf'));logden=torch.logsumexp(logits[anchors],dim=1)
 lp=logits[anchors]-logden[:,None];pos=positive[anchors];safe=torch.where(pos,lp,torch.zeros_like(lp))
 return -(safe.sum(1)/pos.sum(1)).mean(),int(anchors.sum())
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def digest(state):
 h=hashlib.sha256()
 for n,p in state.items():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def main(smoke=False):
 protocol=read(P/'protocol.json');rows=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json')
 ix={role:[x['source_index'] for x in roles if x['outer_fold']==2 and x['role']==role] for role in ['train','validation','calibration','test']}
 groups={k:{rows[i]['group_id'] for i in v} for k,v in ix.items()};assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b)
 pos=sum(rows[i]['label'] for i in ix['train']);pw=torch.tensor((len(ix['train'])-pos)/pos,device='cuda')
 destination=P/('smoke' if smoke else 'runs');head_hash=None;results={};started=time.time()
 save(P/('smoke_status.json' if smoke else 'status.json'),{'state':'running','completed_arms':[],'protocol':protocol,'pid':os.getpid()})
 for arm,ac in protocol['arms'].items():
  folder=destination/arm
  if (folder/'result.json').exists():
   old=read(folder/'result.json');assert (folder/'best_model.pt').is_file();assert old['source_sha256']==hashlib.sha256((R/'src/ghxtox/lora_mamba_1d.py').read_bytes()).hexdigest();assert head_hash is None or head_hash==old['initial_head_sha256'];results[arm]=old;head_hash=old['initial_head_sha256'];continue
  model,tok,cfg=load_model(42)
  initial_head={n:p for n,p in model.named_parameters() if p.requires_grad and '.lora_' not in n};sha=digest(initial_head)
  if head_hash is None:head_hash=read(R/'analysis/lora_multifold_screen_20261028/fold2/runs/no_lora/result.json')['initial_head_sha256']
  assert sha==head_hash
  cfg.update(learning_rate=protocol['head_lr'],lora_learning_rate=ac.get('lora_lr') or protocol['head_lr'])
  def batch(chosen):
   enc=tok([' '.join(rows[i]['sequence']) for i in chosen],padding=True,add_special_tokens=True,return_tensors='pt')
   return enc['input_ids'].cuda(),enc['attention_mask'].cuda(),torch.tensor([len(rows[i]['sequence']) for i in chosen],device='cuda'),torch.tensor([rows[i]['label'] for i in chosen],device='cuda',dtype=torch.float32)
  if arm in ('supcon_all','supcon_unflagged'):
   model.eval();ids,mask,lens,y=batch(ix['train'][:2])
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):before=model(ids,mask,lens)
   for name,m in list(model.plm.named_modules()):
    if isinstance(m,LoRALinear):
     parent=model.plm;parts=name.split('.')
     for part in parts[:-1]:parent=getattr(parent,part)
     setattr(parent,parts[-1],m.base)
   model.plm.gradient_checkpointing_disable();cfg['lora_rank']=0
   assert not any('.lora_' in n for n,p in model.named_parameters())
   assert all(not p.requires_grad for p in model.plm.parameters())
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):after=model(ids,mask,lens)
   torch.testing.assert_close(before,after,rtol=0,atol=0)
  features={}
  hook=model.pool_norm.register_forward_hook(lambda m,i,o:features.__setitem__('pooled',o))
  optimizer=make_training_optimizer(model,cfg);params=[p for p in model.parameters() if p.requires_grad];scaler=torch.cuda.amp.GradScaler(init_scale=128.)
  best=float('inf');best_epoch=None;stale=0;history=[];best_state=None
  epoch_limit=1 if smoke else protocol['max_epochs']
  for epoch in range(1,epoch_limit+1):
   set_seed(42+epoch*10000);gen=torch.Generator().manual_seed(42+epoch*10000)
   order=[ix['train'][i] for i in torch.randperm(len(ix['train']),generator=gen).tolist()]
   if smoke:order=order[:32]
   order_sha=hashlib.sha256(np.array(order,dtype=np.int64).tobytes()).hexdigest()
   if not smoke:
    original=read(R/'analysis/lora_multifold_screen_20261028/fold2/runs/no_lora/result.json')['history']
    if epoch<=len(original):assert order_sha==original[epoch-1]['order_sha256']
   adapters={n:p.detach().clone() for n,p in model.named_parameters() if '.lora_B' in n}
   model.train();loss_sum=0.;successful=0;skipped=0;grad_audit=None;anchor_total=0;contrast_total=0.
   for offset in range(0,len(order),16):
    chosen=order[offset:offset+16];ids,mask,lens,y=batch(chosen);optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
     logits=model(ids,mask,lens);base_loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=pw);eligible=torch.tensor([arm=='supcon_all' or not rows[i]['uniform_flag'] for i in chosen],device='cuda',dtype=torch.bool);cl,nanchors=supcon(features['pooled'],y,[rows[i]['group_id'] for i in chosen],eligible,protocol['temperature']);loss=base_loss+protocol['lambda']*cl
    anchor_total+=nanchors;contrast_total+=float(cl.detach())
    assert torch.isfinite(loss)
    scaler.scale(loss).backward();scaler.unscale_(optimizer);audit=check_lora_gradients(model)
    if grad_audit is None:grad_audit=audit
    norm=torch.nn.utils.clip_grad_norm_(params,1.);finite=bool(torch.isfinite(norm));scaler.step(optimizer);scaler.update()
    successful+=int(finite);skipped+=int(not finite);loss_sum+=float(loss)*len(chosen)
   current=dict(model.named_parameters());updated=sum(not torch.equal(old,current[n].detach()) for n,old in adapters.items())
   if adapters:assert updated==len(adapters),(arm,updated,len(adapters))
   model.eval();val_sum=0.;val_count=0
   val_ids=ix['validation'][:32] if smoke else ix['validation']
   with torch.inference_mode():
    for k in range(0,len(val_ids),16):
     chosen=val_ids[k:k+16];ids,mask,lens,y=batch(chosen)
     with torch.autocast('cuda',dtype=torch.float16):logits=model(ids,mask,lens)
     val_sum+=float(torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),y,reduction='sum'));val_count+=len(chosen)
   vb=val_sum/val_count
   history.append({'epoch':epoch,'train_total_loss':loss_sum/len(order),'contrastive_anchors':anchor_total,'contrastive_loss_sum':contrast_total,'validation_unweighted_BCE':vb,'successful_optimizer_steps':successful,'AMP_skipped_steps':skipped,'order_sha256':order_sha,'gradient_audit':grad_audit,'LoRA_B_tensors_updated':updated,'seconds_since_start':time.time()-started})
   if vb<best:
    best=vb;best_epoch=epoch;stale=0;best_state={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
   else:stale+=1
   save(folder/'progress.json',{'arm':arm,'history':history,'selected_epoch':best_epoch})
   print(f'{arm} epoch={epoch} validation_BCE={vb:.6f} successful={successful} skipped={skipped} updated_B={updated}',flush=True)
   if stale>=protocol['patience']:break
  result={'arm':arm,'smoke_only':smoke,'history':history,'selected_epoch':best_epoch,'validation_BCE':best,'initial_head_sha256':sha,'optimizer_groups':[{'name':g['name'],'lr':g['lr'],'parameter_count':sum(p.numel() for p in g['params'])} for g in optimizer.param_groups],'config':cfg,'roles_used':['train','validation'],'no_outer_evaluation_predictions':True,'source_sha256':hashlib.sha256((R/'src/ghxtox/lora_mamba_1d.py').read_bytes()).hexdigest()}
  if not smoke:torch.save({'artifact_type':'research_lora_development','model_state':best_state,'config':cfg,'arm':arm,'seed':42,'outer_fold':2},folder/'best_model.pt')
  save(folder/'result.json',result);results[arm]=result
  save(P/('smoke_status.json' if smoke else 'status.json'),{'state':'running','completed_arms':list(results),'pid':os.getpid()})
  del model,tok,optimizer,scaler,best_state,initial_head,params,current,adapters;gc.collect();torch.cuda.empty_cache()
 save(P/('smoke_status.json' if smoke else 'status.json'),{'state':'completed','completed_arms':list(results),'smoke_only':smoke,'no_test_evaluation':True,'seconds':time.time()-started})
 print('COMPLETED', 'smoke' if smoke else 'development',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 if not args.smoke and not args.execute:raise SystemExit('Specify --smoke or --execute.')
 main(args.smoke)
