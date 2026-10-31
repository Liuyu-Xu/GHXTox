"""Verify real ProtT5-LoRA/BiMamba weighted gradients, not an OOF performance trial."""
import os,sys,pathlib,json,time,hashlib,gc
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'));sys.path.insert(0,str(R/'src'))
os.environ['OMP_NUM_THREADS']='2';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1'
import torch
from transformers import T5Config,T5EncoderModel,T5Tokenizer
import ghxtox.lora_mamba_1d as lm
from run_trial import sample_weight
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(state):
 h=hashlib.sha256()
 for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def main():
 oldweights=[]
 for seed in [42,123,2025]:
  f=R/f'runs/lora_mamba_full/full_seed{seed}.pt';payload=torch.load(f,map_location='cpu',weights_only=False)
  bs=[v for k,v in payload['model_state'].items() if '.lora_B' in k]
  oldweights.append({'seed':seed,'file':str(f),'B_tensors':len(bs),'B_nonzero_tensors':sum(bool(x.count_nonzero()) for x in bs)})
  del payload,bs
 gc.collect()
 path=R/'analysis/upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027/Rostlab/prot_t5_xl_half_uniref50-enc'
 cfg=read(R/'configs/lora_mamba_1d_autodl.json');cfg['plm']=str(path)
 tc=T5Config.from_pretrained(str(path),local_files_only=True)
 with torch.device('meta'):plm=T5EncoderModel(tc)
 state=torch.load(path/'pytorch_model.bin',map_location='cpu',weights_only=True)
 if 'shared.weight' not in state:state['shared.weight']=state['encoder.embed_tokens.weight']
 if 'encoder.embed_tokens.weight' not in state:state['encoder.embed_tokens.weight']=state['shared.weight']
 state={k:v.to(device='cuda',dtype=torch.float16) for k,v in state.items()}
 plm.load_state_dict(state,strict=True,assign=True);del state;plm.tie_weights()
 lm._set_seed(42) if hasattr(lm,'_set_seed') else torch.manual_seed(42)
 model=lm.build_model(plm,cfg,torch.device('cuda'))
 trainables={n:p for n,p in model.named_parameters() if p.requires_grad}
 initial={n:p.detach().cpu().clone() for n,p in trainables.items()}
 base_digest=digest(initial)
 tokenizer=T5Tokenizer.from_pretrained(str(path),local_files_only=True,do_lower_case=False)
 meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json')
 roles=read(P/'role_weight_contract.json');trainidx=[r['source_index'] for r in roles if r['outer_fold']==0 and r['role']=='train']
 target=next(i for i in trainidx if sample_weight(meta[i],'C')==2)
 low=next(i for i in trainidx if sample_weight(meta[i],'B')==1.5 and sample_weight(meta[i],'C')!=2)
 neg=next(i for i in trainidx if meta[i]['label']==0)
 idx=[target,low,neg];ms=[meta[i] for i in idx]
 encoded=tokenizer([' '.join(m['sequence']) for m in ms],padding=True,add_special_tokens=True,return_tensors='pt')
 ids=encoded['input_ids'].cuda();mask=encoded['attention_mask'].cuda();lengths=torch.tensor([len(m['sequence']) for m in ms],device='cuda')
 labels=torch.tensor([m['label'] for m in ms],device='cuda',dtype=torch.float32)
 positives=sum(meta[i]['label'] for i in trainidx);posweight=torch.tensor((len(trainidx)-positives)/positives,device='cuda')
 outcomes=[]
 for arm in ['A','B','C']:
  model.load_state_dict(initial,strict=False)
  torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  optimizer=torch.optim.AdamW(list(trainables.values()),lr=.0003,weight_decay=.01)
  scaler=torch.cuda.amp.GradScaler(init_scale=128.0)
  model.train();start=time.time();torch.cuda.reset_peak_memory_stats();stepinfo=[]
  weights=torch.tensor([sample_weight(m,arm) for m in ms],device='cuda')
  assert digest({n:p for n,p in trainables.items()})==base_digest
  for step in range(2):
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.float16):
    logits=model(ids,mask,lengths)
    raw=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,pos_weight=posweight,reduction='none')
    loss=(raw*weights).sum()/weights.sum()
   scaler.scale(loss).backward();scaler.unscale_(optimizer)
   bs=[p for n,p in trainables.items() if '.lora_B' in n]
   present=sum(p.grad is not None for p in bs);nonzero=sum(p.grad is not None and bool(p.grad.count_nonzero()) for p in bs)
   assert present==len(bs) and nonzero==len(bs),(present,nonzero,len(bs))
   assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
   norm=torch.nn.utils.clip_grad_norm_(list(trainables.values()),1.0);assert torch.isfinite(norm)
   scaler.step(optimizer);scaler.update()
   stepinfo.append({'step':step+1,'loss':float(loss.item()),'B_grad_present':present,'B_grad_nonzero':nonzero,'B_tensors':len(bs),'gradient_norm_before_clip':float(norm)})
  nonzero=sum(bool(p.count_nonzero()) for n,p in trainables.items() if '.lora_B' in n)
  assert nonzero==144
  outcomes.append({'arm':arm,'samples':[m['sample_id'] for m in ms],'extra_weights':weights.cpu().tolist(),'initial_trainable_sha256':base_digest,'seconds':time.time()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'steps':stepinfo,'B_nonzero_after_steps':nonzero,'outer_eval_or_validation_used':False})
  print('SEQUENCE_WEIGHTED_SMOKE',outcomes[-1],flush=True)
  del optimizer,scaler;gc.collect();torch.cuda.empty_cache()
 output={'old_checkpoint_inspection':oldweights,'new_actual_architecture_weighted_gradient_smoke':outcomes,'trainable_parameters':sum(p.numel() for p in trainables.values()),'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad),'model_source_sha256':hashlib.sha256((R/'src/ghxtox/lora_mamba_1d.py').read_bytes()).hexdigest(),'fix':'non-reentrant activation checkpointing; frozen embeddings no longer disconnect LoRA','no_OOF_performance_claim':True,'no_old_checkpoint_overwrite':True}
 (P/'sequence_training_check.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
if __name__=='__main__':main()
