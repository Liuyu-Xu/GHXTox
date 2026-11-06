"""Fixed A/B/C sequence-expert experiment after LoRA backward repair.

Research-only checkpoints. --execute is required for long training; --check is
the short real-backward rehearsal. Historical frozen3D fusion is conditional.
"""
import os,sys,pathlib,json,csv,time,hashlib,gc,argparse
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'));sys.path.insert(0,str(R/'src'))
os.environ['OMP_NUM_THREADS']='2';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from transformers import T5Config,T5EncoderModel,T5Tokenizer
from sklearn.linear_model import LogisticRegression
from ghxtox.utils import set_seed
import ghxtox.lora_mamba_1d as lm
from run_trial import sample_weight
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def load_model(seed):
 set_seed(seed)
 path=R/'analysis/upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027/Rostlab/prot_t5_xl_half_uniref50-enc'
 cfg=read(R/'configs/lora_mamba_1d_autodl.json');cfg['plm']=str(path)
 with torch.device('meta'):plm=T5EncoderModel(T5Config.from_pretrained(str(path),local_files_only=True))
 state=torch.load(path/'pytorch_model.bin',map_location='cpu',weights_only=True)
 if 'shared.weight' not in state:state['shared.weight']=state['encoder.embed_tokens.weight']
 if 'encoder.embed_tokens.weight' not in state:state['encoder.embed_tokens.weight']=state['shared.weight']
 state={k:v.to(device='cuda',dtype=torch.float16) for k,v in state.items()};plm.load_state_dict(state,strict=True,assign=True);del state;plm.tie_weights()
 model=lm.build_model(plm,cfg,torch.device('cuda'))
 tok=T5Tokenizer.from_pretrained(str(path),local_files_only=True,do_lower_case=False)
 return model,tok,cfg

def main(args):
 if args.check:
  from check_sequence_training import main as check
  check();return
 if not args.execute:raise SystemExit('Use --check for real gradient rehearsal, or --execute for the long, predeclared A/B/C run.')
 meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json')
 roles=read(P/'role_weight_contract.json');background=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json')
 assert all(m['sample_id']==b['sample_id'] and m['sequence']==b['sequence'] and m['label']==int(b['label']) for m,b in zip(meta,background))
 folds=[int(x) for x in args.folds.split(',')];seeds=[int(x) for x in args.seeds.split(',')]
 for seed in seeds:
  for fold in folds:
   rr={role:[r['source_index'] for r in roles if r['outer_fold']==fold and r['role']==role] for role in ['train','validation','calibration','test']}
   groups={role:{meta[i]['group_id'] for i in ix} for role,ix in rr.items()}
   assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b)
   model,tok,cfg=load_model(seed+1000*fold)
   trainable={n:p for n,p in model.named_parameters() if p.requires_grad};initial={n:p.detach().cpu().clone() for n,p in trainable.items()}
   def batch(ix):
    enc=tok([' '.join(meta[i]['sequence']) for i in ix],padding=True,add_special_tokens=True,return_tensors='pt')
    assert enc['input_ids'].shape[1]==max(len(meta[i]['sequence']) for i in ix)+1
    return enc['input_ids'].cuda(),enc['attention_mask'].cuda(),torch.tensor([len(meta[i]['sequence']) for i in ix],device='cuda'),torch.tensor([meta[i]['label'] for i in ix],device='cuda',dtype=torch.float32)
   def predict(ix):
    model.eval();out=[]
    with torch.inference_mode():
     for k in range(0,len(ix),16):
      selected=ix[k:k+16];ids,mask,lens,labels=batch(selected)
      with torch.autocast(device_type='cuda',dtype=torch.float16):v=model(ids,mask,lens)
      out += [{'source_index':i,'logit':float(z)} for i,z in zip(selected,v.cpu().tolist())]
    return out
   pos=sum(meta[i]['label'] for i in rr['train']);pw=torch.tensor((len(rr['train'])-pos)/pos,device='cuda')
   for arm in ['A','B','C']:
    folder=P/'sequence_runs'/f'seed{seed}'/f'fold{fold}'/arm
    if (folder/'result.json').exists():continue
    model.load_state_dict(initial,strict=False);set_seed(seed+fold*1000)
    optimizer=torch.optim.AdamW(list(trainable.values()),lr=.0003,weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(init_scale=128.0)
    best=float('inf');best_state=None;best_epoch=None;history=[];started=time.time()
    for epoch in range(1,16):
     gen=torch.Generator().manual_seed(seed+fold*1000+epoch*10000)
     order=[rr['train'][i] for i in torch.randperm(len(rr['train']),generator=gen).tolist()]
     model.train();loss_sum=0.0;steps=0
     for offset in range(0,len(order),16):
      ix=order[offset:offset+16];ids,mask,lens,labels=batch(ix);w=torch.tensor([sample_weight(meta[i],arm) for i in ix],device='cuda')
      optimizer.zero_grad(set_to_none=True)
      with torch.autocast(device_type='cuda',dtype=torch.float16):
       logits=model(ids,mask,lens)
       per=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,pos_weight=pw,reduction='none');loss=(w*per).sum()/w.sum()
      scaler.scale(loss).backward();scaler.unscale_(optimizer)
      bs=[p for n,p in trainable.items() if '.lora_B' in n]
      if not all(p.grad is not None for p in bs):raise RuntimeError('LoRA backward disconnected; refusing to train silently.')
      torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0)
      scaler.step(optimizer);scaler.update();loss_sum+=float(loss.item())*len(ix);steps+=1
     validation=predict(rr['validation'])
     vl=float(np.mean([np.logaddexp(0,r['logit'])-meta[r['source_index']]['label']*r['logit'] for r in validation]))
     history.append({'epoch':epoch,'train_weighted_loss':loss_sum/len(order),'validation_unweighted_BCE':vl,'steps':steps})
     if vl<best:best=vl;best_epoch=epoch;best_state={n:p.detach().cpu().clone() for n,p in trainable.items()}
     print(f'SEQ seed={seed} fold={fold} arm={arm} epoch={epoch}/15 val={vl:.4f} seconds={time.time()-started:.1f}',flush=True)
    model.load_state_dict(best_state,strict=False)
    cal=predict(rr['calibration']);c=LogisticRegression(C=1,max_iter=1000).fit(np.array([[x['logit']] for x in cal]),np.array([meta[x['source_index']]['label'] for x in cal]))
    test=predict(rr['test']);prob=c.predict_proba(np.array([[x['logit']] for x in test]))[:,1]
    for r,p in zip(test,prob):
     i=r['source_index'];s3=float(background[i]['p_struct']);fusion=.72*float(p)+.28*s3
     r.update(sample_id=meta[i]['sample_id'],label=meta[i]['label'],sequence_probability=float(p),sequence_prediction=int(p>=.5),frozen_structure_probability=s3,conditional_single_seed_fusion_probability=fusion,conditional_single_seed_fusion_prediction=int(fusion>=.47022256531284556))
    save(folder/'result.json',{'arm':arm,'seed':seed,'outer_fold':fold,'config':cfg,'epochs':15,'selected_epoch':best_epoch,'history':history,'training_indices':rr['train'],'validation_indices':rr['validation'],'calibration_records':cal,'platt':{'slope':float(c.coef_[0,0]),'intercept':float(c.intercept_[0])},'predictions':test,'seconds':time.time()-started,'scope':'single-seed actual1D experiment; fixed historical3D conditional fusion NOT published3-seed ensemble or blind full-pipeline validation'})
    torch.save({'artifact_type':'research_weighted_lora_bimamba','config':cfg,'model_state':best_state,'seed':seed,'outer_fold':fold,'arm':arm},folder/'best_model.pt')
    del optimizer,scaler,best_state;gc.collect();torch.cuda.empty_cache()
   del model,initial,trainable;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--folds',default='0,1,2,3,4');ap.add_argument('--seeds',default='42')
 main(ap.parse_args())
