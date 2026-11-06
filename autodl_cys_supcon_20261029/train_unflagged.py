import os,sys,json,random,math,argparse,time,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'src'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false');os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
from transformers import T5Config,T5EncoderModel,T5Tokenizer
from ghxtox.lora_mamba_1d import build_model
from sampling import batches,supcon,group
read=lambda q:json.loads(q.read_text(encoding='utf8'))
def save(q,x):q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def seed(v):random.seed(v);np.random.seed(v);torch.manual_seed(v);torch.cuda.manual_seed_all(v)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',required=True);ap.add_argument('--fold',type=int,default=1);ap.add_argument('--arm',choices=['A','B','C','D'],required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
 if not torch.cuda.is_available():raise SystemExit('A CUDA GPU is required')
 torch.set_num_threads(2);rows=read(P/'data/samples.json');roles=read(P/'data/roles.json');cfg=read(P/'configs/model.json');device=torch.device('cuda');assert cfg['lora_rank']==0
 ix={role:[x['source_index'] for x in roles if x['outer_fold']==a.fold and x['role']==role] for role in ['train','validation','calibration','test']}
 assert all(ix.values());assert all(not {rows[i]['group_id'] for i in ix[x]}&{rows[i]['group_id'] for i in ix[y]} for x in ix for y in ix if x!=y)
 model_path=Path(a.model_dir).resolve();assert all((model_path/n).exists() for n in ['config.json','pytorch_model.bin','spiece.model']);seed(a.seed)
 with torch.device('meta'):plm=T5EncoderModel(T5Config.from_pretrained(str(model_path),local_files_only=True))
 state=torch.load(model_path/'pytorch_model.bin',map_location='cpu',weights_only=True)
 if 'shared.weight' not in state:state['shared.weight']=state['encoder.embed_tokens.weight']
 if 'encoder.embed_tokens.weight' not in state:state['encoder.embed_tokens.weight']=state['shared.weight']
 state={k:v.to(device=device,dtype=torch.float16) for k,v in state.items()};plm.load_state_dict(state,strict=True,assign=True);del state;plm.tie_weights();model=build_model(plm,cfg,device);tok=T5Tokenizer.from_pretrained(str(model_path),local_files_only=True,do_lower_case=False)
 assert all(not p.requires_grad for p in model.plm.parameters());params=[p for p in model.parameters() if p.requires_grad];assert params
 initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params)).hexdigest();opt=torch.optim.AdamW(params,lr=0.0003,weight_decay=0.01);scaler=torch.cuda.amp.GradScaler(init_scale=128.);n_pos=sum(rows[i]['label'] for i in ix['train']);posweight=torch.tensor((len(ix['train'])-n_pos)/n_pos,device=device)
 original_train_n=len(ix['train'])
 ix['train']=[i for i in ix['train'] if not rows[i]['uniform_flag']]
 print(f'train samples: {original_train_n} -> {len(ix["train"])}; original positive weight kept',flush=True)
 output=P/('smoke' if a.smoke else 'runs')/f'fold{a.fold}'/(a.arm+'_unflagged')
 if (output/'result.json').exists():raise SystemExit(f'Result exists: {output}. Preserve existing run; choose new output or archive it first.')
 output.mkdir(parents=True,exist_ok=True);captures={};hook=model.pool_norm.register_forward_hook(lambda m,i,o:captures.__setitem__('pooled',o))
 def pack(ids):
  enc=tok([' '.join(rows[i]['sequence']) for i in ids],padding=True,add_special_tokens=True,return_tensors='pt');return enc['input_ids'].to(device),enc['attention_mask'].to(device),torch.tensor([len(rows[i]['sequence']) for i in ids],device=device),torch.tensor([rows[i]['label'] for i in ids],dtype=torch.float32,device=device)
 def logits(ids):
  model.eval();v=[]
  with torch.inference_mode():
   for k in range(0,len(ids),16):
    ids1,mask,lens,ys=pack(ids[k:k+16])
    with torch.autocast('cuda',dtype=torch.float16):o=model(ids1,mask,lens)
    v.extend(o.float().cpu().tolist())
  return np.array(v)
 best=float('inf');best_state=None;stale=0;hist=[];started=time.time()
 for epoch in range(1,2 if a.smoke else 9):
  seed(a.seed+epoch*10000);orders=batches(rows,ix['train'],a.seed+epoch*10000,a.arm!='A');orders=orders[:2] if a.smoke else orders;model.train();ac={'ordinary':0,'low':0,'high':0};loss_sum=0
  for chosen in orders:
   ids,mask,lens,ys=pack(chosen);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):
    logits1=model(ids,mask,lens);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits1,ys,pos_weight=posweight);eligible=torch.tensor([not rows[i]['uniform_flag'] for i in chosen],device=device,dtype=torch.bool);groups=[rows[i]['group_id'] for i in chosen]
    if a.arm in ('C','D'):
     l,n=supcon(captures['pooled'],ys,groups,eligible);loss=loss+0.1*l;ac['ordinary']+=n
    if a.arm=='D':
     for g in ('low','high'):
      sub=torch.tensor([group(rows[i])==g for i in chosen],device=device);l,n=supcon(captures['pooled'],ys,groups,eligible,subset=sub);loss=loss+0.05*l;ac[g]+=n
   if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
   scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(params,1.0)
   if not torch.isfinite(norm):raise RuntimeError('Non-finite gradients')
   scaler.step(opt);scaler.update();loss_sum+=float(loss)
  v=ix['validation'][:32] if a.smoke else ix['validation'];z=logits(v);yy=np.array([rows[i]['label'] for i in v]);vl=float(np.mean(np.logaddexp(0,z)-yy*z));hist.append({'epoch':epoch,'train_loss_mean':loss_sum/len(orders),'validation_bce':vl,'anchor_counts':ac,'seconds':time.time()-started});save(output/'progress.json',{'history':hist,'arm':a.arm,'smoke':a.smoke});print(json.dumps(hist[-1]),flush=True)
  if vl<best:best=vl;best_state={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad};stale=0
  else:stale+=1
  if stale>=2:break
 model.load_state_dict(best_state,strict=False)
 if a.smoke:save(output/'result.json',{'success':True,'smoke':True,'initial_head_sha256':initial,'history':hist});return
 torch.save({'model_state':best_state,'config':cfg,'fold':a.fold,'arm':a.arm,'seed':a.seed},output/'best_model.pt')
 ci=ix['calibration'];ti=ix['test'];cl=logits(ci);tl=logits(ti);cy=np.array([rows[i]['label'] for i in ci]);classifier=LogisticRegression(C=1,max_iter=1000).fit(cl[:,None],cy);probs=classifier.predict_proba(tl[:,None])[:,1];fusion=np.array([.72*pr+.28*rows[i]['p_struct'] for i,pr in zip(ti,probs)]);y=np.array([rows[i]['label'] for i in ti]);c=np.array([rows[i]['cys_count'] for i in ti]);metrics={}
 for name,pp,threshold in [('sequence',probs,.5),('fusion',fusion,.47022256531284556)]:
  d=pp>=threshold;metrics[name]={'MCC':float(matthews_corrcoef(y,d)),'AUPRC':float(average_precision_score(y,pp)),'FN':int(((y==1)&~d).sum()),'FP':int(((y==0)&d).sum()),'low_positive_FN':int(((y==1)&(c<=1)&~d).sum()),'low_positive_n':int(((y==1)&(c<=1)).sum()),'high_negative_FP':int(((y==0)&(c>=6)&d).sum()),'high_negative_n':int(((y==0)&(c>=6)).sum())}
 save(output/'predictions.json',[{'sample_id':rows[i]['sample_id'],'group_id':rows[i]['group_id'],'label':int(y[j]),'sequence_probability':float(probs[j]),'fusion_probability':float(fusion[j])} for j,i in enumerate(ti)]);save(output/'result.json',{'success':True,'smoke':False,'fold':a.fold,'seed':a.seed,'arm':a.arm,'initial_head_sha256':initial,'selected_validation_bce':best,'history':hist,'metrics':metrics,'note':'historical 3D OOF and threshold; conditional exploratory fusion, not full nested or external validation'})
if __name__=='__main__':main()
