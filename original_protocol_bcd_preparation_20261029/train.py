import os,sys,json,csv,random,math,argparse,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'src'));os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np,torch
from transformers import T5Config,T5EncoderModel,T5Tokenizer
from ghxtox.lora_mamba_1d import build_model,make_training_optimizer
from sampling import batches,supcon,group
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def seed(x):random.seed(x);np.random.seed(x);torch.manual_seed(x);torch.cuda.manual_seed_all(x)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',required=True);ap.add_argument('--arm',choices=['A','B','C','D'],required=True);ap.add_argument('--seed',type=int,choices=[42,123,2025],required=True);ap.add_argument('--fold',choices=['0','1','2','3','4','full'],required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();assert torch.cuda.is_available();torch.set_num_threads(2)
 data=read(P/'data/annotations.json');cfg=read(P/'configs/model.json')
 with (P/'data/folds.csv').open(encoding='utf8',newline='') as h:folds=list(csv.DictReader(h))
 assert len(data)==len(folds)==6387 and all(x['sample_id']==r['sample_id'] and x['sequence']==r['sequence'] and x['label']==int(r['label']) and x['group_id']==r['group_id'] for x,r in zip(data,folds))
 for x in data:x['length']=len(x['sequence'])
 tr=[i for i,r in enumerate(folds) if a.fold=='full' or int(r['fold'])!=int(a.fold)];ev=[] if a.fold=='full' else [i for i,r in enumerate(folds) if int(r['fold'])==int(a.fold)]
 assert tr and (a.fold=='full' or ev) and not ({data[i]['group_id'] for i in tr}&{data[i]['group_id'] for i in ev})
 seed(a.seed+(0 if a.fold=='full' else 1000*int(a.fold)));path=Path(a.model_dir).resolve()
 with torch.device('meta'):plm=T5EncoderModel(T5Config.from_pretrained(str(path),local_files_only=True))
 st=torch.load(path/'pytorch_model.bin',map_location='cpu',weights_only=True)
 if 'shared.weight' not in st:st['shared.weight']=st['encoder.embed_tokens.weight']
 if 'encoder.embed_tokens.weight' not in st:st['encoder.embed_tokens.weight']=st['shared.weight']
 st={k:v.to(device='cuda',dtype=torch.float16) for k,v in st.items()};plm.load_state_dict(st,strict=True,assign=True);del st;plm.tie_weights();model=build_model(plm,cfg,torch.device('cuda'))
 assert cfg['lora_rank']==0 and not any('.lora_' in n for n,p in model.named_parameters())
 assert all(not p.requires_grad for p in model.plm.parameters())
 tok=T5Tokenizer.from_pretrained(str(path),local_files_only=True,do_lower_case=False);optimizer=make_training_optimizer(model,cfg);params=[p for p in model.parameters() if p.requires_grad];scaler=torch.cuda.amp.GradScaler(init_scale=128.);pos=sum(data[i]['label'] for i in tr);weight=torch.tensor((len(tr)-pos)/pos,device='cuda');h={};hook=model.pool_norm.register_forward_hook(lambda m,i,o:h.__setitem__('pool',o));initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params)).hexdigest()
 dest=P/('smoke_no_lora' if a.smoke else 'runs')/a.arm/f'seed{a.seed}'/f'fold{a.fold}'
 if (dest/'result.json').exists():raise SystemExit('Completed result exists; refusing overwrite')
 dest.mkdir(parents=True,exist_ok=True);history=[]
 def pack(ids):
  enc=tok([' '.join(data[i]['sequence']) for i in ids],padding=True,add_special_tokens=True,return_tensors='pt');return enc['input_ids'].cuda(),enc['attention_mask'].cuda(),torch.tensor([len(data[i]['sequence']) for i in ids],device='cuda'),torch.tensor([data[i]['label'] for i in ids],device='cuda',dtype=torch.float32)
 for epoch in range(1,2 if a.smoke else int(cfg['epochs'])+1):
  seed(a.seed+epoch*10000+(0 if a.fold=='full' else int(a.fold)*1000000));order=batches(data,tr,a.seed+epoch*10000+(0 if a.fold=='full' else int(a.fold)*1000000),a.arm!='A');order=order[:2] if a.smoke else order;model.train();loss_sum=0.;anchors={'ordinary':0,'low':0,'high':0}
  for selected in order:
   ids,mask,lens,ys=pack(selected);optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):
    logits=model(ids,mask,lens);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,ys,pos_weight=weight);eligible=torch.tensor([not data[i]['uniform_flag'] for i in selected],device='cuda',dtype=torch.bool);groups=[data[i]['group_id'] for i in selected]
    if a.arm in ('C','D'):
     l,n=supcon(h['pool'],ys,groups,eligible);loss=loss+.1*l;anchors['ordinary']+=n
    if a.arm=='D':
     for g in ('low','high'):
      subset=torch.tensor([group(data[i])==g for i in selected],device='cuda');l,n=supcon(h['pool'],ys,groups,eligible,subset=subset);loss=loss+.05*l;anchors[g]+=n
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   scaler.scale(loss).backward();scaler.unscale_(optimizer)
   if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):raise RuntimeError('nonfinite grad')
   scaler.step(optimizer);scaler.update();loss_sum+=float(loss.detach())
  history.append({'epoch':epoch,'steps':len(order),'mean_loss':loss_sum/len(order),'anchor_counts':anchors});save(dest/'progress.json',{'history':history});print(json.dumps(history[-1]),flush=True)
 trainstate={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad};checkpoint={'model_state':trainstate,'config':cfg,'seed':a.seed,'arm':a.arm,'fold':a.fold,'lora_removed':True};torch.save(checkpoint,dest/'checkpoint.pt')
 if ev and not a.smoke:
  model.eval();probs=[]
  with torch.inference_mode():
   for j in range(0,len(ev),16):
    ids,mask,lens,ys=pack(ev[j:j+16])
    with torch.autocast('cuda',dtype=torch.float16):v=model(ids,mask,lens)
    probs.extend(torch.sigmoid(v.float()).cpu().tolist())
  save(dest/'oof.json',[{'sample_id':data[i]['sample_id'],'fold':int(a.fold),'label':data[i]['label'],'probability':float(p)} for i,p in zip(ev,probs)])
 save(dest/'result.json',{'success':True,'arm':a.arm,'seed':a.seed,'fold':a.fold,'smoke':a.smoke,'n_train':len(tr),'n_eval':len(ev),'history':history,'initial_head_sha256':initial,'pos_weight':float(weight),'fixed_epochs':int(cfg['epochs']),'lora_removed':True,'sampling':('random' if a.arm=='A' else 'cys_stratified'),'training_evidence':'exploratory reproduction, not proven identical to historical training'})
if __name__=='__main__':main()
