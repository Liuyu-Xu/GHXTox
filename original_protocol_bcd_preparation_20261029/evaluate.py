import os,sys,json,csv,argparse
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'src'));os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np,torch
from transformers import T5Config,T5EncoderModel,T5Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
from ghxtox.lora_mamba_1d import build_model
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def csvread(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def logit(p):
 q=np.clip(p,1e-6,1-1e-6);return np.log(q)-np.log1p(-q)
def sigmoid(x):return 1/(1+np.exp(-x))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
ap=argparse.ArgumentParser();ap.add_argument('--model-dir',required=True);ap.add_argument('--arm',choices=['A','B','C','D'],required=True);args=ap.parse_args();assert torch.cuda.is_available();torch.set_num_threads(2)
folds=csvread(P/'data/folds.csv');oof=[]
for seed in [42,123,2025]:
 byid={}
 for f in range(5):
  z=read(P/f'runs/{args.arm}/seed{seed}/fold{f}/oof.json')
  for x in z:
   assert x['sample_id'] not in byid and x['fold']==f;byid[x['sample_id']]=x
 assert len(byid)==len(folds)==6387
 assert all(x['sample_id'] in byid and byid[x['sample_id']]['label']==int(x['label']) and byid[x['sample_id']]['fold']==int(x['fold']) for x in folds)
 oof.append([byid[x['sample_id']]['probability'] for x in folds])
y=np.array([int(x['label']) for x in folds]);mean=np.mean(oof,axis=0);cal=LogisticRegression(C=1e6,max_iter=1000).fit(logit(mean)[:,None],y);assert cal.coef_[0,0]>0
cal_oof=cal.predict_proba(logit(mean)[:,None])[:,1];struct=csvread(P/'data/structure_oof.csv');assert len(struct)==len(folds) and all(x['sample_id']==z['sample_id'] and x['label']==z['label'] for x,z in zip(folds,struct));fusion=.72*cal_oof+.28*np.array([float(x['toxicity_probability']) for x in struct]);save(P/f'runs/{args.arm}/calibration.json',{'slope':float(cal.coef_[0,0]),'intercept':float(cal.intercept_[0]),'C':1e6,'oof_scheme':'mean raw probabilities of three seeds then Platt on logits','fusion_weight':.72,'threshold':.47022256531284556,'warning':'all OOF reused in calibration; not independent for OOF score'})
save(P/f'runs/{args.arm}/fusion_oof.json',[{'sample_id':x['sample_id'],'label':int(x['label']),'probability':float(q)} for x,q in zip(folds,fusion)])
path=Path(args.model_dir).resolve();cfg=read(P/'configs/model.json')
with torch.device('meta'):plm=T5EncoderModel(T5Config.from_pretrained(str(path),local_files_only=True))
st=torch.load(path/'pytorch_model.bin',map_location='cpu',weights_only=True)
if 'shared.weight' not in st:st['shared.weight']=st['encoder.embed_tokens.weight']
if 'encoder.embed_tokens.weight' not in st:st['encoder.embed_tokens.weight']=st['shared.weight']
st={k:v.to('cuda',dtype=torch.float16) for k,v in st.items()};plm.load_state_dict(st,strict=True,assign=True);del st;plm.tie_weights();model=build_model(plm,cfg,torch.device('cuda'));tok=T5Tokenizer.from_pretrained(str(path),local_files_only=True,do_lower_case=False)
def infer(seqs):
 out=[];model.eval()
 with torch.inference_mode():
  for i in range(0,len(seqs),16):
   batch=seqs[i:i+16];x=tok([' '.join(s) for s in batch],padding=True,add_special_tokens=True,return_tensors='pt');l=torch.tensor([len(s) for s in batch],device='cuda')
   with torch.autocast('cuda',dtype=torch.float16):z=model(x['input_ids'].cuda(),x['attention_mask'].cuda(),l)
   out.extend(torch.sigmoid(z.float()).cpu().tolist())
 return np.array(out)
for split in ('test1','test2'):
 rows=csvread(P/f'data/{split}_structure.csv');means=[]
 for seed in [42,123,2025]:
  cp=torch.load(P/f'runs/{args.arm}/seed{seed}/foldfull/checkpoint.pt',map_location='cpu',weights_only=True);assert cp['seed']==seed and cp['arm']==args.arm and cp['fold']=='full';expected={n for n,p in model.named_parameters() if p.requires_grad};assert set(cp['model_state'])==expected;model.load_state_dict(cp['model_state'],strict=False);means.append(infer([x['sequence'] for x in rows]))
 mean=np.mean(means,axis=0);q=cal.predict_proba(logit(mean)[:,None])[:,1];z=.72*q+.28*np.array([float(x['structure_probability']) for x in rows]);y=np.array([int(x['label']) for x in rows]);d=z>=.47022256531284556;c=np.array([x['sequence'].count('C') for x in rows]);metrics={'MCC':float(matthews_corrcoef(y,d)),'AUPRC':float(average_precision_score(y,z)),'AUROC':float(roc_auc_score(y,z)),'FN':int(((y==1)&~d).sum()),'FP':int(((y==0)&d).sum()),'low_positive_FN':int(((y==1)&(c<=1)&~d).sum()),'high_negative_FP':int(((y==0)&(c>=6)&d).sum())};save(P/f'runs/{args.arm}/{split}_metrics.json',metrics);save(P/f'runs/{args.arm}/{split}_predictions.json',[{'sample_id':r['sample_id'],'label':int(v),'probability':float(p)} for r,v,p in zip(rows,y,z)]);print(split,metrics,flush=True)
