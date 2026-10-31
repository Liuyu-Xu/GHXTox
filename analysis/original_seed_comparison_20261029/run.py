import sys,os,json,csv
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'original_protocol_bcd_preparation_20261029';R=P.parent;sys.path[:0]=[str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'),str(P/'src')];os.environ['HF_HUB_OFFLINE']='1'
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score
from transformers import T5Config,T5EncoderModel,T5Tokenizer
from ghxtox.lora_mamba_1d import build_model
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def csvread(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def logit(p):
 q=np.clip(p,1e-6,1-1e-6);return np.log(q)-np.log1p(-q)
torch.set_num_threads(2);path=R/'analysis/upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027/Rostlab/prot_t5_xl_half_uniref50-enc';cfg=read(P/'configs/historical_model.json');folds=csvread(P/'data/folds.csv');oof=[]
for seed in (42,123,2025):
 o=read(P/f'reference/oof_seed{seed}.json');assert o['seed']==seed and o['sample_ids']==[x['sample_id'] for x in folds] and o['labels']==[int(x['label']) for x in folds] and o['folds']==[int(x['fold']) for x in folds];oof.append(o['oof'])
cal=LogisticRegression(C=1e6,max_iter=1000).fit(logit(np.mean(oof,axis=0))[:,None],np.array([int(x['label']) for x in folds]));assert cal.coef_[0,0]>0
with torch.device('meta'):plm=T5EncoderModel(T5Config.from_pretrained(str(path),local_files_only=True))
st=torch.load(path/'pytorch_model.bin',map_location='cpu',weights_only=True)
if 'shared.weight' not in st:st['shared.weight']=st['encoder.embed_tokens.weight']
if 'encoder.embed_tokens.weight' not in st:st['encoder.embed_tokens.weight']=st['shared.weight']
st={k:v.to('cuda',dtype=torch.float16) for k,v in st.items()};plm.load_state_dict(st,strict=True,assign=True);del st;plm.tie_weights();model=build_model(plm,cfg,torch.device('cuda'));tok=T5Tokenizer.from_pretrained(str(path),local_files_only=True,do_lower_case=False)
for split in ['test1','test2']:
 rows=csvread(P/f'data/{split}_structure.csv');ref=csvread(P/f'reference/{split}_original.csv');assert len(rows)==len(ref) and all(x['sample_id']==z['sample_id'] and x['label']==z['label'] for x,z in zip(rows,ref));scores=[]
 for seed in (42,123,2025):
  cp=torch.load(R/f'runs/lora_mamba_full/full_seed{seed}.pt',map_location='cpu',weights_only=True);assert cp['seed']==seed;assert set(cp['model_state'])=={n for n,p in model.named_parameters() if p.requires_grad};assert all(torch.count_nonzero(x)==0 for n,x in cp['model_state'].items() if '.lora_B' in n);model.load_state_dict(cp['model_state'],strict=False);model.eval();v=[]
  with torch.inference_mode():
   for i in range(0,len(rows),16):
    batch=rows[i:i+16];sequences=[x['sequence'] for x in batch];x=tok([' '.join(s) for s in sequences],padding=True,add_special_tokens=True,return_tensors='pt');l=torch.tensor([len(s) for s in sequences],device='cuda')
    with torch.autocast('cuda',dtype=torch.float16):z=model(x['input_ids'].cuda(),x['attention_mask'].cuda(),l)
    v.extend(torch.sigmoid(z.float()).cpu().tolist())
  scores.append(v)
  print(split,seed,flush=True)
 for j,seed in enumerate((42,123,2025)):
  sc=LogisticRegression(C=1e6,max_iter=1000).fit(logit(oof[j])[:,None],np.array([int(x['label']) for x in folds]));assert sc.coef_[0,0]>0
  sp=sc.predict_proba(logit(scores[j])[:,None])[:,1];sq=.72*sp+.28*np.array([float(x['structure_probability']) for x in rows]);yy=np.array([int(x['label']) for x in rows]);dd=sq>=.47022256531284556
  print('SINGLE',split,seed,'MCC',float(matthews_corrcoef(yy,dd)),'AUPRC',float(average_precision_score(yy,sq)),'FN',int(((yy==1)&~dd).sum()),'FP',int(((yy==0)&dd).sum()),flush=True)
  result={'split':split,'seed':seed,'MCC':float(matthews_corrcoef(yy,dd)),'AUPRC':float(average_precision_score(yy,sq)),'FN':int(((yy==1)&~dd).sum()),'FP':int(((yy==0)&dd).sum()),'probabilities':sq.tolist()}
  (R/'analysis/original_seed_comparison_20261029'/f'{split}_seed{seed}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 pp=cal.predict_proba(logit(np.mean(scores,axis=0))[:,None])[:,1];q=.72*pp+.28*np.array([float(x['structure_probability']) for x in rows]);y=np.array([int(x['label']) for x in rows]);old=np.array([float(x['toxicity_probability']) for x in ref]);d=q>=.47022256531284556;old_d=old>=.47022256531284556;result={'n':len(y),'max_difference':float(np.max(np.abs(q-old))),'mean_difference':float(np.mean(np.abs(q-old))),'changed_decisions':int(np.sum(d!=old_d)),'MCC':float(matthews_corrcoef(y,d)),'AUPRC':float(average_precision_score(y,q)),'historical_MCC':float(matthews_corrcoef(y,old_d)),'three_d':'cached historical, not rerun from 3D weights','calibration':'refit original OOF with C=1e6'};(P/f'reference/{split}_reproduction_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(split,json.dumps(result),flush=True)

