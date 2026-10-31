import sys,os,json,csv,gc,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path[:0]=[str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'),str(R/'autodl_cys_supcon_20261029/src')];os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,torch
from transformers import T5Config,T5EncoderModel,T5Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
from ghxtox.lora_mamba_1d import build_model
read=lambda q:json.loads(q.read_text(encoding='utf8'))
def csvread(q):
 with q.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(n,v):(P/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
torch.set_num_threads(2);device=torch.device('cuda');assert torch.cuda.is_available();model_path=R/'analysis/upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027/Rostlab/prot_t5_xl_half_uniref50-enc';cfg=read(R/'autodl_cys_supcon_20261029/configs/model.json');data=read(R/'autodl_cys_supcon_20261029/data/samples.json');roles=read(R/'autodl_cys_supcon_20261029/data/roles.json');test=csvread(R/'runs/single_structure_test_components/test1_structure_predictions.csv');old=csvread(R/'runs/default_multidimensional_fusion_single_structure/test1_predictions.csv');test2=csvread(R/'analysis/test_cys_length_20261028/test2_samples.csv');assert len(test)==len(old)==1126 and all(a['sample_id']==b['sample_id'] and a['label']==b['label'] for a,b in zip(test,old));indices={x['sequence']:i for i,x in enumerate(test)};assert len(indices)==1126;test2_ix=np.array([indices[x['sequence']] for x in test2]);assert len(test2_ix)==582 and all(test[i]['label']==x['label'] for i,x in zip(test2_ix,test2))
with torch.device('meta'):plm=T5EncoderModel(T5Config.from_pretrained(str(model_path),local_files_only=True))
state=torch.load(model_path/'pytorch_model.bin',map_location='cpu',weights_only=True)
if 'shared.weight' not in state:state['shared.weight']=state['encoder.embed_tokens.weight']
if 'encoder.embed_tokens.weight' not in state:state['encoder.embed_tokens.weight']=state['shared.weight']
state={k:v.to(device=device,dtype=torch.float16) for k,v in state.items()};plm.load_state_dict(state,strict=True,assign=True);del state;plm.tie_weights();model=build_model(plm,cfg,device);tok=T5Tokenizer.from_pretrained(str(model_path),local_files_only=True,do_lower_case=False);model.eval();struct=np.array([float(r['structure_probability']) for r in test]);out={a:[] for a in 'ABCD'};audit=[]
def infer(seqs):
 vals=[]
 with torch.inference_mode():
  for i in range(0,len(seqs),16):
   batch=seqs[i:i+16];x=tok([' '.join(s) for s in batch],padding=True,add_special_tokens=True,return_tensors='pt');lens=torch.tensor([len(s) for s in batch],device=device)
   with torch.autocast('cuda',dtype=torch.float16):v=model(x['input_ids'].to(device),x['attention_mask'].to(device),lens)
   vals.extend(v.float().cpu().tolist())
 return np.array(vals)
for f in range(5):
 ci=[r['source_index'] for r in roles if r['outer_fold']==f and r['role']=='calibration'];vi=[r['source_index'] for r in roles if r['outer_fold']==f and r['role']=='test'];cy=np.array([data[i]['label'] for i in ci]);assert len(set(cy))==2
 for a in 'ABCD':
  path=P/f'runs/fold{f}/{a}/best_model.pt';cp=torch.load(path,map_location='cpu',weights_only=True);assert cp['fold']==f and cp['arm']==a and cp['seed']==42;assert set(cp['model_state'])=={n for n,p in model.named_parameters() if p.requires_grad};model.load_state_dict(cp['model_state'],strict=False);model.eval();cl=infer([data[i]['sequence'] for i in ci]);cal=LogisticRegression(C=1,max_iter=1000).fit(cl[:,None],cy);v=infer([data[i]['sequence'] for i in vi]);calv=cal.predict_proba(v[:,None])[:,1]
  rr=(R/f'analysis/autodl_cys_fold{f}_20261029/runs/fold{f}/{a}/predictions.json') if f<3 else R/f'analysis/autodl_cys_fivefold_20261029/runs/fold{f}/{a}/predictions.json';prev=read(rr);assert all(prev[j]['sample_id']==data[i]['sample_id'] for j,i in enumerate(vi));delta=np.abs(calv-np.array([r['sequence_probability'] for r in prev]));ts=infer([r['sequence'] for r in test]);out[a].append(cal.predict_proba(ts[:,None])[:,1]);audit.append({'fold':f,'arm':a,'oof_mean_abs_difference':float(delta.mean()),'oof_max_abs_difference':float(delta.max()),'oof_decision_disagreements_0_5':int(np.sum((calv>=.5)!=(np.array([r['sequence_probability'] for r in prev])>=.5))),'calibration_coef':float(cal.coef_[0,0]),'calibration_intercept':float(cal.intercept_[0])});print(json.dumps(audit[-1]),flush=True);save('audit_progress.json',audit)
y=np.array([int(r['label']) for r in test]);c=np.array([r['sequence'].count('C') for r in test]);p={a:.72*np.mean(out[a],axis=0)+.28*struct for a in out};p['historical']=np.array([float(r['toxicity_probability']) for r in old]);metrics={}
for split,ix in [('test1',np.arange(len(test))),('test2',test2_ix)]:
 metrics[split]={}
 for a,q in p.items():
  yy=y[ix];cc=c[ix];z=q[ix];d=z>=.47022256531284556;metrics[split][a]={'n':len(ix),'MCC':float(matthews_corrcoef(yy,d)),'AUPRC':float(average_precision_score(yy,z)),'AUROC':float(roc_auc_score(yy,z)),'FN':int(((yy==1)&~d).sum()),'FP':int(((yy==0)&d).sum()),'low_cys_positive_FN':int(((cc<=1)&(yy==1)&~d).sum()),'high_cys_negative_FP':int(((cc>=6)&(yy==0)&d).sum())}
save('metrics.json',metrics);save('audit.json',audit);save('test1_predictions.json',[{'sample_id':r['sample_id'],'sequence':r['sequence'],'label':int(y[i]),**{a:float(v[i]) for a,v in p.items()}} for i,r in enumerate(test)]);print('METRICS',json.dumps(metrics),flush=True)
