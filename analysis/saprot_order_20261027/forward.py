"""Frozen forward encoders and same-fold heads; originals must pass anchor gate."""
from pathlib import Path
import os,sys,json,time,gc,argparse,hashlib,faulthandler
faulthandler.dump_traceback_later(60,repeat=True)
print('Starting frozen forward process',flush=True)
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
import numpy as np
import torch
print('Torch and NumPy imported',flush=True)
from torch import nn
from scipy.special import expit,logsumexp,log_expit
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';sys.path.insert(0,str(R/'src'))
class Tee:
 def __init__(self,stream,path):self.stream=stream;self.file=path.open('a',encoding='utf-8',buffering=1)
 def write(self,text):self.stream.write(text);self.file.write(text);self.file.flush()
 def flush(self):self.stream.flush();self.file.flush()
sys.stdout=Tee(sys.stdout,O/(sys.argv[1]+'.live.log'))
from ghxtox.models.physchem_features import physchem_feature_vector
from ghxtox.models.tristat_pooling import tristat_pool
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(n,obj):(O/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
torch.set_num_threads(2);torch.manual_seed(20261027)
class Head(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(3561,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)
combine=lambda a:logsumexp(log_expit(np.stack(a)),axis=0)-logsumexp(log_expit(-np.stack(a)),axis=0)
def head_scores(cohort,features,sequences):
 ix={s:i for i,s in enumerate(sequences)};rawbase=features['base'];rawsap=features['saprot'];out={k:np.full(len(cohort),np.nan) for k in ['baseline2','baseline4','teacher2','fusion_masked4','baseline2_clamped','baseline4_clamped']};old=np.load(A/'saprot_and_robust_classifier_pilot_20261026/frozen_features.npz');oldbase=old['base'];oi={k:i for i,k in enumerate(old['sample_ids'].tolist())};checklog=[]
 for fold in range(5):
  ids=[i for i,r in enumerate(cohort) if r['fold']==fold]
  if not ids:continue
  fi=[ix[cohort[i]['sequence']] for i in ids];b=np.concatenate([rawbase[fi],np.zeros((len(ids),480),np.float32)],axis=1);bc=b.copy();bc[:,3072:3081]=np.array([oldbase[oi[cohort[i]['sample_id']],3072:3081] for i in ids]);t=np.concatenate([np.zeros((len(ids),3081),np.float32),rawsap[fi]],axis=1);bl=[];cl=[];tl=[]
  for seed in [20261026,20261027,20261028,20261029]:
   source=A/'saprot_and_robust_classifier_pilot_20261026/checkpoints'/f'{seed}_fold{fold}_baseline_BCE.pt' if seed<20261028 else A/'saprot_ensemble_size_control_20261026/checkpoints'/f'{seed}_fold{fold}.pt';c=torch.load(source,map_location='cpu');m=Head();m.load_state_dict(c['state_dict']);m.eval();assert all(cohort[i]['sample_id'] in c['test_ids'] and cohort[i]['sample_id'] not in c['train_ids'] for i in ids)
   def pred(raw):
    v=raw.copy();v-=np.array(c['scaler_mean']);v/=np.array(c['scaler_scale'])
    with torch.inference_mode():return np.array(m(torch.tensor(v.tolist(),dtype=torch.float32)).tolist())
   bl.append(pred(b));cl.append(pred(bc));checklog.append(dict(fold=fold,seed=seed,kind='baseline',evaluations=len(ids),all_parent_samples_heldout=True))
  for seed in [20261026,20261027]:
   c=torch.load(A/'saprot_standalone_20261026/checkpoints'/f'{seed}_fold{fold}_SaProt_only_masked.pt',map_location='cpu');m=Head();m.load_state_dict(c['state_dict']);m.eval();assert all(cohort[i]['sample_id'] in c['test_ids'] and cohort[i]['sample_id'] not in c['train_ids'] for i in ids);v=t.copy();v-=np.array(c['scaler_mean']);v/=np.array(c['scaler_scale'])
   with torch.inference_mode():tl.append(np.array(m(torch.tensor(v.tolist(),dtype=torch.float32)).tolist()))
   checklog.append(dict(fold=fold,seed=seed,kind='teacher_masked',evaluations=len(ids),all_parent_samples_heldout=True))
  vals={'baseline2':combine(bl[:2]),'baseline4':combine(bl),'teacher2':combine(tl),'fusion_masked4':combine(bl[:2]+tl),'baseline2_clamped':combine(cl[:2]),'baseline4_clamped':combine(cl)}
  for k,v in vals.items():out[k][ids]=v
 for v in out.values():assert np.isfinite(v).all()
 return out,checklog
def encode(sequences,stage):
 existing=O/f'{stage}_features.npz'
 if existing.exists():
  zz=np.load(existing);assert zz['sequences'].tolist()==sequences;print('Reusing complete '+stage+' encoded features',flush=True);return {k:zz[k] for k in ['base','saprot']}
 print('Importing offline encoders',flush=True)
 from transformers import T5Config,T5EncoderModel,T5Tokenizer,EsmConfig,EsmForMaskedLM,EsmTokenizer
 from safetensors.torch import load_file
 assert torch.cuda.is_available();start=time.time();B=R.parent/'GHXTox-ziliao/huggingface_cache/hub/models--Rostlab--prot_t5_xl_half_uniref50-enc/snapshots';meta=B/'94a6abc029ae13029317b140b7424e012bf8dfbf';weights=B/'d53b8ce63df74def1726cc2ca4582fb0f3f42a9d/model.safetensors';cache=O/f'{stage}_base_partial.npz';base=np.zeros((len(sequences),3081),np.float32);done=0
 if cache.exists():
  zz=np.load(cache);assert zz['sequences'].tolist()==sequences;done=int(zz['done']);base[:done]=zz['base'][:done]
 print('Loading ProtT5 tokenizer/config',flush=True)
 tokenizer=T5Tokenizer(vocab_file=str(meta/'spiece.model'),legacy=True,do_lower_case=False);config=T5Config.from_pretrained(str(meta),local_files_only=True)
 with torch.device('meta'):model=T5EncoderModel(config)
 print('Loading frozen safetensors to GPU',flush=True)
 state=load_file(str(weights),device='cuda')
 if 'shared.weight' not in state:state['shared.weight']=state['encoder.embed_tokens.weight']
 if 'encoder.embed_tokens.weight' not in state:state['encoder.embed_tokens.weight']=state['shared.weight']
 model.load_state_dict(state,strict=True,assign=True);del state;model.tie_weights();model.eval().requires_grad_(False);print(f'{stage}: original ProtT5 loaded, dtype {next(model.parameters()).dtype}, {len(sequences)} sequences',flush=True)
 for i in range(done,len(sequences),8):
  chunk=sequences[i:i+8];tokens=tokenizer([' '.join(s) for s in chunk],add_special_tokens=True,padding=True,return_tensors='pt');assert all(int(tokens['attention_mask'][j].sum())==len(s)+1 for j,s in enumerate(chunk));tokens={k:v.to('cuda') for k,v in tokens.items()}
  with torch.inference_mode():h=model(**tokens).last_hidden_state
  for j,s in enumerate(chunk):
   emb=h[j,:len(s)].float().unsqueeze(0);tri=tristat_pool(emb,torch.ones((1,len(s)),dtype=torch.bool,device='cuda')).squeeze(0).cpu().tolist();base[i+j]=np.array(tri+physchem_feature_vector(s),np.float32)
  if (i+8)%128==0 or i+8>=len(sequences):print(f'{stage} ProtT5 {min(i+8,len(sequences))}/{len(sequences)}, {time.time()-start:.1f}s',flush=True)
  if (i+8)%512==0 or i+8>=len(sequences):np.savez_compressed(cache,sequences=np.array(sequences),base=base,done=min(i+8,len(sequences)))
 del model
 if 'h' in locals():del h
 if 'tokens' in locals():del tokens
 gc.collect();torch.cuda.empty_cache();M=A/'saprot_input_audit_20261026/model';config=EsmConfig.from_json_file(str(M/'config.json'));tokenizer=EsmTokenizer.from_pretrained(str(M),local_files_only=True);model=EsmForMaskedLM(config);state=torch.load(M/'pytorch_model.bin',map_location='cpu',weights_only=True);extra=set(state)-set(model.state_dict());assert extra<={'esm.embeddings.position_ids','esm.embeddings.position_embeddings.weight'}
 for k in extra:state.pop(k)
 model.load_state_dict(state,strict=True);del state;model.eval().requires_grad_(False).half().to('cuda');sap=np.zeros((len(sequences),480),np.float32)
 for i in range(0,len(sequences),64):
  chunk=sequences[i:i+64];tokens=tokenizer([''.join(a+'#' for a in s) for s in chunk],return_tensors='pt',padding=True);assert not bool((tokens['input_ids']==tokenizer.unk_token_id).any());assert all(int(tokens['attention_mask'][j].sum())==len(s)+2 for j,s in enumerate(chunk))
  with torch.inference_mode():h=model.esm(**{k:v.to('cuda') for k,v in tokens.items()}).last_hidden_state
  sap[i:i+len(chunk)]=np.array([h[j,1:1+len(s)].float().mean(0).cpu().tolist() for j,s in enumerate(chunk)],np.float32)
 del model;gc.collect();torch.cuda.empty_cache();assert np.isfinite(base).all() and np.isfinite(sap).all();np.savez_compressed(O/f'{stage}_features.npz',sequences=np.array(sequences),base=base,saprot=sap);save(f'{stage}_encoding.json',dict(n_sequences=len(sequences),seconds=time.time()-start,encoder_parameters_frozen=True,device=torch.cuda.get_device_name(0),base_dim=3081,saprot_dim=480,structure_tokens='allmasked',weights_paths=[str(weights),str(M/'pytorch_model.bin')]))
 return dict(base=base,saprot=sap)
parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['originals','mutants']);arg=parser.parse_args();manifest=read(O/'sequence_manifest.json')
if arg.stage=='originals':
 rr=[r for r in manifest if r['mode']=='original'];seqs=sorted({r['sequence'] for r in rr});features=encode(seqs,'originals');pred,checks=head_scores(rr,features,seqs);old=np.load(A/'saprot_and_robust_classifier_pilot_20261026/frozen_features.npz');oldp=np.load(A/'saprot_ensemble_size_control_20261026/ensemble_scores.npz');oldt=np.load(A/'saprot_standalone_20261026/ensemble_scores.npz');oi={k:i for i,k in enumerate(old['sample_ids'].tolist())};si={s:i for i,s in enumerate(seqs)};cos={}
 for key,ref in [('base',old['base'][:,:3072]),('saprot',old['saprot_masked'])]:
  new=np.stack([features[key][si[r['sequence']],:3072] if key=='base' else features[key][si[r['sequence']]] for r in rr]);orig=np.stack([ref[oi[r['sample_id']]] for r in rr]);cc=np.sum(new*orig,axis=1)/(np.linalg.norm(new,axis=1)*np.linalg.norm(orig,axis=1));cos[key]=dict(min=float(cc.min()),mean=float(cc.mean()),passing=bool((cc>=.995).all()))
 pm={}
 for k,source in [('baseline4',oldp['baseline4']),('teacher2',oldt['SaProt_only_masked'])]:
  previous=np.array([source[oi[r['sample_id']]] for r in rr]);diff=np.abs(expit(pred[k])-expit(previous));agree=float(np.mean((pred[k]>=0)==(previous>=0)));pm[k]=dict(mean_absolute_probability_difference=float(diff.mean()),median_absolute_probability_difference=float(np.median(diff)),decision_agreement=agree,passing=bool(np.median(diff)<=.02 and agree>=.98))
 passed=all(r['passing'] for r in cos.values()) and all(r['passing'] for r in pm.values());gate=dict(passed=passed,feature_cosine=cos,prediction_alignment=pm,scope='required input contract before perturbation; fresh originals are paired reference');save('anchor_gate.json',gate);np.savez_compressed(O/'original_scores.npz',sample_ids=np.array([r['sample_id'] for r in rr]),**pred);save('original_head_checks.json',checks);print(json.dumps(gate,indent=2));assert passed,'ANCHOR FAILED: stop before mutated encoding.'
else:
 assert read(O/'anchor_gate.json')['passed'];rr=[r for r in manifest if r['mode']!='original' and r['eligible']];original=np.load(O/'originals_features.npz');oset=set(original['sequences'].tolist());seqs=sorted({r['sequence'] for r in rr}-oset);features=encode(seqs,'mutants');allseq=seqs+original['sequences'].tolist();features={k:np.concatenate([features[k],original[k]],axis=0) for k in ['base','saprot']};pred,checks=head_scores(rr,features,allseq);np.savez_compressed(O/'mutant_scores.npz',keys=np.array([r['key'] for r in rr]),**pred);save('mutant_head_checks.json',checks);print('Mutated forward complete: '+str(len(rr)),flush=True)
