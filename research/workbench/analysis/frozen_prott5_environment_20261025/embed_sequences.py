"""Frozen original ProtT5 encoder, 1024d residue mean; exclude EOS/padding.
Use local public safetensors weights and config/tokenizer snapshot; no fine-tuning.
"""
from pathlib import Path
import hashlib,json,time,os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import T5Config,T5EncoderModel,T5Tokenizer

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SRC=ROOT/'analysis/environment_feasibility_20261025/model_records.json'
HUB=ROOT.parent/'GHXTox-ziliao/huggingface_cache/hub/models--Rostlab--prot_t5_xl_half_uniref50-enc/snapshots'
META=HUB/'94a6abc029ae13029317b140b7424e012bf8dfbf'
WEIGHTS=HUB/'d53b8ce63df74def1726cc2ca4582fb0f3f42a9d/model.safetensors'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
records=json.loads(SRC.read_text(encoding='utf-8'));seqs=sorted({r['sequence'] for r in records})
assert len(seqs)==1792
OUT.mkdir(exist_ok=True,parents=True)
tokenizer=T5Tokenizer(vocab_file=str(META/'spiece.model'),legacy=True,do_lower_case=False)
config=T5Config.from_pretrained(str(META),local_files_only=True)
device='cuda' if torch.cuda.is_available() else 'cpu';assert device=='cuda','Local GPU expected for this extraction.'
torch.set_num_threads(2)
# Meta initialization avoids loading two model copies into limited CPU RAM.
with torch.device('meta'):model=T5EncoderModel(config)
state=load_file(str(WEIGHTS),device=device)
if 'shared.weight' not in state:state['shared.weight']=state['encoder.embed_tokens.weight']
if 'encoder.embed_tokens.weight' not in state:state['encoder.embed_tokens.weight']=state['shared.weight']
model.load_state_dict(state,strict=True,assign=True);del state
model.tie_weights();model.eval();model.requires_grad_(False)
assert all(not p.requires_grad for p in model.parameters())
assert all(p.device.type=='cuda' for p in model.parameters())
print('Original encoder loaded; no adapters; dtype='+str(next(model.parameters()).dtype),flush=True)
def encode(chunk):
 tokens=tokenizer([' '.join(s) for s in chunk],add_special_tokens=True,padding=True,return_tensors='pt')
 for i,s in enumerate(chunk):assert int(tokens['attention_mask'][i].sum())==len(s)+1
 tokens={k:v.to(device) for k,v in tokens.items()}
 with torch.inference_mode():h=model(**tokens).last_hidden_state
 return np.array([h[i,:len(s)].float().mean(0).cpu().tolist() for i,s in enumerate(chunk)],dtype=np.float32)
start=time.time();vec=np.zeros((len(seqs),1024),dtype=np.float32);bs=8
cache=OUT/'partial_embeddings.npz';done=0
if cache.exists():
 z=np.load(cache);assert z['sequences'].tolist()==seqs;done=int(z['done']);vec[:done]=z['embeddings'][:done]
for k in range(done,len(seqs),bs):
 chunk=seqs[k:k+bs];vec[k:k+len(chunk)]=encode(chunk)
 if k==0 or (k+bs)%128==0 or k+bs>=len(seqs):
  done=min(k+bs,len(seqs));np.savez(cache,sequences=np.array(seqs),embeddings=vec,done=done);print(f'Embedded {done}/{len(seqs)}; {time.time()-start:.1f}s',flush=True)
assert np.isfinite(vec).all() and (np.linalg.norm(vec,axis=1)>0).all()
# Padding/batch consistency check on fixed examples, without inspecting labels.
checks=[]
for i in [0,len(seqs)//2,len(seqs)-1]:
 single=encode([seqs[i]])[0];base=vec[i];cos=float(np.dot(single,base)/(np.linalg.norm(single)*np.linalg.norm(base)));checks.append({'sequence_index':i,'cosine_batched_vs_single':cos});assert cos>.999
np.savez_compressed(OUT/'prott5_mean_embeddings.npz',sequences=np.array(seqs),embeddings=vec)
manifest={'encoder':'Rostlab/prot_t5_xl_half_uniref50-enc','weights_path':str(WEIGHTS),'weights_snapshot':WEIGHTS.parent.name,'config_snapshot':META.name,'weights_sha256':sha(WEIGHTS),'config_sha256':sha(META/'config.json'),'tokenizer_sha256':sha(META/'spiece.model'),'input_sha256':sha(SRC),'sequence_count':len(seqs),'dimension':1024,'pooling':'residue mean, no EOS or padding','adapters':False,'fine_tuning':False,'all_parameters_frozen':True,'dtype':str(next(model.parameters()).dtype),'batch_size':bs,'device':torch.cuda.get_device_name(0),'elapsed_seconds':time.time()-start,'batch_consistency_checks':checks,'cache_sha256':sha(OUT/'prott5_mean_embeddings.npz')}
(OUT/'embedding_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Embedding cache and provenance saved.',flush=True)
