"""Reuse frozen mean features; encode only missing auxiliary sequences locally."""
from pathlib import Path
import os,json,hashlib,time
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;A=R/'analysis/hemolysis_auxiliary_feasibility_20261026'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(2)
main=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json');aux=read(A/'auxiliary_source_records.json');folds=read(A/'fold_manifests.json')
used={i for f in folds for i in f['aux_train_ids']};aux=[r for r in aux if r['aux_id'] in used]
cachepath=R/'runs/prottox_tsk_v1/features.pt';c=torch.load(cachepath,map_location='cpu',weights_only=False)
byseq={r['sequence']:np.array(r['tristat'][:1024].tolist(),dtype=np.float32) for r in c['records']};del c
assert len(byseq)==6387
zpath=R/'analysis/frozen_prott5_environment_20261025/prott5_mean_embeddings.npz';z=np.load(zpath)
other={str(s):e for s,e in zip(z['sequences'],z['embeddings'])};common=sorted(byseq.keys()&other.keys())
cos=[float(byseq[s]@other[s]/np.linalg.norm(byseq[s])/np.linalg.norm(other[s])) for s in common];assert min(cos)>.9999
for s,e in other.items():
 if s not in byseq:byseq[s]=e
needed=sorted({r['sequence'] for r in aux}-byseq.keys());start=time.time();checks=[]
print(f'Reusing6387 main means; auxiliary union{len(aux)}, missing{len(needed)} sequences.',flush=True)
if needed:
 from safetensors.torch import load_file
 from transformers import T5Config,T5EncoderModel,T5Tokenizer
 oldmanifest=read(R/'analysis/frozen_prott5_environment_20261025/embedding_manifest.json');weights=Path(oldmanifest['weights_path']);meta=weights.parents[1]/oldmanifest['config_snapshot']
 tokenizer=T5Tokenizer(vocab_file=str(meta/'spiece.model'),legacy=True,do_lower_case=False);config=T5Config.from_pretrained(str(meta),local_files_only=True)
 assert torch.cuda.is_available()
 with torch.device('meta'):model=T5EncoderModel(config)
 state=load_file(str(weights),device='cuda')
 if 'shared.weight' not in state:state['shared.weight']=state['encoder.embed_tokens.weight']
 if 'encoder.embed_tokens.weight' not in state:state['encoder.embed_tokens.weight']=state['shared.weight']
 model.load_state_dict(state,strict=True,assign=True);del state
 model.tie_weights();model.eval();model.requires_grad_(False)
 def encode(chunk):
  tokens=tokenizer([' '.join(s) for s in chunk],add_special_tokens=True,padding=True,return_tensors='pt')
  assert all(int(tokens['attention_mask'][i].sum())==len(s)+1 for i,s in enumerate(chunk))
  with torch.inference_mode():h=model(**{k:v.to('cuda') for k,v in tokens.items()}).last_hidden_state
  return [np.array(h[i,:len(s)].float().mean(0).cpu().tolist(),dtype=np.float32) for i,s in enumerate(chunk)]
 for i in range(0,len(needed),8):
  chunk=needed[i:i+8]
  byseq.update(zip(chunk,encode(chunk)))
 for s in common[:3]:
  e=encode([s])[0];co=float(e@byseq[s]/np.linalg.norm(e)/np.linalg.norm(byseq[s]));assert co>.9999;checks.append(dict(sequence=s,cosine_reencoded_vs_main=co))
 del model;torch.cuda.empty_cache()
out=O/'frozen_mean_features.npz'
np.savez_compressed(out,main_ids=np.array([r['sample_id'] for r in main]),main_sequences=np.array([r['sequence'] for r in main]),main_x=np.array([byseq[r['sequence']] for r in main]),aux_ids=np.array([r['aux_id'] for r in aux]),aux_sequences=np.array([r['sequence'] for r in aux]),aux_x=np.array([byseq[r['sequence']] for r in aux]))
manifest=dict(encoder='Rostlab/prot_t5_xl_half_uniref50-enc',parameters_frozen=True,pooling='residue mean excluding padding/EOS',main_cache_sha256=sha(cachepath),aux_old_cache_sha256=sha(zpath),main_cache_overlap_checks=len(common),old_cache_min_cosine=min(cos),new_aux_sequences_encoded=len(needed),aux_sequences=len(aux),reencoded_checks=checks,elapsed_seconds=time.time()-start,output_sha256=sha(out),no_encoder_training=True)
(O/'embedding_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
