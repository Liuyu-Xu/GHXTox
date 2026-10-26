"""Frozen SaProt structure-aware versus all-structure-masked paired features."""
from pathlib import Path
import os,json,hashlib,time
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import numpy as np
from transformers import EsmConfig,EsmForMaskedLM,EsmTokenizer
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis/saprot_input_audit_20261026';M=A/'model'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=read(A/'structure_sequences.json');cachepath=R/'runs/prottox_tsk_v1/features.pt';cache=torch.load(cachepath,map_location='cpu',weights_only=False);by={r['sequence']:r for r in cache['records']}
base=[]
for r in rows:
 c=by[r['sequence']];assert int(c['label'])==r['label'];base.append(c['tristat'].tolist()+c['physchem'].tolist())
base=np.array(base,dtype=np.float32);assert base.shape==(6387,3081);del cache,by
torch.set_num_threads(2);torch.manual_seed(20261026);assert torch.cuda.is_available()
config=EsmConfig.from_json_file(str(M/'config.json'));tokenizer=EsmTokenizer.from_pretrained(str(M),local_files_only=True);model=EsmForMaskedLM(config)
state=torch.load(M/'pytorch_model.bin',map_location='cpu',weights_only=True)
legacy_extra=set(state)-set(model.state_dict())
assert legacy_extra<={'esm.embeddings.position_ids','esm.embeddings.position_embeddings.weight'}
assert not legacy_extra or config.position_embedding_type=='rotary'
for key in legacy_extra:state.pop(key)
model.load_state_dict(state,strict=True);del state
model=model.eval().requires_grad_(False).half().to('cuda');dim=config.hidden_size;assert dim==480
def encode(batch,key):
 strings=[r[key] for r in batch];tokens=tokenizer(strings,return_tensors='pt',padding=True)
 assert all(int(tokens['attention_mask'][i].sum())==r['length']+2 for i,r in enumerate(batch))
 assert not bool((tokens['input_ids']==tokenizer.unk_token_id).any())
 with torch.inference_mode():h=model.esm(**{k:v.to('cuda') for k,v in tokens.items()}).last_hidden_state
 return np.array([h[i,1:1+r['length']].float().mean(0).cpu().tolist() for i,r in enumerate(batch)],dtype=np.float32)
struct=np.zeros((6387,dim),dtype=np.float32);masked=struct.copy();start=time.time()
for i in range(0,len(rows),64):
 batch=rows[i:i+64];struct[i:i+len(batch)]=encode(batch,'combined');masked[i:i+len(batch)]=encode(batch,'all_structure_masked')
 if i%1024==0 or i+64>=len(rows):print(f'Encoded paired SaProt inputs {min(i+64,6387)}/6387',flush=True)
identical=np.array([r['combined']==r['all_structure_masked'] for r in rows]);difference=float(np.max(abs(struct[identical]-masked[identical])))
assert difference<.02
# Identical inputs must not gain an artificial feature difference from numerical batch effects.
struct[identical]=masked[identical]
assert np.isfinite(struct).all() and np.isfinite(masked).all() and np.isfinite(base).all()
out=O/'frozen_features.npz';np.savez_compressed(out,sample_ids=np.array([r['sample_id'] for r in rows]),sequences=np.array([r['sequence'] for r in rows]),labels=np.array([r['label'] for r in rows]),folds=np.array([r['fold'] for r in rows]),groups=np.array([r['group_id'] for r in rows]),base=base,saprot_struct=struct,saprot_masked=masked)
manifest=dict(n=6387,base_dim=3081,saprot_dim=dim,encoder='westlake-repl/SaProt_35M_AF2',revision=read(A/'hf_model_metadata.json')['sha'],weights_sha256=sha(M/'pytorch_model.bin'),input_sequences_sha256=sha(A/'structure_sequences.json'),base_cache_sha256=sha(cachepath),output_sha256=sha(out),encoder_frozen=True,backbone_reconstructed_CB=True,plddt_mask_threshold=70,allmasked_rows=int(identical.sum()),allmasked_numerical_batch_max_difference=difference,allmasked_features_set_identical=True,pooling='mean residues excludingCLS_EOS_padding',dtype='float16 inference,float32 saved',elapsed_seconds=time.time()-start,device=torch.cuda.get_device_name(0),unused_legacy_rotary_position_keys=sorted(legacy_extra))
(O/'embedding_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))
