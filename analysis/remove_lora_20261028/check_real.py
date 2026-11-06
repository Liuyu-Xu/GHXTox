import os,sys,json,gc
from pathlib import Path
r=Path('D:/dachuang2026/GHXTox');sys.path[:0]=[str(r/'analysis/repository_sync_20261026/validation_numpy1_clean'),str(r/'src'),str(r/'analysis/group_weight_trial_20261027')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from run_sequence_trial import load_model
from ghxtox.lora_mamba_1d import remove_inactive_lora,LoRALinear,validate_public_checkpoint_payload
torch.set_num_threads(2)
model,tok,cfg=load_model(42)
rows=json.loads((r/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json').read_text(encoding='utf-8'))
seq=[rows[i]['sequence'] for i in [0,100,1000,3000]]
enc=tok([' '.join(x) for x in seq],padding=True,return_tensors='pt');ids=enc['input_ids'].cuda();mask=enc['attention_mask'].cuda();lengths=torch.tensor([len(x) for x in seq],device='cuda')
results=[]
for seed in [42,123,2025]:
 if seed!=42:
  del model;gc.collect();torch.cuda.empty_cache();model,tok,cfg=load_model(42)
 cp_path=r/f'runs/lora_mamba_full/full_seed{seed}.pt';cp=torch.load(cp_path,map_location='cpu',weights_only=False)
 reference=json.loads((r/'configs/lora_mamba_1d_autodl.json').read_text(encoding='utf-8'))
 validate_public_checkpoint_payload(cp,reference,expected_seed=seed,model=model,checkpoint_path=cp_path)
 model.load_state_dict(cp['model_state'],strict=False);model.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):before=model(ids,mask,lengths)
 removed=remove_inactive_lora(model.plm.encoder)
 assert removed==144 and not any(isinstance(x,LoRALinear) for x in model.modules())
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):after=model(ids,mask,lengths)
 diff=(before-after).abs().max().item();assert diff==0
 results.append({'seed':seed,'removed_adapters':removed,'samples':len(seq),'max_abs_logit_difference':diff})
 print(results[-1],flush=True)
(r/'analysis/remove_lora_20261028/real_checkpoint_parity.json').write_text(json.dumps({'passed':True,'results':results,'scope':'four sequences, three real checkpoints, same GPU fp16 inference'},indent=2),encoding='utf-8')
