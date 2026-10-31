import os,sys,json
from pathlib import Path
r=Path('D:/dachuang2026/GHXTox');sys.path[:0]=[str(r/'analysis/repository_sync_20261026/validation_numpy1_clean'),str(r/'src'),str(r/'analysis/group_weight_trial_20261027')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from run_sequence_trial import load_model
from ghxtox.lora_mamba_1d import remove_inactive_lora,make_training_optimizer,train_epoch,validate_no_lora_checkpoint_payload,LoRALinear
torch.set_num_threads(2)
model,tok,_=load_model(42);assert remove_inactive_lora(model.plm.encoder)==144
cfg=json.loads((r/'configs/no_lora_mamba_1d_autodl.json').read_text(encoding='utf-8'));rows=json.loads((r/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json').read_text(encoding='utf-8'));seq=[rows[i]['sequence'] for i in [0,100]];enc=tok([' '.join(x) for x in seq],padding=True,return_tensors='pt')
batch={'input_ids':enc['input_ids'],'attention_mask':enc['attention_mask'],'length':torch.tensor([len(x) for x in seq]),'label':torch.tensor([float(rows[i]['label']) for i in [0,100]])}
before=model.classifier.weight.detach().clone();opt=make_training_optimizer(model,cfg);scaler=torch.cuda.amp.GradScaler(init_scale=128.)
train_epoch(model,[batch],opt,torch.nn.BCEWithLogitsLoss(),torch.device('cuda'),True,scaler)
assert not torch.equal(before,model.classifier.weight.detach())
assert not any(isinstance(m,LoRALinear) for m in model.modules())
assert all(p.grad is None for p in model.plm.parameters())
state={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad};validate_no_lora_checkpoint_payload({'config':cfg,'model_state':state},cfg,model=model)
result={'passed':True,'real_prott5':True,'no_adapter_modules':True,'frozen_encoder_no_grad':True,'classifier_updated':True,'new_checkpoint_contract_passed':True,'batch_size':2}
(r/'analysis/remove_lora_20261028/training_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result,flush=True)
