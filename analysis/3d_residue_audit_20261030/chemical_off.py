import csv,json,sys
from pathlib import Path
import numpy as np,torch
root=Path(r'D:\dachuang2026\GHXTox');base=Path(r'D:\dachuang2026\GHXTox-ziliao');out=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'src'))
from ghxtox.data import collate_peptides
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device
torch.set_num_threads(2)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processed=torch.load(base/'train_chemical_sites_final_esm2_multisource_1_30_full2795.pt',map_location='cpu',weights_only=False)
records={str(r['sample_id']):r for r in processed['records']}
with (root/'data/folds/train_cdhit080_fallback_folds.csv').open(encoding='utf8',newline='') as f: folds=list(csv.DictReader(f))
with (base/'original_lineage_residual_package_20260809_v1/evidence/original_lineage_selected_base_oof_v1/full_predictions.csv').open(encoding='utf8',newline='') as f: ref={r['sample_id']:float(r['toxicity_probability']) for r in csv.DictReader(f)}
idx=json.loads((root/'configs/provenance/historical_selected_base_v1/index.json').read_text())
assert set(records)=={r['sample_id'] for r in folds}==set(ref)
for e in idx['entries']:
 fold=e['fold'];dest=out/f'chemical_off_fold{fold}.npz'
 if dest.exists(): print('already exists',fold,flush=True);continue
 cp=e['checkpoint'].replace('dachuang2026\\ziliao','dachuang2026\\GHXTox-ziliao');ck=torch.load(cp,map_location='cpu',weights_only=False);model=GHXToxModel(ck['config']).to(device);model.load_state_dict(ck['model_state'],strict=True);model.eval()
 model.chemical_site_branch.register_forward_hook(lambda module, args, output: (torch.zeros_like(output[0]), output[1]))
 rows=[r for r in folds if int(r['fold'])==fold];features=[];pred=[]
 with torch.inference_mode():
  for start in range(0,len(rows),16):
   chunk=rows[start:start+16];batch=move_batch_to_device(collate_peptides([records[r['sample_id']] for r in chunk],include_structure=True,include_atom=False),device);o=model(batch);h=o['spatial_nodes'];mask=batch['mask'].unsqueeze(-1);mean=(h*mask).sum(1)/mask.sum(1).clamp_min(1);maximum=h.masked_fill(~mask,-1e4).max(1).values;features.append(torch.cat([mean,maximum],-1).cpu().numpy());pred.extend(torch.sigmoid(o['logits']).cpu().tolist())
 diff=np.abs(np.asarray(pred)-np.asarray([ref[r['sample_id']] for r in rows]));print('fold',fold,'n',len(rows),'max_diff',float(diff.max()),flush=True)
 np.savez_compressed(dest,ids=np.array([r['sample_id'] for r in rows]),features=np.concatenate(features),probability=np.array(pred),max_diff=diff.max());del model
