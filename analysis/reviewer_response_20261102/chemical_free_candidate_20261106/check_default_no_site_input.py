import csv,os,sys,tempfile,torch,numpy as np
from pathlib import Path
from ghxtox.multidimensional_default import DEFAULT_STRUCTURE_CHECKPOINT,validate_public_default_config
from ghxtox.structure_inference import predict_structure
from ghxtox.utils import load_json
root=Path(r'D:\dachuang2026\GHXTox');validate_public_default_config(load_json(root/'configs/default_multidimensional_fusion.json'))
p=torch.load(r'D:\dachuang2026\GHXTox-ziliao\test1_chemical_sites_final_esm2.pt',map_location='cpu',weights_only=False);p['records']=p['records'][:16]
for row in p['records']:
 for k in list(row):
  if k.startswith('chemical_site_'):del row[k]
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'no_sites.pt';torch.save(p,path);v=predict_structure(path,root/DEFAULT_STRUCTURE_CHECKPOINT,device=torch.device('cuda'),batch_size=8)
with (root/'runs/default_no_chemical_20261106/test1_structure_predictions.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))[:16]
assert v['sample_ids']==[r['sample_id'] for r in rows] and not v['fallback_used'];dif=np.max(np.abs(v['base_probabilities']-np.array([float(r['structure_probability']) for r in rows])));assert dif<1e-5
print('DEFAULT 3D NO SITE INPUT: 16 samples, max error',dif,'fallback',v['fallback_used'])
