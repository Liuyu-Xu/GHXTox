import csv,sys,time,statistics,json,pathlib,torch
from functools import partial
from torch.utils.data import DataLoader
R=pathlib.Path(r'D:\dachuang2026\GHXTox');sys.path.insert(0,str(R/'src'))
from ghxtox.data import PeptideTensorDataset,collate_peptides
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device
O=R/'analysis/reviewer_response_20261102/chemical_free_candidate_20261106';old=torch.load(R/'runs/multisource_1_30_full2795_rank_stable_candidate/best_model.pt',map_location='cpu',weights_only=False);new=torch.load(O/'candidate_no_chemical.pt',map_location='cpu',weights_only=False);models=[]
for ck in (old,new):
 m=GHXToxModel(ck['config']).cuda();m.load_state_dict(ck['model_state'],strict=True);m.eval();models.append(m)
ds=PeptideTensorDataset(r'D:\dachuang2026\GHXTox-ziliao\test1_chemical_sites_final_esm2.pt',require_labels=True);loader=DataLoader(ds,batch_size=16,shuffle=False,collate_fn=partial(collate_peptides,include_structure=True,include_atom=False))
with (R/'runs/default_multidimensional_fusion_single_structure/test1_predictions.csv').open(encoding='utf-8-sig',newline='') as f:p1=[float(r['p_1d']) for r in csv.DictReader(f)]
def run(model):
 result=[];torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():
  for batch in loader:
   b=move_batch_to_device(batch,torch.device('cuda'));result.extend(torch.sigmoid(model(b)['base_logits']).cpu().tolist())
 fused=[.72*a+.28*b for a,b in zip(p1,result)];assert len(fused)==len(ds);torch.cuda.synchronize();return time.perf_counter()-start
for model in models:run(model)
obs={'original':[],'candidate':[]}
for i in range(3):
 for name,model in ([('original',models[0]),('candidate',models[1])] if i%2==0 else [('candidate',models[1]),('original',models[0])]):obs[name].append(run(model))
res={'test1_n':len(ds),'batch':16,'gpu':'RTX 4060 laptop','scope':'cached already-loaded structures; includes DataLoader/collate CPU, transfer, 3D forward, fixed fusion from cached 1D probabilities; excludes ProtT5 1D inference, structure generation, disk read','seconds':obs,'median_seconds':{k:statistics.median(v) for k,v in obs.items()}}
(O/'speed.json').write_text(json.dumps(res,indent=2),encoding='utf8');print(json.dumps(res,indent=2))
