import copy,csv,hashlib,json,pathlib,sys,time,statistics
from functools import partial
import torch
from torch.utils.data import DataLoader
R=pathlib.Path(r'D:\dachuang2026\GHXTox');sys.path.insert(0,str(R/'src'))
from ghxtox.data import PeptideTensorDataset,collate_peptides,validate_plm_feature_dim
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device
O=R/'analysis/reviewer_response_20261102/chemical_free_candidate_20261106';OLD=R/'runs/multisource_1_30_full2795_rank_stable_candidate/best_model.pt';P=R/'analysis/reviewer_response_20261102/nested_1d_fusion_20261106/chemical_fold_audit';dest=O/'candidate_no_chemical.pt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD)=='a3c5d3c2d9da80ab56bc1e7b375ee4e9e34bb1bede11cfd20f367c5bc24db5c1'
ck=torch.load(OLD,map_location='cpu',weights_only=False);cfg=copy.deepcopy(ck['config']);assert cfg['model']['chemical_site_branch'] is True;cfg['model']['chemical_site_branch']=False
state=ck['model_state'];removed={k:v for k,v in state.items() if k.startswith('chemical_site_branch.')};retained={k:v for k,v in state.items() if not k.startswith('chemical_site_branch.')};assert len(removed)==28
candidate={'model_state':retained,'config':cfg,'epoch':ck.get('epoch'),'artifact_type':'ghxtox_no_chemical_exploratory_candidate','provenance':{'original_checkpoint_sha256':sha(OLD),'changes':'removed chemical_site_branch state and set chemical_site_branch=false; other weights unchanged; no training','probability_source':'base_logits'}}
model=GHXToxModel(cfg).cuda();model.load_state_dict(retained,strict=True);model.eval();assert model.chemical_site_branch is None
if dest.exists():raise FileExistsError(dest)
torch.save(candidate,dest);reloaded=torch.load(dest,map_location='cpu',weights_only=False);assert all(torch.equal(v,reloaded['model_state'][k]) for k,v in retained.items());assert len(reloaded['model_state'])==len(retained)
summary={'source_sha256':sha(OLD),'candidate_sha256':sha(dest),'retained_tensors':len(retained),'removed_tensors':len(removed),'removed_parameters':sum(t.numel() for t in removed.values()),'source_bytes':OLD.stat().st_size,'candidate_bytes':dest.stat().st_size,'datasets':{}}
for split in ('test1','test2'):
 dpath=pathlib.Path(r'D:\dachuang2026\GHXTox-ziliao')/(split+'_chemical_sites_final_esm2.pt');ds=PeptideTensorDataset(dpath,require_labels=True);validate_plm_feature_dim(ds.records,1280,dpath)
 with (P/(split+'_raw_3d.csv')).open(encoding='utf-8-sig',newline='') as f:ref=list(csv.DictReader(f))
 with (P/(split+'_paired.csv')).open(encoding='utf-8-sig',newline='') as f:fused=list(csv.DictReader(f))
 assert len(ds)==len(ref)==len(fused) and all(a['sample_id']==b['sample_id']==c['sample_id'] and a['sequence']==b['sequence']==c['sequence'] and int(a['label'])==int(b['label'])==int(c['label']) for a,b,c in zip(ds.records,ref,fused))
 loader=DataLoader(ds,batch_size=16,shuffle=False,collate_fn=partial(collate_peptides,include_structure=True,include_atom=False));out=[];t=time.perf_counter()
 with torch.inference_mode():
  for batch in loader:
   batch=move_batch_to_device(batch,torch.device('cuda'));ps=torch.sigmoid(model(batch)['base_logits']).cpu().tolist()
   for sid,score in zip(batch['sample_id'],ps):out.append((sid,score))
 torch.cuda.synchronize();seconds=time.perf_counter()-t;assert len(out)==len(ref);diff=[];fpdiff=[];changes=0
 for (sid,p),x,y in zip(out,ref,fused):
  assert sid==x['sample_id'];diff.append(abs(p-float(x['p_3d_no_site'])));fp=.72*float(y['p1d'])+.28*p;fpdiff.append(abs(fp-float(y['p_fused_no_site'])));changes+=int((fp>=.47022256531284556)!=(int(y['pred_no_site'])==1))
 assert max(diff)<1e-5 and max(fpdiff)<1e-5 and changes==0
 with (O/(split+'_candidate_3d.csv')).open('w',encoding='utf8',newline='') as f:w=csv.writer(f);w.writerow(['sample_id','structure_probability']);w.writerows(out)
 summary['datasets'][split]={'n':len(out),'max_3d_diff':max(diff),'max_fused_diff':max(fpdiff),'decision_mismatches':changes,'data_loader_plus_3d_gpu_seconds':seconds,'input_cache_sha256':sha(dpath)}
 print(split,summary['datasets'][split],flush=True)
(O/'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
