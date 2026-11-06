"""Actual GHXTox structure A/B/C, fixed historical 1D for retrospective fusion."""
import os,sys,pathlib,json,csv,hashlib,time,random,copy,argparse,gc
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
sys.path.insert(0,str(R/'src'))
os.environ['OMP_NUM_THREADS']='2';os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from ghxtox.models import GHXToxModel
from ghxtox.data import collate_peptides
from ghxtox.utils import move_batch_to_device,set_seed
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,x):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def digest(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

TARGETS={'belongs to the mcd family','belongs to the non-disulfide-bridged peptide (ndbp) superfamily'}
def sample_weight(meta,arm):
 if arm=='A' or meta['label']!=1 or meta['cys_count']>=4:return 1.0
 parents=meta['current_family_sidecar']['parent_families']
 if arm=='C' and len(parents)==1 and parents[0] in TARGETS:return 2.0
 return 1.5

def prepare():
 cache=pathlib.Path('D:/dachuang2026/GHXTox-ziliao/train_chemical_sites_final_esm2_multisource_1_30_full2795.pt')
 checkpoint=pathlib.Path('D:/dachuang2026/GHXTox-ziliao/best_model(1).pt')
 rolefile=R/'analysis/upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027/runs/structure_candidates_20261022/protocol/nested_roles.csv'
 metafile=R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json'
 seqfile=R/'runs/lora_mamba_full/oof_calibrated.json'
 meta=load(metafile);seq=load(seqfile)
 data=torch.load(cache,map_location='cpu',weights_only=False);records=data['records']
 assert len(records)==len(meta)==len(seq['oof'])==6387
 for i,(r,m) in enumerate(zip(records,meta)):
  assert r['sample_id']==m['sample_id']==seq['sample_ids'][i] and r['sequence']==m['sequence'] and int(r['label'])==m['label']==seq['labels'][i]
  assert int(seq['folds'][i])==m['fold']
  assert r['plm_features'].shape==(len(r['sequence']),1280)
 # Public single-structure mode. Retain unused module architecture, do not activate training conformers.
 records=[{**r,'conformer_available':False} for r in records]
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
 config=copy.deepcopy(ck['config'])
 audit=GHXToxModel(config);audit.load_state_dict(ck['model_state'],strict=True)
 params=sum(p.numel() for p in audit.parameters());del audit,ck
 config['train']={'epochs':6,'batch_size':32,'learning_rate':.0005,'weight_decay':.0001,'pos_weight':'none','coord_noise_std':0.0,'all_used_parameters_trainable':True,'initial_checkpoint':None}
 roles={f:{r:[] for r in ['train','validation','calibration','test']} for f in range(5)}
 with rolefile.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):
   i=int(r['source_index']);o=int(r['outer_fold']);m=meta[i]
   assert r['sample_id']==m['sample_id'] and r['sequence']==m['sequence'] and int(r['label'])==m['label'] and r['group_id']==m['group_id']
   roles[o][r['role']].append(i)
 for fold,rr in roles.items():
  assert sorted(i for ix in rr.values() for i in ix)==list(range(6387))
  groups={k:{meta[i]['group_id'] for i in ix} for k,ix in rr.items()}
  for k in groups:
   for v in groups:
    if k!=v:assert not groups[k]&groups[v]
  assert all(meta[i]['fold']==fold for i in rr['test'])
 target=[m['sample_id'] for m in meta if sample_weight(m,'C')==2]
 assert len(target)==11
 model_sources=[R/'src/ghxtox/models/dual_modal.py',R/'src/ghxtox/models/layers.py',R/'src/ghxtox/models/chemical_sites.py',R/'src/ghxtox/data.py']
 manifest={'cache':{'path':str(cache),'bytes':cache.stat().st_size,'sha256':sha(cache)},'config_checkpoint':{'path':str(checkpoint),'sha256':sha(checkpoint),'weights_used_only_for_strict_architecture_check':True},'roles':{'path':str(rolefile),'sha256':sha(rolefile)},'meta':{'path':str(metafile),'sha256':sha(metafile)},'frozen_1d':{'path':str(seqfile),'sha256':sha(seqfile),'historical_Platt':seq['platt']},'source_sha256':{str(f):sha(f) for f in model_sources},'model_params':params,'target_MCD_NDBP':target,'group_isolation_passed':True,'cuda':torch.cuda.get_device_name(0),'torch':torch.__version__,'numpy':np.__version__,'roles_by_fold':{str(f):{r:len(ix) for r,ix in rr.items()} for f,rr in roles.items()}}
 save(P/'asset_and_role_manifest.json',manifest);save(P/'model_config.json',config)
 side=[]
 for fold,rr in roles.items():
  for role,ix in rr.items():
   for i in ix:
    m=meta[i];side.append({'outer_fold':fold,'role':role,'source_index':i,'sample_id':m['sample_id'],'label':m['label'],'group_id':m['group_id'],'uniform_flag':m['uniform_flag'],'weights_if_training':{a:sample_weight(m,a) if role=='train' else 1.0 for a in ['A','B','C']}})
 save(P/'role_weight_contract.json',side)
 return records,meta,config,roles,seq

def collate(batch):return collate_peptides(batch,include_structure=True,include_atom=False)
def evaluate(model,records,indices):
 model.eval();output=[]
 with torch.inference_mode():
  for start in range(0,len(indices),32):
   ix=indices[start:start+32];batch=move_batch_to_device(collate([records[i] for i in ix]),torch.device('cuda'))
   pred=model(batch);logits=pred['base_logits'];assert torch.allclose(logits,pred['logits'])
   for i,v in zip(ix,logits.cpu().tolist()):output.append({'source_index':i,'logit':v})
 return output

def train_arm(records,meta,config,roles,seq,fold,arm,seed,smoke=False):
 out=P/('smoke' if smoke else 'runs')/f'seed{seed}'/f'fold{fold}'/arm
 if (out/'result.json').exists() and not smoke:
  print('reuse completed',fold,arm,flush=True);return load(out/'result.json')
 rr=roles[fold];set_seed(seed+fold*1000)
 model=GHXToxModel(config).cuda();initial=digest(model)
 optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
 weights={m['sample_id']:sample_weight(m,arm) for m in meta}
 best=float('inf');history=[];best_state=None;best_epoch=None
 start=time.time();torch.cuda.reset_peak_memory_stats()
 # Same initialization, training order and optimizer steps across arms.
 for epoch in range(1,(1 if smoke else 6)+1):
  gen=torch.Generator().manual_seed(seed+fold*1000+epoch*10000)
  order=[rr['train'][i] for i in torch.randperm(len(rr['train']),generator=gen).tolist()]
  if smoke:
   # A real diverse training-role batch to exercise C's family weights, never outer-test records.
   chosen=[i for i in rr['train'] if sample_weight(meta[i],'C')==2]
   chosen += [i for i in rr['train'] if sample_weight(meta[i],'B')==1.5 and i not in chosen][:8]
   chosen += [i for i in order if i not in chosen]
   order=chosen[:64]
  model.train();total=0.0;n=0;steps=0
  for st in range(0,len(order),32):
   ix=order[st:st+32];batch=move_batch_to_device(collate([records[i] for i in ix]),torch.device('cuda'))
   w=torch.tensor([weights[records[i]['sample_id']] for i in ix],device='cuda')
   optimizer.zero_grad(set_to_none=True);output=model(batch)
   assert torch.allclose(output['logits'],output['base_logits'])
   per=F.binary_cross_entropy_with_logits(output['base_logits'],batch['labels'],reduction='none')
   loss=(w*per).sum()/w.sum();assert torch.isfinite(loss)
   loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
   assert np.isfinite(norm);optimizer.step()
   total+=float(loss.item())*len(ix);n+=len(ix);steps+=1
  if smoke:
   changed=digest(model)!=initial
   result={'arm':arm,'outer_fold':fold,'initial_sha256':initial,'actual_gradient_steps':steps,'n_train_examples_used':n,'loss':total/n,'parameters_changed':changed,'seconds':time.time()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'validation_or_outer_test_used':False,'not_a_performance_trial':True}
   save(out/'result.json',result);print('SMOKE',result,flush=True)
   del model,optimizer;gc.collect();torch.cuda.empty_cache();return result
  val=evaluate(model,records,rr['validation'])
  vl=np.mean([np.logaddexp(0,r['logit'])-meta[r['source_index']]['label']*r['logit'] for r in val])
  history.append({'epoch':epoch,'train_weighted_BCE':total/n,'validation_unweighted_BCE':float(vl),'steps':steps})
  if vl<best:
   best=float(vl);best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  print(f'fold={fold} arm={arm} epoch={epoch}/6 train={total/n:.4f} val={vl:.4f} seconds={time.time()-start:.1f}',flush=True)
 model.load_state_dict(best_state,strict=True)
 calibration=evaluate(model,records,rr['calibration'])
 X=np.array([[r['logit']] for r in calibration]);y=np.array([meta[r['source_index']]['label'] for r in calibration])
 platt=LogisticRegression(C=1.0,max_iter=1000).fit(X,y)
 test=evaluate(model,records,rr['test']);pp=platt.predict_proba(np.array([[r['logit']] for r in test]))[:,1]
 predictions=[]
 for r,p in zip(test,pp):
  i=r['source_index'];sf=float(seq['oof'][i]);fused=.72*sf+.28*float(p)
  predictions.append({'sample_id':meta[i]['sample_id'],'source_index':i,'fold':fold,'label':meta[i]['label'],'structure_logit':r['logit'],'structure_probability':float(p),'structure_prediction':int(p>=.5),'sequence_probability_frozen':sf,'fusion_probability':fused,'fusion_prediction':int(fused>=.47022256531284556)})
 result={'arm':arm,'seed':seed,'outer_fold':fold,'initial_sha256':initial,'history':history,'selected_epoch':best_epoch,'platt':{'slope':float(platt.coef_[0,0]),'intercept':float(platt.intercept_[0])},'seconds':time.time()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'role_counts':{k:len(v) for k,v in rr.items()},'calibration_records':calibration,'predictions':predictions}
 out.mkdir(parents=True,exist_ok=True)
 torch.save({'model_state':best_state,'config':config,'arm':arm,'seed':seed,'outer_fold':fold,'selected_epoch':best_epoch,'training_indices':rr['train']},out/'best_model.pt')
 save(out/'result.json',result);print('COMPLETE',fold,arm,'seconds',round(result['seconds'],1),flush=True)
 del model,optimizer,best_state;gc.collect();torch.cuda.empty_cache();return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--folds',default='0');ap.add_argument('--arms',default='A,B,C');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--acknowledge-structure-only-limit',action='store_true')
 args=ap.parse_args()
 if not args.smoke and args.folds!='0' and not args.acknowledge_structure_only_limit:
  raise SystemExit('Structure-only fixed fusion cannot rescue8/11 target samples. Use the sequence experiment for the main A/B/C hypothesis; explicit acknowledgement required to extend this diagnostic.')
 records,meta,config,roles,seq=prepare()
 for fold in map(int,args.folds.split(',')):
  for arm in args.arms.split(','):
   assert arm in ['A','B','C'];train_arm(records,meta,config,roles,seq,fold,arm,args.seed,args.smoke)
