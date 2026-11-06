"""Fixed retrospective comparison; calibration uses training OOF only."""
import csv,json,hashlib,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'analysis/reviewer_response_20261102/three_seed_full_autodl_20261106/src'))
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
from transformers import T5Config,T5EncoderModel,T5Tokenizer
from ghxtox.lora_mamba_1d import build_model
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def logit(p):
 q=np.clip(p,1e-6,1-1e-6);return np.log(q)-np.log1p(-q)
train=read(ROOT/'data/folds/train_cdhit080_fallback_folds.csv');assert len(train)==6387 and len({r['sample_id'] for r in train})==6387
ids=[r['sample_id'] for r in train];y=np.array([int(r['label']) for r in train]);old=(42,123,2025);new=(314,2718,2026);streams=[]
for seed in old:
 p=ROOT/f'runs/lora_mamba_full/oof_seed{seed}.json';d=json.loads(p.read_text(encoding='utf8'));assert d['seed']==seed and d['sample_ids']==ids and d['labels']==y.tolist() and d['folds']==[int(r['fold']) for r in train];streams.append(np.array(d['oof'],dtype=float))
for seed in new:
 by={}
 for fold in range(5):
  base=ROOT/f'analysis/reviewer_response_20261102/six_seed_received_20261106/six_seed_new_results/runs/seed{seed}_fold{fold}';meta=json.loads((base/'result.json').read_text());p=base/'test_predictions.csv';assert sha(p)==meta['predictions_sha256']['test'] and meta['seed']==seed and meta['fold']==fold and meta['status']=='complete'
  for r in read(p):
   assert r['sample_id'] not in by and int(r['seed'])==seed and int(r['fold'])==fold;by[r['sample_id']]=r
 assert set(by)==set(ids)
 for r in train:
  x=by[r['sample_id']];assert r['sequence']==x['sequence'] and r['label']==x['label'] and r['group_id']==x['group_id'] and r['fold']==x['fold']
 streams.append(np.array([float(by[i]['probability']) for i in ids]))
oof=np.array(streams);assert oof.shape==(6,6387) and np.isfinite(oof).all() and ((oof>=0)&(oof<=1)).all()
cal={}
for count in (3,6):
 lr=LogisticRegression(C=1e6,max_iter=1000).fit(logit(oof[:count].mean(axis=0))[:,None],y);assert lr.coef_[0,0]>0;cal[count]=lr
print('OOF calibration fixed',[(k,float(v.coef_[0,0]),float(v.intercept_[0])) for k,v in cal.items()],flush=True)
path=ROOT/'analysis/upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027/Rostlab/prot_t5_xl_half_uniref50-enc';assert sha(path/'pytorch_model.bin')=='7f51ba885541c7dc569d46b796af57cc7a2ba7945107dced4f19d1b5ec091157'
cfg=json.loads((ROOT/'analysis/reviewer_response_20261102/three_seed_full_autodl_20261106/config/model.json').read_text());cfg['plm']=str(path)
torch.set_num_threads(2);assert torch.cuda.is_available()
with torch.device('meta'):plm=T5EncoderModel(T5Config.from_pretrained(str(path),local_files_only=True))
st=torch.load(path/'pytorch_model.bin',map_location='cpu',weights_only=True)
if 'shared.weight' not in st:st['shared.weight']=st['encoder.embed_tokens.weight']
if 'encoder.embed_tokens.weight' not in st:st['encoder.embed_tokens.weight']=st['shared.weight']
st={k:v.to('cuda',dtype=torch.float16) for k,v in st.items()};plm.load_state_dict(st,strict=True,assign=True);del st;plm.tie_weights();model=build_model(plm,cfg,torch.device('cuda'));tok=T5Tokenizer.from_pretrained(str(path),local_files_only=True,do_lower_case=False)
refs=read(ROOT/'analysis/dataset_comparison_20261025/all_records.csv');samples=[r for r in refs if r['dataset']=='GHXTox' and r['split']=='test1'];assert len(samples)==1126 and len({r['sequence'] for r in samples})==1126
all_scores=[];weight_hashes={}
for seed in old+new:
 p=ROOT/(f'runs/lora_mamba_full/full_seed{seed}.pt' if seed in old else f'analysis/reviewer_response_20261102/three_seed_full_received_20261106/three_seed_full_results/runs/seed{seed}/model.pt');weight_hashes[str(seed)]=sha(p);cp=torch.load(p,map_location='cpu',weights_only=True);assert cp['seed']==seed
 state=cp['model_state'];bs=[v for k,v in state.items() if '.lora_B' in k]
 if seed in old:assert len(bs)==144 and all(torch.count_nonzero(v).item()==0 for v in bs)
 else:assert not any('.lora_' in k for k in state)
 head={k:v for k,v in state.items() if '.lora_' not in k};expect={n for n,p in model.named_parameters() if p.requires_grad};assert set(head)==expect,(seed,len(head),len(expect));load=model.load_state_dict(head,strict=False);assert not load.unexpected_keys and set(load.missing_keys)==set(model.state_dict())-set(head)
 model.eval();v=[];start=time.time()
 with torch.inference_mode():
  for i in range(0,len(samples),16):
   batch=samples[i:i+16];seq=[r['sequence'] for r in batch];x=tok([' '.join(s) for s in seq],padding=True,add_special_tokens=True,return_tensors='pt');length=torch.tensor([len(s) for s in seq],device='cuda')
   with torch.autocast('cuda',dtype=torch.float16):z=model(x['input_ids'].cuda(),x['attention_mask'].cuda(),length)
   v.extend(torch.sigmoid(z.float()).cpu().tolist())
 assert len(v)==1126 and np.isfinite(v).all();all_scores.append(v);print('seed',seed,'seconds',round(time.time()-start,1),flush=True)
scores=np.array(all_scores);seqidx={r['sequence']:i for i,r in enumerate(samples)};TH=.47022256531284556;results={}
for split in ('test1','test2'):
 records=[r for r in refs if r['dataset']=='GHXTox' and r['split']==split];saved=read(ROOT/f'runs/default_no_chemical_20261106/{split}_predictions.csv');cache={r['sample_id']:r for r in saved};assert len(cache)==len(saved)==len(records) and set(cache)=={r['source_id'] for r in records}
 ix=np.array([seqidx[r['sequence']] for r in records]);labels=np.array([int(r['label']) for r in records]);assert all(int(cache[r['source_id']]['label'])==int(r['label']) for r in records)
 one3=cal[3].predict_proba(logit(scores[:3,ix].mean(axis=0))[:,None])[:,1];one6=cal[6].predict_proba(logit(scores[:,ix].mean(axis=0))[:,None])[:,1];p3d=np.array([float(cache[r['source_id']]['p_3d']) for r in records]);base=np.array([float(cache[r['source_id']]['toxicity_probability']) for r in records]);saved_one=np.array([float(cache[r['source_id']]['p_1d']) for r in records]);schemes={'historical_fusion_cached':base,'six_sequence':one6,'six_plus_3d':.72*one6+.28*p3d,'three_fusion_recomputed':.72*one3+.28*p3d};rng=np.random.default_rng(20261106+(split=='test2'));pos=np.flatnonzero(labels==1);neg=np.flatnonzero(labels==0);ind=np.concatenate([rng.choice(pos,(5000,len(pos))),rng.choice(neg,(5000,len(neg)))],axis=1)
 def metrics(p):
  pred=p>=TH;tp=int(((labels==1)&pred).sum());fp=int(((labels==0)&pred).sum());fn=int(((labels==1)&~pred).sum());tn=int(((labels==0)&~pred).sum());return {'MCC':float(matthews_corrcoef(labels,pred)),'AUROC':float(roc_auc_score(labels,p)),'AUPRC':float(average_precision_score(labels,p)),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'TP':tp,'FN':fn,'TN':tn,'FP':fp}
 def bootm(p):
  a=labels[ind];b=(p>=TH)[ind];tp=((a==1)&b).sum(axis=1);tn=((a==0)&~b).sum(axis=1);fp=((a==0)&b).sum(axis=1);fn=((a==1)&~b).sum(axis=1);den=np.sqrt((tp+fp).astype(float)*(tp+fn)*(tn+fp)*(tn+fn));return np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den),where=den!=0)
 baseboot=bootm(base);stats={k:metrics(p) for k,p in schemes.items()};diffs={}
 for k in ('six_sequence','six_plus_3d'):
  b=bootm(schemes[k])-baseboot;diffs[k]={'MCC_delta_vs_cached':stats[k]['MCC']-stats['historical_fusion_cached']['MCC'],'MCC_delta_ci95':np.quantile(b,[.025,.975]).tolist()}
 results[split]={'n':len(records),'metrics':stats,'paired':diffs,'old_1d_max_abs_difference':float(np.max(np.abs(one3-saved_one))),'cached_vs_recomputed_decision_changes':int(((schemes['three_fusion_recomputed']>=TH)!=(base>=TH)).sum())}
 with (OUT/f'{split}_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['sample_id','sequence','label']+[f'p1d_seed{s}' for s in old+new]+['calibrated_3seed','calibrated_6seed','p3d']+list(schemes));
  for j,r in enumerate(records):w.writerow([r['source_id'],r['sequence'],int(labels[j])]+scores[:,ix[j]].tolist()+[one3[j],one6[j],p3d[j]]+[v[j] for v in schemes.values()])
(OUT/'summary.json').write_text(json.dumps({'protocol':'PROTOCOL.md','calibration':{str(k):{'slope':float(v.coef_[0,0]),'intercept':float(v.intercept_[0])} for k,v in cal.items()},'checkpoint_sha256':weight_hashes,'results':results},indent=2),encoding='utf8');print(json.dumps(results,indent=2),flush=True)
