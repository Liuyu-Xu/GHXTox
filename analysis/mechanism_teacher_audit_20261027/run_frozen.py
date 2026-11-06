"""Fixed author classification checkpoints; no fit or GHXTox label access in inference."""
import os,sys,pathlib,json,time,hashlib,csv,gc
P=pathlib.Path(__file__).resolve().parent
R=P.parents[1]
os.environ['OMP_NUM_THREADS']='2'
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
import torch
from transformers import AutoTokenizer,EsmConfig,EsmForSequenceClassification
torch.set_num_threads(2)
samples=json.loads((R/'analysis/ghxtox_shortfall_registry_20261027/all_sample_shortfall_index.json').read_text(encoding='utf-8'))

def run(modelname,folder,threshold):
    tokenizer=AutoTokenizer.from_pretrained(str(folder),local_files_only=True)
    if (folder/'model.safetensors').exists():
        model,loading=EsmForSequenceClassification.from_pretrained(str(folder),local_files_only=True,output_loading_info=True)
        assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading['mismatched_keys'],loading
        weight=folder/'model.safetensors'
    else:
        weight=folder/'pytorch_model.bin'
        model=EsmForSequenceClassification(EsmConfig.from_pretrained(str(folder),local_files_only=True))
        state=torch.load(weight,map_location='cpu',weights_only=True)
        extras=set(state)-set(model.state_dict())
        allowed={'esm.embeddings.position_ids','esm.embeddings.position_embeddings.weight'}
        assert extras<=allowed and model.config.position_embedding_type=='rotary',extras
        # Old Transformers saved absolute-position tables even in rotary mode;
        # current rotary forward never reads them. No trainable used weight is dropped.
        for k in extras:state.pop(k)
        model.load_state_dict(state,strict=True);del state
        loading={'manual_safe_torch_load':'weights_only=True, strict=True after audited rotary-unused compatibility keys','ignored_rotary_unused_keys':sorted(extras)}
    model.eval().to('cuda')
    rows=[{'sample_id':s['sample_id'],'sequence':s['sequence'],'population':'GHXTox'} for s in samples]
    csvpath=P/modelname/'Dataset'/('peptides' if modelname=='ntxpred2' else '')/'independent_dataset.csv'
    for i,r in enumerate(csv.DictReader(csvpath.open(encoding='utf-8-sig'))):
        rows.append({'sample_id':f'author_independent_row_{i+2}','sequence':r.get('SEQUENCE',r.get('Sequence')),'population':'author_independent','author_endpoint_label':int(r.get('label',r.get('Label')))})
    start=time.time()
    order=sorted(range(len(rows)),key=lambda i:len(rows[i]['sequence']))
    with torch.inference_mode():
        for offset in range(0,len(order),64):
            idx=order[offset:offset+64]
            inputs=tokenizer([rows[i]['sequence'] for i in idx],padding=True,truncation=True,return_tensors='pt')
            assert inputs['input_ids'].shape[1]==max(len(rows[i]['sequence']) for i in idx)+2
            output=model(**{k:v.to('cuda') for k,v in inputs.items()})
            probs=output.logits.softmax(dim=1)[:,1].cpu().tolist()
            logits=output.logits.cpu().tolist()
            for i,p,l in zip(idx,probs,logits):
                rows[i].update(probability=p,prediction=int(p>=threshold),threshold=threshold,logits=l)
            if offset%1024==0:print(modelname,offset,'/',len(order),flush=True)
    # Check batching invariance against independent single forward passes, including longest.
    check=[]
    with torch.inference_mode():
        for i in [order[0],order[len(order)//2],order[-1]]:
            inputs=tokenizer(rows[i]['sequence'],padding=True,truncation=True,return_tensors='pt')
            prob=model(**{k:v.to('cuda') for k,v in inputs.items()}).logits.softmax(dim=1)[0,1].item()
            diff=abs(prob-rows[i]['probability']);assert diff<2e-5,(i,diff)
            check.append({'sample_id':rows[i]['sample_id'],'single_batch_probability_difference':diff})
    manifest={'model':modelname,'classification_variant':'ESM-only (-m 3), not MERCI hybrid' if modelname=='hemopi2' else 'peptide ESM2-t30 (-m 1), reproducible Model.zip NOT Model_whole.zip','folder':str(folder),'weight_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'threshold':threshold,'threshold_source':'author current code actual branch default; no selection on GHXTox','torch':torch.__version__,'cuda':torch.cuda.get_device_name(0),'parameter_count':sum(x.numel() for x in model.parameters()),'loading':loading,'seconds':time.time()-start,'rows':len(rows),'batch_checks':check,'tokenization':'same author padding/truncation, no max_length override, float32, eval/inference_mode, softmax[:,1]','train_calls':0}
    (P/(modelname+'_predictions.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (P/(modelname+'_inference.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(manifest,flush=True)
    del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    for name,folder,thr in [('hemopi2',P/'hemopi2/Model',0.58),('ntxpred2',P/'ntxpred2/Model/Peptide/saved_model_t30',0.45)]:
        if not (P/(name+'_predictions.json')).exists():run(name,folder,thr)
