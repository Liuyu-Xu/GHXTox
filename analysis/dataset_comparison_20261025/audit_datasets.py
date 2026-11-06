"""Reproducible sequence/label audit. Source labels are preserved, never corrected.

Run: D:/anaconda/envs/pytorch_env/python.exe analysis/dataset_comparison_20261025/audit_datasets.py
"""
from pathlib import Path
from collections import Counter
from itertools import combinations
import csv, hashlib, json, re
import pandas as pd

OUT=Path(__file__).resolve().parent
SRC=OUT/'sources'
ROOT=Path('D:/dachuang2026/GHXTox')
OLD=OUT.parent/'ghxtox_error_atlas_20261025/evidence_phase2'
AA=set('ACDEFGHIKLMNPQRSTVWY')

def save(name,df):
    df.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.17g')

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def fasta(path):
    name=None;seq=[]
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line=line.strip()
        if not line:continue
        if line.startswith('>'):
            if name is not None:yield name,''.join(seq)
            name=line[1:];seq=[]
        else:
            assert name is not None,(path,line)
            seq.append(line)
    if name is not None:yield name,''.join(seq)

def build():
    rows=[];inputs=[]
    def add(dataset,split,source_id,seq,label,path,**extra):
        raw=str(seq);seq=re.sub(r'\s+','',raw).upper()
        assert int(label) in (0,1)
        rows.append(dict(dataset=dataset,split=split,record_id=f'{dataset}:{split}:{len(rows)+1}',
            source_id=str(source_id),sequence=seq,raw_sequence=raw,label=int(label),length=len(seq),
            standard20=bool(seq) and set(seq)<=AA,cys_count=seq.count('C'),
            source_file=str(path),endpoint='broad_toxicity_annotation',
            assay_species='unknown',assay_concentration='unknown',assay_time='unknown',
            molecular_modification='not_retained_in_file',label_evidence='binary_label_only',**extra))
    def track(path):
        if str(path) not in {x['path'] for x in inputs}:
            inputs.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    # Local deployed-data identities, without inspecting new prediction outcomes.
    path=ROOT/'runs/unified_inference_oof_v1/contract.csv';track(path)
    df=pd.read_csv(path)
    for r in df.to_dict('records'):add('GHXTox','train',r['sample_id'],r['sequence'],r['label'],path)
    for split in ['test1','test2']:
        path=ROOT/f'runs/single_structure_test_components/{split}_structure_predictions.csv';track(path)
        for r in pd.read_csv(path).to_dict('records'):add('GHXTox',split,r['sample_id'],r['sequence'],r['label'],path)
    fs=[('CAPTP','jiaoshihu__CAPTP','train','data/train_data.txt'),('CAPTP','jiaoshihu__CAPTP','test1','data/test1.txt'),('CAPTP','jiaoshihu__CAPTP','test2','data/test2.txt'),
        ('ToxMSRC','Renjingyi123__ToxMSRC','train','Raw data/train_data.fasta'),('ToxMSRC','Renjingyi123__ToxMSRC','test1','Raw data/test1.fasta'),('ToxMSRC','Renjingyi123__ToxMSRC','test2','Raw data/test2.fasta'),
        ('StrucToxNet','jiaoshihu__StrucToxNet','train','data/train_data.txt'),('StrucToxNet','jiaoshihu__StrucToxNet','test','data/test_data.txt')]
    for cutoff in ['0.8','0.9']:
        for split in ['train','test']:fs.append((f'ToxiPep_{cutoff}','GGCL7__ToxiPep',split,f'Dataset/{cutoff}/{split}.txt'))
    fs.append(('ToxiPep','GGCL7__ToxiPep','independent','Dataset/Independent set/independent test set.txt'))
    for split in ['train','test1','test2']:
        dr={'train':'train','test1':'Indtest1','test2':'Indtest2'}[split]
        for lab in ['pos','neg']:fs.append(('ToxPLTC','yunyunliang88__ToxPLTC',split,f'{dr}/{lab}_fasta.txt'))
    for split in ['train','test']:
        for lab in ['pos','neg']:fs.append(('HyPepTox-Fuse','cbbl-skku-org__HyPepTox-Fuse',split,f'raw_dataset/{split}_{lab}.fa'))
    for dataset,repo,split,file in fs:
        path=SRC/repo/file;track(path)
        for i,(name,seq) in enumerate(fasta(path)):
            if '|0' in name or '|1' in name:label=int(name.split('|')[-1])
            elif re.match(r'pos',name,re.I):label=1
            elif re.match(r'neg',name,re.I):label=0
            elif 'pos_' in file:label=1
            elif 'neg_' in file:label=0
            else:raise ValueError((path,name))
            add(dataset,split,name,seq,label,path)
    for split in ['train','test','independent_test']:
        path=SRC/'Marscolono__ToxMamba/data/Toxi_data'/f'{split}.csv';track(path)
        for r in pd.read_csv(path,sep='\t').to_dict('records'):add('ToxMamba',split,r['name'],r['seq'],r['label'],path)
    for split in ['train','test']:
        for lab in ['pos','neg']:
            path=SRC/'raghavagps__toxinpred3/dataset'/f'{split}_{lab}.csv';track(path)
            for i,seq in enumerate(path.read_text(encoding='utf-8-sig').splitlines()):
                if seq.strip():add('ToxinPred3',split,f'{lab}_{i+1}',seq,int(lab=='pos'),path)
    path=SRC/'WYLqlit__PeptiTox/Data/merged_data.csv';track(path)
    for i,r in enumerate(pd.read_csv(path).to_dict('records')):add('PeptiTox','unspecified',str(i+1),r['sequence'],r['label'],path)
    for split,file in [('cross_val','cross_val_dataset.csv'),('independent','independent_dataset.csv')]:
        path=SRC/'raghavagps__HemoPI2/Dataset'/file;track(path)
        for i,r in enumerate(pd.read_csv(path).to_dict('records')):
            add('HemoPI2',split,str(i+1),r['SEQUENCE'],r['label'],path,hc50_uM=r['μM'])
    path=SRC/'cihebiyql__ToxiCompass/data/benchmark/binary_manifest.csv';track(path)
    for r in pd.read_csv(path,keep_default_na=False).to_dict('records'):
        add('ToxiCompass',r['split'],r['sample_id'],r['sequence'],r['label'],path,
            source_set=r['source_set'],target_labels=r['target_labels_experiment'],source_cluster=r['mmseqs90_cluster_id'])
    df=pd.DataFrame(rows)
    df.loc[df.dataset=='HemoPI2','endpoint']='mammalian_RBC_HC50'
    df.loc[df.dataset=='HemoPI2','label_evidence']='reported_HC50; assay_details_not_retained'
    df.loc[df.dataset=='ToxiCompass','endpoint']='toxin_like_vs_operational_comparator'
    df.loc[(df.dataset=='ToxiCompass')&(df.label==0),'label_evidence']='reviewed_non_toxin_non_AMP_operational_comparator'
    # Attach earlier exact identity evidence; it flags boundary concerns, never validates source labels.
    prior=pd.read_csv(OLD/'training_evidence_flags.csv',keep_default_na=False)
    evidence=prior[['sequence','any_fragment_match','identity_status','exact_match_accessions']].copy()
    df=df.merge(evidence,on='sequence',how='left',validate='many_to_one')
    df['prior_fragment_review_flag']=df.any_fragment_match.eq(True)
    df['prior_evidence_available']=df.identity_status.notna()
    save('all_records.csv',df);dump('input_manifest.json',inputs)
    summary=[];sets={};maps={}
    for (dataset,split),g in df.groupby(['dataset','split'],sort=False):
        key=dataset+'/'+split;s=set(g.sequence);sets[key]=s;maps[key]=g.groupby('sequence').label.agg(lambda x:set(x)).to_dict()
        summary.append(dict(dataset=dataset,split=split,n_records=len(g),n_unique=len(s),positive=int(g.label.sum()),negative=int((g.label==0).sum()),
            duplicate_records=len(g)-len(s),within_split_conflicting_sequences=int((g.groupby('sequence').label.nunique()>1).sum()),
            min_length=int(g.length.min()),max_length=int(g.length.max()),nonstandard20=int((~g.standard20).sum()),
            prior_fragment_review_records=int(g.prior_fragment_review_flag.sum())))
    save('dataset_inventory.csv',pd.DataFrame(summary))
    overlap=[]
    for a,b in combinations(sets,2):
        shared=sets[a]&sets[b]
        overlap.append(dict(dataset_split_a=a,dataset_split_b=b,unique_a=len(sets[a]),unique_b=len(sets[b]),exact_shared=len(shared),
            percent_a=100*len(shared)/len(sets[a]),percent_b=100*len(shared)/len(sets[b]),
            sequence_label_disagreements=sum(maps[a][s]!=maps[b][s] for s in shared),same_sequence_set=sets[a]==sets[b]))
    save('pairwise_exact_overlap.csv',pd.DataFrame(overlap))
    conflicts=[]
    # RBC and broad toxin labels are different endpoints, deliberately not pooled into "label errors".
    for scope,mask in [('broad_annotation',df.endpoint.ne('mammalian_RBC_HC50')),('RBC_HC50',df.endpoint.eq('mammalian_RBC_HC50'))]:
        for seq,g in df[mask].groupby('sequence'):
            if g.label.nunique()<2:continue
            conflicts.append(dict(scope=scope,sequence=seq,length=len(seq),datasets=';'.join(sorted(set(g.dataset))),
                positive_sources=';'.join(sorted(set(g.loc[g.label==1,'dataset']+'/'+g.loc[g.label==1,'split']))),
                negative_sources=';'.join(sorted(set(g.loc[g.label==0,'dataset']+'/'+g.loc[g.label==0,'split']))),
                ghxtox_label=';'.join(map(str,sorted(set(g.loc[g.dataset=='GHXTox','label'])))),
                interpretation='label_disagreement_requires_entity_endpoint_assay_review; not_confirmed_mislabel'))
    save('same_sequence_label_review.csv',pd.DataFrame(conflicts))
    train=sets['GHXTox/train'];tests=sets['GHXTox/test1']|sets['GHXTox/test2'];known=train|tests
    cohort=[]
    for (dataset,split),g in df.groupby(['dataset','split'],sort=False):
        s=set(g.sequence)
        cohort.append(dict(dataset=dataset,split=split,n_unique=len(s),overlap_ghxtox_train=len(s&train),overlap_ghxtox_historical_tests=len(s&tests),
            unseen_exact=len(s-known),unseen_standard20_len11_50=len(set(g.loc[g.standard20&g.length.between(11,50),'sequence'])-known)))
    save('external_screen_counts.csv',pd.DataFrame(cohort))
    # Split sources kept distinct; these are candidates for evidence review, not a final validated test set.
    external_names=['ToxiPep','ToxiPep_0.8','ToxiPep_0.9','ToxMamba','ToxinPred3','HyPepTox-Fuse','PeptiTox','HemoPI2','ToxiCompass']
    cand=df[df.dataset.isin(external_names)&~df.sequence.isin(known)&df.standard20&df.length.between(11,50)].copy()
    cand['status']='pending_alignment_and_original_evidence_review'
    save('external_record_candidates.csv',cand)
    gh= df[df.dataset=='GHXTox'].groupby('split')
    checks={}
    for method,split2 in [('CAPTP','test1'),('ToxMSRC','test1'),('StrucToxNet','test'),('ToxPLTC','test1')]:
        for split in ['train','test1']+(['test2'] if method in ['CAPTP','ToxMSRC','ToxPLTC'] else []):
            target=split2 if split=='test1' else split
            a=df[(df.dataset=='GHXTox')&(df.split==split)][['sequence','label']]
            b=df[(df.dataset==method)&(df.split==target)][['sequence','label']]
            checks[f'GHXTox/{split}=={method}/{target}']={'same_multiset':Counter(map(tuple,a.values))==Counter(map(tuple,b.values)),
                'same_order':a.values.tolist()==b.values.tolist()}
    # Separate ToxMSRC label CSVs are independently checked, not just trusted FASTA names.
    for split,file in [('train','y_train.csv'),('test1','y_test1.csv'),('test2','y_test2.csv')]:
        path=SRC/'Renjingyi123__ToxMSRC/Raw data'/file;track(path)
        lab=pd.read_csv(path).iloc[:,0].astype(int).tolist()
        checks[f'ToxMSRC/{split}_csv_vs_fasta_labels']={'exact_agreement':lab==df[(df.dataset=='ToxMSRC')&(df.split==split)].label.tolist()}
    checks['GHXTox_test2_subset_test1']=sets['GHXTox/test2']<=sets['GHXTox/test1']
    dump('verification.json',checks);dump('input_manifest.json',inputs)
    dump('summary.json',dict(total_records=len(df),unique_sequences=df.sequence.nunique(),dataset_splits=len(summary),
        methods=df.dataset.nunique(),broad_label_disagreements=sum(x['scope']=='broad_annotation' for x in conflicts),
        RBC_disagreements=sum(x['scope']=='RBC_HC50' for x in conflicts),
        external_candidate_records=len(cand),external_candidate_unique=cand.sequence.nunique()))
    print(json.dumps(json.loads((OUT/'summary.json').read_text()),ensure_ascii=False),flush=True)

if __name__=='__main__':build()
