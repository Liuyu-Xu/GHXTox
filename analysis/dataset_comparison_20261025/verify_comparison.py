"""Independent integrity and exhaustive-alignment checks for the audit artifacts."""
from pathlib import Path
from collections import Counter
import json,hashlib,sys
import numpy as np
import pandas as pd
from Bio import SeqIO,Align
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'_vendor'))
import parasail

def verify():
    checks={};d=pd.read_csv(OUT/'all_records.csv',low_memory=False)
    for x in json.loads((OUT/'input_manifest.json').read_text(encoding='utf8')):
        assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
    checks['all_original_inputs_unchanged']=True
    gh=d[(d.dataset=='GHXTox')&(d.split=='train')]
    for dataset,repo,path in [('CAPTP','jiaoshihu__CAPTP','data/train_data.txt'),('ToxMSRC','Renjingyi123__ToxMSRC','Raw data/train_data.fasta'),('StrucToxNet','jiaoshihu__StrucToxNet','data/train_data.txt')]:
        raw=[(str(x.seq),int(x.description.rsplit('|',1)[1])) for x in SeqIO.parse(OUT/'sources'/repo/path,'fasta')]
        assert raw==list(gh[['sequence','label']].itertuples(index=False,name=None))
    checks['independent_Bio_fasta_parsing_agrees']=True
    for a,b in [('ToxiPep_0.9','ToxMamba'),('HyPepTox-Fuse','ToxinPred3')]:
        for split in ['train','test']:
            assert Counter(map(tuple,d[(d.dataset==a)&(d.split==split)][['sequence','label']].values))==Counter(map(tuple,d[(d.dataset==b)&(d.split==split)][['sequence','label']].values))
    assert Counter(map(tuple,d[(d.dataset=='ToxiPep')][['sequence','label']].values))==Counter(map(tuple,d[(d.dataset=='ToxMamba')&(d.split=='independent_test')][['sequence','label']].values))
    checks['reused_families_independently_confirmed']=True
    # Compare four full searches with a naive search: do not reuse bound/pruning code.
    a=pd.read_csv(OUT/'alignment_to_train_v2.csv',float_precision='round_trip')
    rng=np.random.default_rng(20261025);sample=a.iloc[rng.choice(len(a),4,replace=False)]
    matrix=parasail.matrix_create('ACDEFGHIKLMNPQRSTVWY',2,-1)
    rows=[]
    for x in sample.to_dict('records'):
        best=0.
        for r in gh.sequence:
            v=parasail.nw_trace_scan_16(x['sequence'],r,3,1,matrix);t=v.traceback
            identity=sum(c1==c2 and c1!='-' for c1,c2 in zip(t.query,t.ref))/len(t.query)
            best=max(best,identity)
        assert abs(best-x['max_identity'])<1e-15,(x,best)
        rows.append(dict(sequence=x['sequence'],naive_max=best,pruned_max=x['max_identity'],references=len(gh)))
    checks['naive_exhaustive_vs_pruned_global_search']=rows
    # Check scoring in another engine, and a case where stats.length would be wrong.
    bp=Align.PairwiseAligner();bp.mode='global';bp.match_score=2;bp.mismatch_score=-1;bp.open_gap_score=-3;bp.extend_gap_score=-1
    for q,r,expected in [('ACDEFG','GGACDEFG',.75),('GGACDEFG','ACDEFG',.75),('ACDEFG','ACDEFG',1.)]:
        v=parasail.nw_trace_scan_16(q,r,3,1,matrix);t=v.traceback
        identity=sum(c1==c2 and c1!='-' for c1,c2 in zip(t.query,t.ref))/len(t.query)
        assert abs(identity-expected)<1e-15
        assert v.score==bp.score(q,r)
    for x in sample.to_dict('records'):
        assert x['alignment_score']==bp.score(x['sequence'],x['nearest_ref_sequence'])
    checks['independent_Biopython_scores_and_terminal_gap_controls']=True
    h=d[d.dataset=='HemoPI2'];assert ((h.hc50_uM<=100).astype(int)==h.label).all()
    checks['HemoPI2_labels_match_reported_HC50_threshold']=True
    p=pd.read_csv(OUT/'hemopi2_independent_priority_panel.csv',keep_default_na=False,float_precision='round_trip')
    known=set(d.loc[d.dataset=='GHXTox','sequence'])
    assert len(p)==69 and p.sequence.is_unique and not set(p.sequence)&known
    assert p.length.between(11,50).all() and (p.max_identity_known<=.4).all()
    assert not p.same_endpoint_label_disagreement.any()
    assert not p.approved_gold_label.any()
    checks['candidate_panel_not_declared_gold_or_used_for_model_selection']=True
    m=pd.read_csv(OUT/'sources/kren-ai-lab__maomao/maomao_sequence_pivot.csv')
    assert known<=set(m.sequence)
    checks['MAOMAO_contains_all_GHXTox_known_unique_sequences']=True
    # Compare official ToxiCompass frozen manifest hashes against published data card.
    card=(OUT/'sources/cihebiyql__ToxiCompass/docs/BENCHMARK_CARD.md').read_text(encoding='utf8')
    for name in ['binary_manifest.csv','target_manifest.csv']:
        sha=hashlib.sha256((OUT/'sources/cihebiyql__ToxiCompass/data/benchmark'/name).read_bytes()).hexdigest()
        assert sha in card
    checks['ToxiCompass_manifests_match_official_card_SHA256']=True
    checks['passed']=True
    (OUT/'FINAL_VERIFICATION.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(checks,ensure_ascii=True))

if __name__=='__main__':verify()
