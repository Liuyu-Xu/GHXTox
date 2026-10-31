"""Exhaustive global alignment search, with proven composition/length upper bounds.

Identity = identical residue pairs / full alignment columns (including terminal gaps).
Needleman-Wunsch, match +2, mismatch -1, gap-open 3, extension 1, all end gaps penalized.
For each sequence pair one score-optimal alignment is used (parasail stats tie rule).
Maximum is over every reference pair under these settings, NOT CD-HIT's identity definition.
Use full traceback columns: parasail stats.length omits initial gap columns in some cases.
"""
from pathlib import Path
import sys,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'_vendor'))
import parasail
ALPHABET='ACDEFGHIKLMNPQRSTVWY'

def init_worker(refs):
    global REFS,SEQS,COUNTS,LENS,MATRIX,LOOKUP
    REFS=refs;SEQS=[x['sequence'] for x in refs];LENS=np.array(list(map(len,SEQS)))
    COUNTS=np.array([[s.count(a) for a in ALPHABET] for s in SEQS],dtype=np.int16)
    MATRIX=parasail.matrix_create(ALPHABET,2,-1);LOOKUP={s:i for i,s in enumerate(SEQS)}

def search_batch(items):
    results=[]
    for s in items:
        best=-1.;bi=-1;matches=cols=score=0;calls=0
        if s in LOOKUP:
            bi=LOOKUP[s];best=1.;matches=cols=len(s);score=2*len(s)
        else:
            q=np.array([s.count(a) for a in ALPHABET],dtype=np.int16)
            # No alignment can match more residues than its composition intersection,
            # and every global alignment has >=max(sequence lengths) columns.
            upper=np.minimum(COUNTS,q).sum(axis=1)/np.maximum(LENS,len(s))
            for i in np.argsort(-upper,kind='stable'):
                if upper[i]<=best:break
                r=parasail.nw_trace_scan_16(s,SEQS[i],3,1,MATRIX)
                assert not r.saturated
                trace=r.traceback
                fullcols=len(trace.query)
                fullmatches=sum(a==b and a!='-' for a,b in zip(trace.query,trace.ref))
                assert fullcols>=max(len(s),len(SEQS[i]))
                identity=fullmatches/fullcols;calls+=1
                if identity>best:
                    best=identity;bi=int(i);matches=fullmatches;cols=fullcols;score=r.score
        ref=REFS[bi]
        results.append(dict(sequence=s,max_identity=best,nearest_ref_id=ref['source_id'],nearest_ref_split=ref['split'],
            nearest_ref_sequence=ref['sequence'],nearest_ref_label=int(ref['label']),matches=int(matches),
            alignment_columns=int(cols),alignment_score=int(score),aligned_pairs=calls))
    return results

def run():
    d=pd.read_csv(OUT/'all_records.csv',low_memory=False)
    # Include both historical tests, plus all eligible new candidates; each unique sequence once.
    c=pd.read_csv(OUT/'external_record_candidates.csv',low_memory=False)
    queries=sorted(set(c.sequence)|set(d[(d.dataset=='GHXTox')&(d.split!='train')].sequence))
    t=time.time()
    for refname,mask in [('train',(d.dataset=='GHXTox')&(d.split=='train')),('known',(d.dataset=='GHXTox'))]:
        refs=d[mask][['sequence','source_id','split','label']].drop_duplicates('sequence').to_dict('records')
        path=OUT/f'alignment_to_{refname}_v2.csv'
        done=pd.read_csv(path).to_dict('records') if path.exists() else []
        seen={x['sequence'] for x in done};todo=[s for s in queries if s not in seen]
        print(refname,'references',len(refs),'queries_remaining',len(todo),flush=True)
        batches=[todo[i:i+100] for i in range(0,len(todo),100)]
        with ProcessPoolExecutor(max_workers=3,initializer=init_worker,initargs=(refs,)) as pool:
            futures=[pool.submit(search_batch,b) for b in batches]
            for j,f in enumerate(as_completed(futures)):
                done.extend(f.result())
                if (j+1)%5==0 or j+1==len(futures):
                    pd.DataFrame(done).sort_values('sequence').to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')
                    print(refname,'completed',len(done),'elapsed_s',round(time.time()-t),flush=True)
    (OUT/'alignment_protocol.json').write_text(json.dumps(dict(engine='parasail '+parasail.__version__,algorithm='Needleman-Wunsch global affine gap',
        match=2,mismatch=-1,gap_open=3,gap_extend=1,end_gaps='penalized',identity='matches/full_alignment_columns',
        reference_sets=['GHXTox/train','GHXTox/train+test1+test2 unique'],
        exhaustive=True,pruning='composition-overlap/max_length proven upper bound',
        tie_rule='parasail traceback selected score-optimal alignment; first maximum reference in stable upper-bound ordering',
        caveat='Not CD-HIT identity; family and assay/study separation still required',queries=len(queries),seconds=time.time()-t),indent=2))

if __name__=='__main__':run()
