"""Re-use author's motif scanner + six postprocessors without executing CLI side effects."""
import pathlib,json,ast,subprocess,hashlib
import numpy as np
import pandas as pd
P=pathlib.Path(__file__).resolve().parent
M=P/'hemopi2';W=M/'motif_diagnostic';W.mkdir(exist_ok=True)
rows=json.loads((P/'hemopi2_predictions.json').read_text(encoding='utf-8'))
fasta=W/'input.fasta'
fasta.write_text(''.join('>'+r['sample_id']+'\n'+r['sequence']+'\n' for r in rows),encoding='utf-8')
manifest=[]
for pol in ['p','n']:
    for n in [1,2]:
        motif=M/'motif'/f'{"pos" if pol=="p" else "neg"}_motif_{n}.txt'
        cmd=['C:/Strawberry/perl/bin/perl.exe',str(M/'merci/MERCI_motif_locator.pl'),'-p',str(fasta),'-i',str(motif),'-o',str(W/f'merci_{pol}_{n}.txt')]
        if n==2:cmd+=['-c','KOOLMAN-ROHM']
        if not (W/f'merci_{pol}_{n}.txt').exists():
            run=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
            (W/f'scanner_{pol}_{n}.log').write_text(run.stdout+run.stderr,encoding='utf-8')
            assert run.returncode==0,(cmd,run.stderr)
        manifest.append({'command':cmd,'returncode':0,'motif_sha256':hashlib.sha256(motif.read_bytes()).hexdigest()})
tree=ast.parse((P/'hemopi2_code.txt').read_text(encoding='utf-8'))
names={'MERCI_Processor_p','MERCI_Processor_n','Merci_after_processing_p','Merci_after_processing_n','hybrid'}
defs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
assert len(defs)==5
exec(compile(ast.Module(body=defs,type_ignores=[]),'author_functions','exec'),globals())
seqids=['>'+r['sample_id'] for r in rows]
for pol in ['p','n']:
    for n in [1,2]:
        globals()['MERCI_Processor_'+pol](str(W),f'merci_{pol}_{n}.txt',f'merci_output_{pol}_{n}.csv',seqids)
        globals()['Merci_after_processing_'+pol](str(W),f'merci_output_{pol}_{n}.csv',f'merci_hybrid_{pol}_{n}.csv')
# Public writer rounds ESM score to four decimals before hybrid reads it.
df=pd.DataFrame({'SeqID':[r['sample_id'] for r in rows],'Seq':[r['sequence'] for r in rows],'ML Score':[float(f"{r['probability']:.4f}") for r in rows]})
hybrid(str(W),df,'merci_hybrid_p_1.csv','merci_hybrid_n_1.csv','merci_hybrid_p_2.csv','merci_hybrid_n_2.csv',0.58,'hybrid_output.csv')
byid={r['SeqID']:r for r in pd.read_csv(W/'hybrid_output.csv').to_dict('records')}
out=[]
for r in rows:
    z=byid[r['sample_id']]
    assert z['Seq']==r['sequence']
    out.append({**r,'esm_probability':r['probability'],'probability':max(0.0,min(1.0,z['Hybrid Score'])),'raw_hybrid_score':z['Hybrid Score'],'merci_score':z['MERCI Score'],'prediction':int(z['Prediction']=='Hemolytic'),'threshold':0.58})
(P/'hemopi2_hybrid_predictions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
(P/'hemopi2_hybrid_inference.json').write_text(json.dumps({'variant':'author default -m 4, ESM2-t6 + MERCI','official_functions_unmodified':sorted(names),'threshold':0.58,'comparison':'> after public hybrid round(3), ESM file round(4); clipped score is not calibrated total toxicity probability','scanners':manifest,'rows':len(rows),'fit_calls':0},indent=2),encoding='utf-8')
print('hybrid completed',len(out),flush=True)
