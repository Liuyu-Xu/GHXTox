import argparse,json
from pathlib import Path
from sklearn.metrics import matthews_corrcoef,average_precision_score
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,default=1);a=ap.parse_args();root=P/'runs'/f'fold{a.fold}'
rows={};results={}
for arm in 'ABCD':
 rows[arm]=json.loads((root/arm/'predictions.json').read_text(encoding='utf8'));results[arm]=json.loads((root/arm/'result.json').read_text(encoding='utf8'))
assert len({r['initial_head_sha256'] for r in results.values()})==1
assert all([(v['sample_id'],v['label']) for v in rows[x]]==[(v['sample_id'],v['label']) for v in rows['A']] for x in 'BCD')
y=[x['label'] for x in rows['A']];out={}
for mode,threshold in [('sequence',.5),('fusion',.47022256531284556)]:
 out[mode]={};base=[x[mode+'_probability']>=threshold for x in rows['A']]
 for arm in 'ABCD':
  pp=[x[mode+'_probability'] for x in rows[arm]];d=[z>=threshold for z in pp]
  out[mode][arm]={'MCC':matthews_corrcoef(y,d),'AUPRC':average_precision_score(y,pp),'errors':sum(x!=z for x,z in zip(y,d)),'corrected_vs_A':sum(b!=z and v==z for b,v,z in zip(base,d,y)),'harmed_vs_A':sum(b==z and v!=z for b,v,z in zip(base,d,y)),'group_metrics':results[arm]['metrics'][mode]}
(root/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False,indent=2))
