import csv, json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
src = P.parent / 'official_bcd_oof_20261030' / 'analyze.py'
ns = {'__file__': str(src)}
exec(compile(src.read_text(encoding='utf8').split('def metrics(mask):')[0], str(src), 'exec'), ns)
y, cs, fold, flags, meta, t = (ns[k] for k in ('y','cs','fids','flags','meta','t'))
old, new = (ns['models'][k] for k in ('historical_three','D_three'))
od, nd = old >= t, new >= t
corrected, harmed = (od != y) & (nd == y), (od == y) & (nd != y)

def record(mask, name):
    n = int(mask.sum()); c = int((mask & corrected).sum()); h = int((mask & harmed).sum())
    return {'group':name,'n':n,'old_errors':int((mask & (od != y)).sum()),'D_errors':int((mask & (nd != y)).sum()),'corrected':c,'harmed':h,'net_errors_reduced':c-h}

bands = [('0-1', cs <= 1), ('2-5', (cs >= 2)&(cs <= 5)), ('6+', cs >= 6)]
results = [record(np.ones(len(y), dtype=bool), 'all')]
for name, m in bands:
    for label in (1,0): results.append(record(m & (y == label), f'{name}_label{label}'))
for f in range(5): results.append(record(fold == f, f'fold{f}'))
for flag in (True, False): results.append(record(flags == flag, f'flag_{flag}'))
for label in (1,0):
    for flag in (True,False): results.append(record((flags==flag)&(y==label), f'label{label}_flag_{flag}'))
rows=[]
for i in np.flatnonzero(corrected | harmed):
    m=meta[i]; side=m.get('current_family_sidecar') or {}
    rows.append({'sample_id':m['sample_id'],'sequence':m['sequence'],'label':int(y[i]),'fold':int(fold[i]),'sequence_group':m['group_id'],'cys_count':int(cs[i]),'length':len(m['sequence']),'status':'corrected' if corrected[i] else 'harmed','historical_probability':float(old[i]),'D_probability':float(new[i]),'historical_prediction':int(od[i]),'D_prediction':int(nd[i]),'label_concern_flag':int(flags[i]),'fragment_flag':m.get('fragment',0),'conflict_flag':m.get('conflict',0),'family':'; '.join(side.get('parent_families',[])),'name':m.get('name','')})
with (P/'changed_samples.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
with (P/'group_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
(P/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
cc = ns['models']['C_three'] >= t
cd = []
for name, m in bands:
    for label in (1,0):
        mask=m & (y==label)
        cd.append({'group':f'{name}_label{label}','n':int(mask.sum()),'C_errors':int((mask & (cc!=y)).sum()),'D_errors':int((mask & (nd!=y)).sum()),'corrected':int((mask & (cc!=y) & (nd==y)).sum()),'harmed':int((mask & (cc==y) & (nd!=y)).sum())})
with (P/'C_to_D_groups.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cd[0].keys());w.writeheader();w.writerows(cd)
print(json.dumps(results,ensure_ascii=False,indent=2))
