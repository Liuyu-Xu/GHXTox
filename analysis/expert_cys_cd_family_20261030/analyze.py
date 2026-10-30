import csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent
root=P.parents[1]
src=P.parent/'official_bcd_oof_20261030'/'analyze.py'
ns={'__file__':str(src)}
exec(compile(src.read_text(encoding='utf8').split('def metrics(mask):')[0],str(src),'exec'),ns)
meta,y,fold,cs,flags=(ns[k] for k in ('meta','y','fids','cs','flags'))
with (P.parent/'expert_disagreement_20261025'/'training_expert_atlas.csv').open(encoding='utf-8-sig',newline='') as f: atlas=list(csv.DictReader(f))
assert len(atlas)==len(meta) and all(a['sample_id']==m['sample_id'] and int(a['label'])==int(v) for a,m,v in zip(atlas,meta,y))

def cb(n): return '0-1' if n<=1 else '2-5' if n<=5 else '6+'
def lb(n): return '<=30' if n<=30 else '31-50' if n<=50 else '>50'
def dump(name,rows):
 with (P/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

groups=defaultdict(list)
for a in atlas:
 groups[(cb(int(a['cys_count'])),lb(int(a['length'])),int(a['label']))].append(a)
out=[]
for (c,l,label),items in sorted(groups.items()):
 counts=Counter(r['expert_group'] for r in items)
 out.append({'cys':c,'length':l,'label':label,'n':len(items),'both_correct':counts['both_correct'],'seq_wrong_struct_right':counts['structure_only_correct'],'seq_right_struct_wrong':counts['sequence_only_correct'],'both_wrong':counts['both_wrong']})
 assert sum(counts.values())==len(items) and sum(out[-1][k] for k in ('both_correct','seq_wrong_struct_right','seq_right_struct_wrong','both_wrong'))==len(items)
dump('expert_by_cys_length_label.csv',out)

th=ns['t']; C=ns['models']['C_three']>=th; D=ns['models']['D_three']>=th
changed=[]
for i in np.flatnonzero(C!=D):
 m=meta[i]; a=atlas[i]; side=m.get('current_family_sidecar') or {}; fam=side.get('parent_families') or []
 changed.append({'sample_id':m['sample_id'],'sequence':m['sequence'],'group_id':m['group_id'],'fold':int(fold[i]),'label':int(y[i]),'cys_count':int(cs[i]),'length':len(m['sequence']),'cys_bin':cb(cs[i]),'length_bin':lb(len(m['sequence'])),'outcome':'corrected' if D[i]==y[i] else 'harmed','C_probability':float(ns['models']['C_three'][i]),'D_probability':float(ns['models']['D_three'][i]),'flag':int(flags[i]),'family':' | '.join(fam),'family_status':side.get('match_status','unknown'),'name':m.get('name','')})
dump('C_D_changed_samples.csv',changed)

# Sequence-group counts are the existing CD-HIT 80% grouping, not proof of family independence.
seq=defaultdict(list); families=defaultdict(list)
for r in changed:
 seq[r['group_id']].append(r)
 for fam in r['family'].split(' | ') if r['family'] else ['UNKNOWN']: families[fam].append(r)
seqout=[]
for group,rows in sorted(seq.items(),key=lambda kv:-len(kv[1])):
 c=Counter(r['outcome'] for r in rows);seqout.append({'group_id':group,'n_changed':len(rows),'corrected':c['corrected'],'harmed':c['harmed'],'sample_ids':' '.join(r['sample_id'] for r in rows)})
dump('C_D_sequence_groups.csv',seqout)
famout=[]
for fam,rows in sorted(families.items(),key=lambda kv:-len(kv[1])):
 c=Counter(r['outcome'] for r in rows);famout.append({'family':fam,'n_changed':len(rows),'corrected':c['corrected'],'harmed':c['harmed'],'sequence_groups':len(set(r['group_id'] for r in rows))})
dump('C_D_family_counts.csv',famout)
print('expert categories',Counter(a['expert_group'] for a in atlas))
print('changed',len(changed),Counter(r['outcome'] for r in changed),'distinct groups',len(seq),'largest groups',seqout[:6])
print('family counts',famout[:12]); print('unknown',sum(not r['family'] for r in changed))
print('top combined groups',out)
