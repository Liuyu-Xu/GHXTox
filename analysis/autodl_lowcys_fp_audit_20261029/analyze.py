from pathlib import Path
import json,collections,numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];read=lambda q:json.loads(q.read_text(encoding='utf8'))
meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');idx={x['sample_id']:x for x in meta};t=.47022256531284556;result={};all_rows=[]
for f in (1,2):
 root=R/f'analysis/autodl_cys_fold{f}_20261029/runs/fold{f}';arms={a:read(root/a/'predictions.json') for a in 'ABCD'};assert all([r['sample_id'] for r in arms[a]]==[r['sample_id'] for r in arms['A']] for a in arms)
 target=[]
 for j,r in enumerate(arms['A']):
  m=idx[r['sample_id']];assert r['label']==m['label']
  if m['cys_count']>1 or m['label']!=0:continue
  v={'sample_id':m['sample_id'],'group_id':m['group_id'],'fold':f,'label_flag':bool(m['uniform_flag']),'fragment':bool(m['fragment']),'conflict':bool(m['conflict']),'family':m['current_family_sidecar'].get('full_families',[]),'database':m.get('original_database'),'cys_count':m['cys_count'],'length':len(m['sequence'])}
  for a in arms:
   v[a+'_fusion']=arms[a][j]['fusion_probability'];v[a+'_seq']=arms[a][j]['sequence_probability'];v[a+'_wrong']=v[a+'_fusion']>=t
  v['role']='new_fp' if not v['A_wrong'] and v['B_wrong'] else 'corrected' if v['A_wrong'] and not v['B_wrong'] else 'both_wrong' if v['A_wrong'] else 'both_correct';target.append(v)
 def summary(rr):
  c=collections.Counter(f for r in rr for f in set(r['family']));seq=np.array([r['B_seq']-r['A_seq'] for r in rr]);diff=np.array([r['B_fusion']-r['A_fusion'] for r in rr]);post=np.array([r['B_fusion']-t for r in rr]);return {'n':len(rr),'flagged':sum(r['label_flag'] for r in rr),'fragment':sum(r['fragment'] for r in rr),'conflict':sum(r['conflict'] for r in rr),'family_annotated_n':sum(bool(r['family']) for r in rr),'top_families':c.most_common(8),'databases':dict(collections.Counter(str(r['database']) for r in rr)),'seq_delta_median':float(np.median(seq)) if len(rr) else None,'fusion_delta_median':float(np.median(diff)) if len(rr) else None,'new_fp_margin_bins':{'0_to_under_0.05':int(sum((post>=0)&(post<.05))),'0.05_to_under_0.15':int(sum((post>=.05)&(post<.15))),'at_least_0.15':int(sum(post>=.15))} if len(rr) else {}}
 result[str(f)]={role:summary([r for r in target if r['role']==role]) for role in ['new_fp','corrected','both_wrong','both_correct']};result[str(f)]['entire_low_negative']=summary(target);all_rows+=target
(P/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');(P/'samples.json').write_text(json.dumps(all_rows,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
