import json,csv,hashlib
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
paths={'baseline':R/'analysis/lora_multifold_screen_20261028/fold3/predictions_no_lora.json','weight':R/'analysis/family_weight_no_lora_20261028/predictions_B.json','contrast':R/'analysis/supcon_pilot_20261028/predictions_B.json','contrast_unflagged':R/'analysis/supcon_pilot_20261028/predictions_C.json','oof':R/'analysis/lowcys_weight_multifold_20261028/all_oof_predictions.json','meta':R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json'}
data={k:read(v) for k,v in paths.items()};meta={m['sample_id']:m for m in data['meta']};base=data['baseline'];ids=[x['sample_id'] for x in base]
assert len(ids)==len(set(ids))==1291 and all([r['sample_id'] for r in data[k]]==ids for k in ['weight','contrast','contrast_unflagged'])
oof={x['sample_id']:x for x in data['oof']};assert len(oof)==6387
rows=[]
for j,x in enumerate(base):
 m=meta[x['sample_id']];y=int(x['label']);oo=oof[x['sample_id']];assert y==m['label']==oo['label'] and x['group_id']==m['group_id']==oo['group_id'] and oo['fold']==3
 row={'sample_id':x['sample_id'],'group_id':x['group_id'],'label':y,'cys_count':m['cys_count'],'uniform_flag':int(bool(m['uniform_flag'])),'family':'; '.join(m['current_family_sidecar']['parent_families']),'category':('low_positive' if y==1 and m['cys_count']<4 else 'high_negative' if y==0 and m['cys_count']>=4 else 'other')}
 for mode,t in [('sequence',.5),('fusion',.47022256531284556)]:
  for k in ['baseline','weight','contrast','contrast_unflagged']:
   q=data[k][j];assert q['sample_id']==x['sample_id'] and q['label']==y and q['group_id']==x['group_id'];p=float(q[mode+'_probability']);row[mode+'_'+k+'_probability']=p;row[mode+'_'+k+'_correct']=int((p>=t)==y)
  assert abs(row[mode+'_baseline_probability']-oo[mode+'_A'])<1e-12 and abs(row[mode+'_weight_probability']-oo[mode+'_B'])<1e-12
 rows.append(row)
def analyze(rr,mode):
 out={'n':len(rr),'groups':len({x['group_id'] for x in rr})}
 for k in ['baseline','weight','contrast','contrast_unflagged']:
  out[k]={'errors':sum(not x[mode+'_'+k+'_correct'] for x in rr),'corrected':sum(not x[mode+'_baseline_correct'] and x[mode+'_'+k+'_correct'] for x in rr),'harmed':sum(x[mode+'_baseline_correct'] and not x[mode+'_'+k+'_correct'] for x in rr)}
 out['patterns']=dict(Counter(''.join(str(x[mode+'_'+k+'_correct']) for k in ['baseline','weight','contrast']) for x in rr))
 out['both_methods_correct_baseline_wrong']=sum(not x[mode+'_baseline_correct'] and x[mode+'_weight_correct'] and x[mode+'_contrast_correct'] for x in rr)
 out['only_weight_correct_baseline_wrong']=sum(not x[mode+'_baseline_correct'] and x[mode+'_weight_correct'] and not x[mode+'_contrast_correct'] for x in rr)
 out['only_contrast_correct_baseline_wrong']=sum(not x[mode+'_baseline_correct'] and not x[mode+'_weight_correct'] and x[mode+'_contrast_correct'] for x in rr)
 out['both_wrong_baseline_correct']=sum(x[mode+'_baseline_correct'] and not x[mode+'_weight_correct'] and not x[mode+'_contrast_correct'] for x in rr)
 return out
panels={'all':lambda x:True,'low_positive':lambda x:x['category']=='low_positive','high_negative':lambda x:x['category']=='high_negative','other':lambda x:x['category']=='other','flagged':lambda x:x['uniform_flag']==1,'unflagged_not_gold':lambda x:x['uniform_flag']==0,'low_positive_flagged':lambda x:x['category']=='low_positive' and x['uniform_flag']==1,'low_positive_unflagged_not_gold':lambda x:x['category']=='low_positive' and x['uniform_flag']==0,'high_negative_flagged':lambda x:x['category']=='high_negative' and x['uniform_flag']==1,'high_negative_unflagged_not_gold':lambda x:x['category']=='high_negative' and x['uniform_flag']==0}
fold3={name:{mode:analyze([x for x in rows if fn(x)],mode) for mode in ['sequence','fusion']} for name,fn in panels.items()}
other={}
for f in [0,1,2,4]:
 z=[x for x in data['oof'] if x['fold']==f];other[str(f)]={}
 for category,fn in [('all',lambda x:True),('low_positive',lambda x:x['label']==1 and x['cys_count']<4),('high_negative',lambda x:x['label']==0 and x['cys_count']>=4),('flagged',lambda x:x['uniform_flag']),('unflagged_not_gold',lambda x:not x['uniform_flag'])]:
  panel=[x for x in z if fn(x)];other[str(f)][category]={}
  for mode,t in [('sequence',.5),('fusion',.47022256531284556)]:
   bc=[int((x[mode+'_A']>=t)==x['label']) for x in panel];wc=[int((x[mode+'_B']>=t)==x['label']) for x in panel]
   other[str(f)][category][mode]={'n':len(panel),'baseline_errors':len(panel)-sum(bc),'weight_errors':len(panel)-sum(wc),'corrected':sum(not a and b for a,b in zip(bc,wc)),'harmed':sum(a and not b for a,b in zip(bc,wc))}
assert all(other[str(f)]['all']['fusion']['n']>0 for f in [0,1,2,4]);save('fold3_panels.json',fold3);save('other_folds_weight_panels.json',other);save('manifest.json',{'inputs':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in paths.items()},'n_fold3':len(rows),'fold3_thresholds':{'sequence':.5,'fusion':.47022256531284556},'contrast_only_fold3':True,'no_training':True,'no_Test1_Test2':True})
with (P/'fold3_samples.csv').open('w',encoding='utf-8-sig',newline='') as h:
 w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('fold3 fusion',json.dumps({k:fold3[k]['fusion'] for k in ['all','low_positive','high_negative','other']},ensure_ascii=False));print('other folds fusion',json.dumps({k:{c:v[c]['fusion'] for c in ['all','low_positive','high_negative']} for k,v in other.items()},ensure_ascii=False))
