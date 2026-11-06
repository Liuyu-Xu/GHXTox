import json,math,random,collections
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf8'))
meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json')
plans={'moderate':{'low_pos':2,'low_neg':3,'mid_pos':2,'mid_neg':3,'high_pos':4,'high_neg':2},'light':{'low_pos':1,'low_neg':4,'mid_pos':2,'mid_neg':3,'high_pos':5,'high_neg':1},'strong':{'low_pos':3,'low_neg':3,'mid_pos':1,'mid_neg':2,'high_pos':4,'high_neg':3}}
def category(m):return ('low' if m['cys_count']<=1 else 'high' if m['cys_count']>=6 else 'mid')+('_pos' if m['label']==1 else '_neg')
out={}
for plan,quota in plans.items():
 assert sum(quota.values())==16;out[plan]=[]
 for f in range(5):
  tr=[v for v in roles if v['outer_fold']==f and v['role']=='train'];assert all(meta[x['source_index']]['sample_id']==x['sample_id'] for x in tr)
  lists={k:[] for k in quota}
  for x in tr:
   i=x['source_index'];lists[category(meta[i])].append(i)
  rng=random.Random(20261029+f);queues={k:[] for k in quota};exposure=collections.Counter(); anchors=collections.Counter();eligible=collections.Counter();omitted=collections.Counter();steps=math.ceil(len(tr)/16)
  for _ in range(steps):
   batch=[];used=set()
   for cat,n in quota.items():
    for j in range(n):
     candidates=[i for i in lists[cat] if meta[i]['group_id'] not in used]
     if not candidates:raise RuntimeError('not enough distinct groups')
     least=min(exposure[i] for i in candidates)
     i=rng.choice([i for i in candidates if exposure[i]==least]);used.add(meta[i]['group_id']);batch.append(i);exposure[i]+=1
   for i in batch:
    m=meta[i];c=category(m).split('_')[0]
    if m['uniform_flag']:continue
    eligible[c]+=1
    others=[meta[j] for j in batch if j!=i and not meta[j]['uniform_flag'] and category(meta[j]).split('_')[0]==c]
    if any(x['label']==m['label'] for x in others) and any(x['label']!=m['label'] for x in others):anchors[c]+=1
    else:omitted[c]+=1
  summaries={}
  for cat,items in lists.items():
   x=[exposure[i] for i in items];summaries[cat]={'n':len(items),'draws':sum(x),'seen':sum(v>0 for v in x),'mean_exposure':sum(x)/len(x),'max_exposure':max(x),'min_exposure':min(x),'repeat_samples':sum(v>1 for v in x),'flagged_draws':sum(exposure[i] for i in items if meta[i]['uniform_flag'])}
  out[plan].append({'fold':f,'steps':steps,'quota':quota,'categories':summaries,'eligible_anchor_draws':dict(eligible),'effective_anchor_draws':dict(anchors),'effective_fraction':{k:anchors[k]/eligible[k] if eligible[k] else None for k in ['low','mid','high']}})
(P/'results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8')
for k,folds in out.items():
 print(k)
 for x in folds:
  cats=x['categories'];print(x['fold'],'steps',x['steps'],'lowpos',round(cats['low_pos']['mean_exposure'],2),cats['low_pos']['max_exposure'],'highneg',round(cats['high_neg']['mean_exposure'],2),cats['high_neg']['max_exposure'],'anchor', {k:round(v,2) for k,v in x['effective_fraction'].items()})
