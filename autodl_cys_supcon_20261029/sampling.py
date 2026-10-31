import math,random,collections
QUOTA={'low_pos':2,'low_neg':3,'mid_pos':2,'mid_neg':3,'high_pos':4,'high_neg':2}
def group(m):return 'low' if m['cys_count']<=1 else 'high' if m['cys_count']>=6 else 'mid'
def category(m):return group(m)+('_pos' if m['label']==1 else '_neg')
def batches(rows,indices,seed,balanced):
 rng=random.Random(seed);steps=math.ceil(len(indices)/16)
 if not balanced:
  order=list(indices);rng.shuffle(order)
  return [order[k:k+16] for k in range(0,len(order),16)]
 pools={k:[] for k in QUOTA}
 for i in indices:pools[category(rows[i])].append(i)
 if any(not x for x in pools.values()):raise ValueError('Empty required category')
 exposure=collections.Counter();result=[]
 for _ in range(steps):
  batch=[];used=set()
  for cat,n in QUOTA.items():
   for _ in range(n):
    choices=[i for i in pools[cat] if rows[i]['group_id'] not in used]
    if not choices:raise ValueError('Not enough distinct sequence groups in batch')
    least=min(exposure[i] for i in choices);i=rng.choice([i for i in choices if exposure[i]==least]);batch.append(i);used.add(rows[i]['group_id']);exposure[i]+=1
  result.append(batch)
 return result

def supcon(features,labels,groups,eligible,temperature=0.1,subset=None):
 import torch
 z=torch.nn.functional.normalize(features.float(),dim=1); logits=z@z.T/temperature;n=len(labels)
 allowed=torch.tensor([[groups[i]!=groups[j] for j in range(n)] for i in range(n)],device=labels.device,dtype=torch.bool)
 e=eligible if subset is None else eligible & subset
 valid=allowed&e[:,None]&e[None,:];pos=valid&(labels[:,None]==labels[None,:]);neg=valid&(labels[:,None]!=labels[None,:]);anchors=(pos.sum(1)>0)&(neg.sum(1)>0)
 if not anchors.any():return features.sum()*0,0
 masked=logits.masked_fill(~valid,float('-inf'));lp=masked[anchors]-torch.logsumexp(masked[anchors],dim=1)[:,None];pa=pos[anchors];safe=torch.where(pa,lp,torch.zeros_like(lp));return -(safe.sum(1)/pa.sum(1)).mean(),int(anchors.sum())
