"""Read-only index of existing research reports and experiment summary files."""
import json,hashlib,re
from pathlib import Path
R=Path('D:/dachuang2026/GHXTox');O=Path(__file__).resolve().parent
paths=set()
for root in ['analysis','reports','runs','docs','experiments','.planning']:
 for p in (R/root).rglob('*'):
  if not p.is_file() or O in p.parents or (R/'.planning/2026-10-26-research-review') in p.parents:continue
  name=p.name.lower()
  if p.suffix.lower()=='.md' or (p.suffix.lower()=='.json' and any(k in name for k in ['summary','decision','evaluation','comparison'])):
   if p.stat().st_size<1_000_000:paths.add(p)
keys=re.compile(r'decision|conclusion|promotion|promote|replace_default|retain|reject|no_go|status|reason|recommend',re.I)
def extract(x,prefix='',depth=0):
 out=[]
 if depth>6:return out
 if isinstance(x,dict):
  for k,v in x.items():
   key=prefix+'.'+k if prefix else k
   if keys.search(k) and isinstance(v,(str,bool,int,float,type(None))):out.append(dict(key=key,value=v[:700] if isinstance(v,str) else v))
   elif isinstance(v,(dict,list)):out+=extract(v,key,depth+1)
 elif isinstance(x,list) and len(x)<=30:
  for i,v in enumerate(x):out+=extract(v,prefix+f'[{i}]',depth+1)
 return out[:40]
records=[]
for p in sorted(paths):
 raw=p.read_bytes();text=raw.decode('utf-8-sig',errors='replace');rec=dict(path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),kind=p.suffix)
 if p.suffix.lower()=='.json':
  try:
   x=json.loads(text);rec['top_keys']=list(x)[:25] if isinstance(x,dict) else [];rec['decisions']=extract(x)
  except ValueError:rec['parse_error']=True
 else:
  rec['headings']=[s for s in text.splitlines() if s.startswith('#')][:40]
  rec['decision_passages']=[s for s in text.splitlines() if re.search('结论|不通过|停止|不晋级|不进入|未通过|不支持|暂缓|negative|Decision|promotion|not met|do not',s,re.I)][:20]
 records.append(rec)
(O/'source_inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
counts={kind:sum(r['kind']==kind for r in records) for kind in ['.md','.json']}
(O/'inventory_summary.json').write_text(json.dumps(dict(n_files=len(records),counts=counts,analysis_topics=sorted(p.name for p in (R/'analysis').iterdir() if p.is_dir() and p!=O),index_only_no_experiments_run=True),ensure_ascii=False,indent=2),encoding='utf-8')
for rec in records:
 if rec['path'].startswith('runs/') and rec.get('decisions'):
  selected=[v for v in rec['decisions'] if re.search('decision|promote|replace_default|retain|conclusion',v['key'],re.I)]
  if selected: print(rec['path'],str(selected)[:350])
print('INDEXED',len(records),counts)
