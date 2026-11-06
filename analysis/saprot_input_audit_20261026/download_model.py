"""Download pinned public35M weights; keep all artifacts within workspace."""
from pathlib import Path
import requests,json,hashlib,time
O=Path(__file__).resolve().parent;M=O/'model';M.mkdir(exist_ok=True)
meta=json.loads((O/'hf_model_metadata.json').read_text());rev=meta['sha'];repo=meta['id'];url=f'https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true';r=requests.get(url,timeout=30);r.raise_for_status();tree=r.json();(O/'hf_file_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
names=['config.json','special_tokens_map.json','tokenizer_config.json','vocab.txt','pytorch_model.bin'];manifest=[]
for name in names:
 info=next(x for x in tree if x['path']==name);p=M/name;download=f'https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true'
 expected=info.get('lfs',{}).get('oid')
 if p.exists() and (not expected or hashlib.sha256(p.read_bytes()).hexdigest()==expected):pass
 else:
  temp=M/(name+'.partial');h=hashlib.sha256();size=0
  with requests.get(download,stream=True,timeout=(30,60)) as response:
   response.raise_for_status()
   with temp.open('wb') as w:
    for chunk in response.iter_content(4*1024*1024):
     if not chunk:continue
     w.write(chunk);h.update(chunk);size+=len(chunk)
     if size%(16*1024*1024)==0:print(f'{name}: {size//1024//1024}MiB',flush=True)
  assert size==info['size'],(name,size,info['size'])
  if expected:assert h.hexdigest()==expected
  temp.replace(p)
 actual=hashlib.sha256(p.read_bytes()).hexdigest();manifest.append(dict(file=name,revision=rev,bytes=p.stat().st_size,sha256=actual,lfs_sha256=expected,url=download))
(O/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Pinned SaProt35M files ready; hashes checked.',flush=True)
