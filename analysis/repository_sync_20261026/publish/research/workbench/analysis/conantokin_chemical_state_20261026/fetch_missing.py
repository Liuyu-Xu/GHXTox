"""Fetch only three absent full UniProt records and two primary-paper metadata records."""
import json,hashlib,time
from pathlib import Path
import requests
O=Path(__file__).resolve().parent; S=O/'sources';S.mkdir(exist_ok=True)
manifest=[]
jobs=[(a+'.json','https://rest.uniprot.org/uniprotkb/'+a+'.json',None) for a in ['P0C8E2','P0DP00','P0C8E1']]
jobs.append(('primary_papers.json','https://www.ebi.ac.uk/europepmc/webservices/rest/search',{'query':'SRC:MED AND (EXT_ID:17153339 OR EXT_ID:17962189)','format':'json','resultType':'core','pageSize':10}))
for name,url,params in jobs:
 path=S/name
 if path.exists():
  print('reuse',name,flush=True);continue
 response=requests.get(url,params=params,timeout=40);response.raise_for_status(); data=response.json()
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 manifest.append(dict(file=str(path),url=response.url,status=response.status_code,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 print('saved',name,flush=True)
 time.sleep(.4)
(O/'fetch_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
