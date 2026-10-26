"""Bounded public-source retrieval, no model or sequence design."""
from pathlib import Path
import requests,json,hashlib,concurrent.futures
O=Path(__file__).resolve().parent;S=O/'sources';S.mkdir(exist_ok=True)
jobs=[('PG1_BioC.xml','https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC12044605/unicode'),('cyclic2025.xml','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11891913/fullTextXML'),('PG1_repository.json','https://api.github.com/repos/ziul-bio/DMS_ML_AMP')]
def get(job):
 name,url=job
 try:
  r=requests.get(url,timeout=35,headers={'User-Agent':'GHXTox-data-evidence-audit/1.0'})
  (S/name).write_bytes(r.content)
  return dict(name=name,url=url,status=r.status_code,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),content_type=r.headers.get('Content-Type'))
 except requests.RequestException as e:return dict(name=name,url=url,error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:manifest=list(ex.map(get,jobs))
repo=S/'PG1_repository.json'
try:
 meta=json.loads(repo.read_text());branch=meta['default_branch'];m=get(('PG1_tree.json',f'https://api.github.com/repos/ziul-bio/DMS_ML_AMP/git/trees/{branch}?recursive=1'));manifest.append(m)
except (KeyError,ValueError):pass
(O/'retrieval_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
tree=S/'PG1_tree.json'
if tree.exists():
 for f in json.loads(tree.read_text()).get('tree',[]):
  if f.get('type')=='blob':print(f['path'],f.get('size'))
