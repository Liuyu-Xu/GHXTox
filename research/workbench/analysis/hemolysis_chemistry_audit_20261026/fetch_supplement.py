import requests,pathlib
p=pathlib.Path('analysis/hemolysis_chemistry_audit_20261026/sources')
u='https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2021.753874/full'
r=requests.get(u,timeout=40);(p/'frontiers_full.html').write_bytes(r.content)
import re
for a in re.findall(r'https?[^\s"<>]+',r.text):
 if 'Table_4' in a or 'Table_1.DOCX' in a: print(a)
import requests,pathlib,concurrent.futures
p=pathlib.Path('analysis/hemolysis_chemistry_audit_20261026/sources')
def get(n):
 u=f'https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2021.753874/supplementary-material/10.3389-fmicb.2021.753874.s00{n}.docx'
 r=requests.get(u,timeout=25)
 if r.content[:2]==b'PK':
  (p/f'frontiers_s00{n}.docx').write_bytes(r.content); print(n,'ok',len(r.content),u)
 else: print(n,r.status_code,r.headers.get('content-type'),len(r.content))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(get,[3,6]))
