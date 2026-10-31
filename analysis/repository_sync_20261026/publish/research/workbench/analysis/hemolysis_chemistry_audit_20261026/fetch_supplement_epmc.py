import requests,pathlib,zipfile,io
p=pathlib.Path('analysis/hemolysis_chemistry_audit_20261026/sources')
u='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8606745/supplementaryFiles'
r=requests.get(u,timeout=40);print(r.status_code,len(r.content),r.headers.get('content-type'))
if r.content[:2]==b'PK':
 (p/'PMC8606745_supplementary.zip').write_bytes(r.content)
 with zipfile.ZipFile(io.BytesIO(r.content)) as z:
  print(z.namelist())
  for n in z.namelist():
   if pathlib.PurePosixPath(n).name.lower() in ['table_1.docx','table_4.docx']: (p/pathlib.PurePosixPath(n).name).write_bytes(z.read(n))
