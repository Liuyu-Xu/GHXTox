import pathlib,urllib.request,urllib.parse,json,hashlib,datetime
P=pathlib.Path(__file__).resolve().parent
params={'query':'reviewed:true AND length:[7 TO 50]','format':'tsv','fields':'accession,id,protein_name,organism_name,length,sequence,cc_similarity'}
url='https://rest.uniprot.org/uniprotkb/stream?'+urllib.parse.urlencode(params)
f=P/'uniprot_reviewed_7_50_family.tsv'
if not f.exists():
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'GHXTox-family-field-completion'}),timeout=60) as r:
  d=r.read();headers=dict(r.headers)
 f.write_bytes(d)
 (P/'family_fetch_manifest.json').write_text(json.dumps({'url':url,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'headers':headers,'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'purpose':'symmetric missing-family-field completion, no toxicity label certification'},ensure_ascii=False,indent=2),encoding='utf-8')
print(f.stat().st_size, f.read_text(encoding='utf-8').splitlines()[0],flush=True)
