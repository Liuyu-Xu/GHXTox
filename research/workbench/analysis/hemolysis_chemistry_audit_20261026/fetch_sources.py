import requests,json,pathlib
p=pathlib.Path('analysis/hemolysis_chemistry_audit_20261026/sources')
u='https://www.ebi.ac.uk/europepmc/webservices/rest/search'
r=requests.get(u,params={'query':'EXT_ID:(17980935 OR 10879468 OR 28230084 OR 29108968 OR 34361810 OR 34819923)','format':'json','resultType':'core'},timeout=45);r.raise_for_status();(p/'europepmc_six_core.json').write_bytes(r.content)
for x in r.json()['resultList']['result']:
 print(x['id'],x.get('title'),x.get('pmcid'),x.get('doi'),x.get('fullTextUrlList'))
