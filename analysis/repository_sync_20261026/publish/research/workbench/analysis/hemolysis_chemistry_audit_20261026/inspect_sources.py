import csv,json,pathlib,xml.etree.ElementTree as E
p=pathlib.Path('analysis')
b=json.load(open(p/'label_ambiguity_map_20261025/public_terminal_pair_blocks.json',encoding='utf8'))
rows=list(csv.DictReader(open(p/'environment_feasibility_20261025/sources/Hemolytik2_complete_data.csv',encoding='utf-8-sig')))
print(rows[0].keys())
ids={i for x in b for r in x['records'] for i in r['supporting_database_ids']}
selected=[r for r in rows if str(next(iter(r.values()))) in ids]
print('selected',len(selected)); print(json.dumps(selected[:2],ensure_ascii=False,indent=2))
for x in b:
 print(x['pmid'],x['sequence'],x['dose_uM'],x['pH'],[(r['id'],r['name'],r['c_terminal'],r['n_terminal'],r['hemolysis_percent']) for r in x['records']])
for f in ['PMC8348142','PMC5322399','PMC8606745']:
 root=E.parse(p/'hemolysis_chemistry_audit_20261026/sources'/f'{f}.xml').getroot()
 print('\n',f)
 for t in root.iter('table-wrap'):
  text=' '.join(t.itertext())
  if 'FIRRI' in text or 'MreB' in text or 'Supplementary' in text: print(text[:4500])
 for s in root.iter('supplementary-material'): print(E.tostring(s,encoding='unicode')[:1500])

