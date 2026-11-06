import pathlib,json,csv
b=json.load(open('analysis/label_ambiguity_map_20261025/public_terminal_pair_blocks.json',encoding='utf8'))
r={x['id']:x for x in csv.DictReader(open('analysis/environment_feasibility_20261025/sources/Hemolytik2_complete_data.csv',encoding='utf-8-sig'))}
print('same smiles',sum(r[x['records'][0]['id']]['smiles']==r[x['records'][1]['id']]['smiles'] for x in b))
for x in b:
 if x['pmid']=='29108968': print([r[z['id']] for z in x['records']])
