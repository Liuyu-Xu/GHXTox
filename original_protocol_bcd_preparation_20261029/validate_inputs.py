import json,csv,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
with (P/'data/folds.csv').open(encoding='utf8',newline='') as f:folds=list(csv.DictReader(f))
a=json.loads((P/'data/annotations.json').read_text(encoding='utf8'));assert len(a)==len(folds)==6387
assert all(x['sample_id']==z['sample_id'] and x['sequence']==z['sequence'] and x['group_id']==z['group_id'] and x['label']==int(z['label']) for x,z in zip(a,folds))
for f in range(5):
 tr=[x for x in folds if int(x['fold'])!=f];ev=[x for x in folds if int(x['fold'])==f];assert not {x['group_id'] for x in tr}&{x['group_id'] for x in ev};print('fold',f,'train',len(tr),'heldout',len(ev))
for path in sorted(P.rglob('*')):
 if path.is_file() and not any(t in path.parts for t in ('smoke','runs','__pycache__')) and path.name!='manifest.json':print(hashlib.sha256(path.read_bytes()).hexdigest(),path.relative_to(P))
print('Input consistency passed; this does not verify GPU, training equivalence, or scientific validity.')
