"""Verify the research export and optionally restore compressed text inputs."""
from pathlib import Path
import argparse,json,hashlib,gzip
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
digest=lambda data:hashlib.sha256(data).hexdigest()
records=json.loads((root/'ASSET_MANIFEST.json').read_text(encoding='utf-8'))
restored=0
for record in records:
 path=(root/record['published']).resolve()
 if not path.is_relative_to(root):raise ValueError('Invalid manifest path')
 raw=path.read_bytes()
 if digest(raw)!=record['published_sha256']:raise ValueError('Published checksum mismatch: '+record['published'])
 if record['compressed']:
  content=gzip.decompress(raw)
  if digest(content)!=record['source_sha256']:raise ValueError('Source checksum mismatch: '+record['source'])
  if not args.verify_only:
   target=path.with_suffix('')
   if target.exists() and target.read_bytes()!=content:raise FileExistsError(str(target))
   if not target.exists():target.write_bytes(content);restored+=1
print('Verified',len(records),'assets; restored',restored)
