"""Inventory and back up research sources before publication; never alter source files."""
from pathlib import Path
import json,hashlib,subprocess,zipfile
R=Path('D:/dachuang2026/GHXTox');O=R/'analysis/repository_sync_20261026';B=O/'backup';B.mkdir(exist_ok=True)
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
allowed={'.py','.json','.md','.toml','.txt','.sh','.ps1','.csv','.tsv','.svg','.yaml','.yml','.sha256','.ini','.cfg','.rst'}
excluded={'__pycache__','vendor','model','node_modules','.git','checkpoints','sources','source','rounded_probability_archive','repo_cache','download_cache','evidence_phase2'}
files=[];skipped=[]
for root in ['src','configs','scripts','analysis','reports','runs','docs','tests']:
 for p in (R/root).rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(R);parts=set(rel.parts)
  reason=None
  if 'repository_sync_20261026' in parts:reason='sync tooling/private backup'
  elif parts & excluded:reason='raw/cache/third-party/checkpoint directory'
  elif p.suffix.lower() not in allowed:reason='binary/generated/non-text'
  elif any(t in p.name.lower() for t in ['partial','stdout','stderr','bootstrap_replicates','replicate_metrics']):reason='partial/log/resampling duplicate'
  elif p.name.startswith('.') or p.name in ['AGENTS.md']:reason='local agent instructions'
  if reason:skipped.append(dict(path=rel.as_posix(),reason=reason));continue
  files.append(p)
for name in ['README.md','pyproject.toml','requirements.txt']:
 files.append(R/name)
manifest=[dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
(O/'selection_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');(O/'excluded_manifest.json').write_text(json.dumps(skipped,ensure_ascii=False,indent=2),encoding='utf-8')
(B/'tracked_changes.patch').write_bytes(git('diff','--binary','HEAD'));(B/'git_status.txt').write_bytes(git('status','--porcelain=v1','--untracked-files=all'));(B/'git_remotes.txt').write_bytes(git('remote','-v'))
with zipfile.ZipFile(B/'research_sources.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
 for p in files:archive.write(p,p.relative_to(R).as_posix())
if not (B/'original_git_history.bundle').exists():subprocess.run(['git','bundle','create',str(B/'original_git_history.bundle'),'--all'],cwd=R,check=True)
(B/'backup_checksums.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in B.iterdir() if p.is_file() and p.name!='backup_checksums.json'},indent=2),encoding='utf-8')
print(json.dumps(dict(selected_files=len(files),raw_selected_MB=round(sum(p.stat().st_size for p in files)/1e6,2),backup_zip_MB=round((B/'research_sources.zip').stat().st_size/1e6,2),excluded_files=len(skipped)),indent=2))
