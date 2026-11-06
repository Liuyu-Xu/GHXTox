"""Validate snapshot integrity, Python syntax, links and unchanged public tree."""
from pathlib import Path
import json,ast,gzip,re,subprocess,hashlib
O=Path(__file__).resolve().parent;P=O/'publish';W=P/'research/workbench'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
syntax=[]
for root in [P/'src',P/'tests',W/'src',W/'scripts',W/'analysis',W/'tests']:
 for path in root.rglob('*.py'):
  try:ast.parse(path.read_text(encoding='utf-8-sig'))
  except Exception as e:syntax.append(dict(path=str(path.relative_to(P)),error=str(e)))
assert not syntax,syntax
diff=subprocess.check_output(['git','diff','--name-only','upstream/main','--','src','configs','tests','dataset','pyproject.toml','requirements.txt','research/multiconformer'],cwd=P,text=True).splitlines();assert not diff,diff
links=[]
for p in [W/'README.md',W/'REPRODUCIBILITY.md',W/'reports/研究分析总览_20261025.md',W/'reports/尝试总结与研究优先级_20261026.md',W/'analysis/saprot_contribution_20261026/REPORT.md',P/'README.md']:
 for t in re.findall(r'\]\(([^\n)]+)\)',p.read_text(encoding='utf-8')):
  t=t.strip('<>').split('#')[0]
  if not t or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',t):continue
  dest=(p.parent/t).resolve()
  if not dest.exists():links.append(dict(page=str(p.relative_to(P)),target=t))
files=list(W.rglob('*'));large=[str(p.relative_to(P)) for p in files if p.is_file() and p.stat().st_size>50*1024*1024];assert not large
report=dict(public_code_configs_tests_data_unchanged=True,python_syntax_errors=syntax,files_over50MiB=large,critical_entry_missing_links=links,selected_assets=len(read(W/'ASSET_MANIFEST.json')),asset_verify_command='python research/workbench/restore_assets.py --verify-only',secret_scan_note='No high-confidence tokens or private keys; initial AWS-shaped matches were amino-acid sequences and pattern was refined to require digits and boundaries.')
(O/'publication_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
