"""Publish selected research beside an unchanged upstream public implementation."""
from pathlib import Path
import json,gzip,hashlib,subprocess,re,os
R=Path('D:/dachuang2026/GHXTox');O=R/'analysis/repository_sync_20261026';P=O/'publish';W=P/'research/workbench'
assert P.resolve()==Path('D:/dachuang2026/GHXTox/analysis/repository_sync_20261026/publish').resolve()
W.mkdir(parents=True,exist_ok=True);read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));sha=lambda b:hashlib.sha256(b).hexdigest()
selected=read(O/'selection_manifest.json');mapping={};manifest=[];secret_hits=[]
patterns=[r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{40,}',r'(?<![A-Z0-9])AKIA(?=[A-Z0-9]{0,15}[0-9])[A-Z0-9]{16}(?![A-Z0-9])',r'sk-proj-[A-Za-z0-9_-]{40,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
for m in selected:
 rel=m['path'];data=(R/rel).read_bytes();assert sha(data)==m['sha256'],rel
 text=data.decode('utf-8-sig',errors='replace')
 for pat in patterns:
  if re.search(pat,text):secret_hits.append(dict(path=rel,pattern=pat))
 if rel=='README.md':dest='HISTORICAL_README.md'
 else:dest=rel
 packed=len(data)>512*1024
 if packed:dest+='.gz'
 mapping[rel]=dest
 manifest.append(dict(source=rel,published=dest,source_bytes=len(data),source_sha256=sha(data),compressed=packed))
if secret_hits:
 (O/'secret_scan_findings.json').write_text(json.dumps(secret_hits,indent=2),encoding='utf-8');raise RuntimeError('Potential credentials found; see local path-only findings')
for m in manifest:
 data=(R/m['source']).read_bytes();dest=W/m['published'];dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.suffix=='.md':
  text=data.decode('utf-8-sig')
  def link(match):
   target=match.group(1).strip('<>');fragment=''
   if '#' in target:target,fragment=target.split('#',1);fragment='#'+fragment
   normalized=target.replace('\\','/');prefix='D:/dachuang2026/GHXTox/'
   if normalized.startswith(prefix):source=normalized[len(prefix):]
   elif not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) and not target.startswith('#'):
    source=os.path.normpath(str(Path(m['source']).parent/target)).replace('\\','/')
   else:return match.group(0)
   if source in mapping:
    local=os.path.relpath(W/mapping[source],dest.parent).replace('\\','/');return ']('+local+fragment+')'
   if source.startswith('research/multiconformer/'):
    local=os.path.relpath(P/source,dest.parent).replace('\\','/');return ']('+local+fragment+')'
   return match.group(0)
  text=re.sub(r'\]\(([^\n)]+)\)',link,text);data=text.encode('utf-8')
 if m['compressed']:data=gzip.compress(data,compresslevel=9,mtime=0)
 dest.write_bytes(data);m.update(published_bytes=len(data),published_sha256=sha(data))
# Preserve fork's own historical additions/edits in a non-runtime archive.
base=subprocess.check_output(['git','merge-base','origin/main','upstream/main'],cwd=P,text=True).strip()
changed=subprocess.check_output(['git','diff','--name-only',base,'origin/main'],cwd=P,text=True,encoding='utf-8').splitlines();legacy=[]
for rel in changed:
 if Path(rel).suffix.lower() not in {'.py','.md','.json','.toml','.txt','.sh','.ps1','.csv','.png','.svg'} and rel!='.gitignore':continue
 r=subprocess.run(['git','show',f'origin/main:{rel}'],cwd=P,capture_output=True)
 if r.returncode:continue
 data=r.stdout
 if len(data)>5*1024*1024:continue
 dest=P/'research/legacy_hypeptox'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);legacy.append(dict(path=rel,bytes=len(data),sha256=sha(data)))
(W/'ASSET_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(P/'research/legacy_hypeptox/ARCHIVE_NOTE.md').write_text('# Legacy fork work\n\nFork commit `49a330f` contained early HyPepTox-Fuse tools and reports. Its changed text/small figure files are preserved here before synchronizing upstream. This archive is not imported by the public inference code. Run paths and README instructions are historical; weights/caches were never part of this source export.\n',encoding='utf-8')
(P/'research/legacy_hypeptox/ARCHIVE_MANIFEST.json').write_text(json.dumps(legacy,indent=2),encoding='utf-8')
(W/'README.md').write_text('''# GHXTox研究工作台（2026-10-26）

这里保留研究版源码、冻结协议、正负结果和分析报告。仓库根目录采用上游公开实现；这里的实验不自动改变公开模型。原工作目录未删除或覆盖。

## 从哪里读

- [全部研究总览](reports/研究分析总览_20261025.md)
- [尝试总结与后续优先级](reports/尝试总结与研究优先级_20261026.md)
- [SaProt分类及同规模融合](analysis/saprot_standalone_20261026/REPORT.md)
- [SaProt逐样本贡献分析](analysis/saprot_contribution_20261026/REPORT.md)
- [完整纠错与添错名单](analysis/saprot_contribution_20261026/SAMPLE_LIST.md)
- [历史完整研究README](HISTORICAL_README.md)
- [研究复现和资产说明](REPRODUCIBILITY.md)

## 最新发现

冻结表征上的轻量分类头：同规模4成员基线MCC0.82594，SaProt结构融合0.84006，全结构掩码融合0.83889。结构融合纠正101条、添错63条，净少38条；其中34条来自减少原负类误报。低Cys原正类净添8条错，76条膜作用注释阳性纠0添6。尚不能归因为三维/溶血机制。101条纠错中只有18条也是历史完整GHXTox的错题。

以上不是完整GHXTox的新成绩。两融合仍为研究候选，SaProt蒸馏尚未执行。下一步需同时控制总体性能与低Cys正类/膜作用候选的漏判，并保留普通自蒸馏及全掩码对照。

## 文件组织

`src/`、`configs/`、`scripts/`、`tests/`为研究版实现；`analysis/`和`runs/`为协议与保留结果；`reports/`为汇总入口。大于512KiB的文本结果用`.gz`无损保存，SHA256和原路径在`ASSET_MANIFEST.json`。权重、特征、下载原文及运行日志未打包。负结果同样保留，不能把文件名视作方法已验证成功。
''',encoding='utf-8')
(W/'REPRODUCIBILITY.md').write_text('''# 研究复现与资产说明

1. 根目录是同步的上游公开模型；研究工作台是独立代码快照，不能混用checkpoint、数据终点或阈值。默认完整模型与轻量分类试验分开报告。
2. 每个分析目录保留协议、脚本和紧凑结果。大文本为gzip；先在本目录运行`python restore_assets.py --verify-only`校验，再运行`python restore_assets.py`恢复原路径。恢复只创建不存在的文件，不覆盖已有不同内容。
3. `ASSET_MANIFEST.json`记录原文件及发布文件SHA256。Markdown仅调整发布副本的本地链接，原备份未变。原README保存为HISTORICAL_README.md。
4. 此快照不是全量训练数据包。权重/PLM特征/ESMFold与原始骨架缓存不上传；部分第三方原始下载与中间文件也不上传。不能声称仅clone即可重训所有历史实验。旧source路径、D盘绝对路径和Python环境名保留为溯源，跨机器执行需要按脚本数据路径配置；不要对缺失实验输入用伪数据替代。
5. SaProt关键逐样本概率、纠错/添错、bootstrap及核验结果已保留；没有重新训练或挑选新种子。原标签疑点并非确认错标，旧测试集不是盲外部队列。
6. 根目录依赖与研究版pyproject/requirements有区别，应在独立虚拟环境检查研究模块，避免覆盖已有科学计算环境。研究版代码以`PYTHONPATH=research/workbench/src`运行；这不会自动验证未提供资产的训练任务。
7. 下载论文PDF/Word、模型二进制、原始第三方源码目录、临时规划、私人备份与日志不进入发布成果。历史fork独有工具另在../legacy_hypeptox，不能覆盖根目录最新版。
''',encoding='utf-8')
(W/'restore_assets.py').write_text('''"""Verify the research export and optionally restore compressed text inputs."""
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
''',encoding='utf-8')
rootread=P/'README.md';text=rootread.read_text(encoding='utf-8');text+='''

## Fork research workbench (2026-10-26)

This fork synchronizes the upstream public implementation at `a03f846` and
preserves additional research in [research/workbench](research/workbench/README.md).
The public model and its default inference configuration remain upstream code.
Research protocols, negative results, SaProt fusion controls and per-sample
contribution analyses are documented separately. Frozen-feature head results
must not be reported as improvements to the full GHXTox model, and SaProt
distillation has not yet been run. Large text results are compressed with a
checksum-bearing restoration helper; checkpoints and feature caches are not
included. Earlier fork-specific tools are archived in
[research/legacy_hypeptox](research/legacy_hypeptox/ARCHIVE_NOTE.md).
''';rootread.write_text(text,encoding='utf-8')
ignore=P/'.gitignore';ignore.write_text(ignore.read_text(encoding='utf-8')+'\n# Research snapshot: selected text assets only; binaries remain excluded.\n!research/workbench/**/\n!research/workbench/**/*.csv\n!research/workbench/**/*.tsv\n!research/workbench/**/*.gz\n!research/workbench/scripts/**\n!research/workbench/runs/**\n!research/legacy_hypeptox/**/\n!research/legacy_hypeptox/**/*.csv\n!research/legacy_hypeptox/**/*.tsv\n',encoding='utf-8')
report=dict(selected_files=len(manifest),compressed_files=sum(m['compressed'] for m in manifest),published_MB=round(sum(m['published_bytes'] for m in manifest)/1e6,2),legacy_files=len(legacy),potential_secret_findings=0,source_hashes_all_match=True)
(O/'publication_selection_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
