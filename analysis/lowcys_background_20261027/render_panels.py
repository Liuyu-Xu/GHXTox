from pathlib import Path
import json
P=Path(__file__).resolve().parent
t=json.loads((P/'all_family_background_tables.json').read_text(encoding='utf-8'))
lines=['# 全部原正类低Cys家族面板','','家族入组仅依据序列和统一外部注释；没有按错误率筛选。full为完整SIMILARITY；parent为更宽上级家族，不能混为严格同背景证据。未带标记不是原标签可靠。每行同层的成员不重叠。','']
for layer,title in [('full_families','完整家族注释'),('parent_families','上级家族（探索性）')]:
 lines+=['## '+title,'','|家族|低Cys n/序列组|旧/新漏判数|低Cys有标记数|高Cys n/序列组|旧/新漏判数|','|---|---:|---:|---:|---:|---:|']
 for r in sorted(t[layer+'_label1'],key=lambda x:(-x['strata']['low']['n'],x['family'])):
  a,b=r['strata']['low'],r['strata']['high']
  if a['n']==0:continue
  lines.append(f"|{r['family']}|{a['n']}/{a['groups']}|{a['old_wrong']}/{a['new_wrong']}|{a['flagged']}|{b['n']}/{b['groups']}|{b['old_wrong']}/{b['new_wrong']}|")
 lines.append('')
(P/'ALL_FAMILY_PANELS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
