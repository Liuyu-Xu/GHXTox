"""Class-wide descriptive evidence for the new 27 family/sequence context pairs."""
import json,math
from pathlib import Path
from collections import defaultdict
import pandas as pd
R=Path('D:/dachuang2026/GHXTox');O=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x): (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
meta=read(O/'all_training_annotation_index.json');sg=read(O/'sg_geometry_proxy.json')
pairs=pd.read_csv(O/'supplement_matched_context.csv',keep_default_na=False)
pairs=pairs[pairs.supplement=='crossfold_all327_errors_vs_all_correct'].to_dict('records')
d=pd.read_csv(R/'analysis/expert_disagreement_20261025/training_expert_atlas.csv',float_precision='round_trip').set_index('sample_id')
needed={p[k] for p in pairs for k in ['case_id','control_id']};missing=needed-set(sg)
if missing:
 import torch
 payload=torch.load('D:/dachuang2026/GHXTox-ziliao/train_chemical_sites_final_esm2_multisource_1_30_full2795.pt',map_location='cpu',mmap=True,weights_only=False)
 for r in payload['records']:
  sid=r['sample_id']
  if sid not in missing:continue
  assert r['sequence']==d.loc[sid,'sequence']
  seq=r['sequence'];mask=r['chemical_site_mask'].tolist();coords=r['chemical_site_coords'].tolist();ci=[i for i,x in enumerate(seq) if x=='C'];valid=[i for i in ci if mask[i][0]]
  dist=sorted((math.dist(coords[i][0],coords[j][0]),i+1,j+1) for n,i in enumerate(valid) for j in valid[n+1:]);used=set();edges=[]
  for distance,i,j in dist:
   if distance<=2.5 and i not in used and j not in used:edges.append((i,j));used.update([i,j])
  sg[sid]=dict(cys_n=len(ci),sg_site_n=len(valid),sg_min_distance=dist[0][0] if dist else None,sg_close_pairs=edges,sg_close_edge_fraction=len(edges)/(len(ci)//2) if len(ci)>=2 and len(ci)==len(valid) else None,sg_proxy_arc_crossings=sum(i<k<j<l or k<i<l<j for n,(i,j) in enumerate(edges) for k,l in edges[n+1:]),sg_proxy_not_experimental_bond=True)
 del payload
save('expanded_sg_geometry_proxy.json',{s:sg[s] for s in sorted(needed)})
features=['cys_count','cys_span_fraction','net_charge_proxy','mean_hydropathy','hydrophobic_contact_excess','positive_hydrophobic_contact_excess','like_charge_contact_excess','charge_hydrophobic_separation','hydrophobic_patch_fraction','local_positive_excess','ideal_alpha_hydrophobic_moment','mean_plddt','sequence_seed_std']
rows=[]; evidence=[]
for p in pairs:
 s,t=p['case_id'],p['control_id'];m,n=meta[s],meta[t];common=set(m['name_class'])&set(n['name_class']);family=sorted(common)[0] if common else p['family']
 x=dict(p,family_group=family,case_name=m['name'],control_name=n['name'],case_fragment=m['fragment_reference_flag'],control_fragment=n['fragment_reference_flag'],case_feature_record_available=bool(m['full_record_accessions']),control_feature_record_available=bool(n['full_record_accessions']),case_disulfide_status=m['disulfide_evidence_status'],control_disulfide_status=n['disulfide_evidence_status'],case_chemical_status=m['modification_evidence_status'],control_chemical_status=n['modification_evidence_status'])
 for f in features:
  a,b=float(d.loc[s,f]),float(d.loc[t,f]);x['case_'+f]=a;x['control_'+f]=b;x['diff_'+f]=a-b
 for f in ['sg_close_edge_fraction','sg_min_distance','sg_proxy_arc_crossings']:
  x['case_'+f]=sg[s][f];x['control_'+f]=sg[t][f];x['diff_'+f]=sg[s][f]-sg[t][f] if sg[s][f] is not None and sg[t][f] is not None else None
 rows.append(x);evidence.append(dict(pair=p,family_group=family,case_evidence=m,control_evidence=n,case_SG=sg[s],control_SG=sg[t]))
df=pd.DataFrame(rows);df.to_csv(O/'class_pair_features.csv',index=False,encoding='utf-8-sig');save('class_pair_evidence.json',evidence)
groups=[]
for (family,label),g in df.groupby(['family_group','label']):
 r=dict(family=family,original_label=int(label),n_pairs=len(g),case_ids=list(g.case_id),control_ids=list(g.control_id),case_fragments=int(g.case_fragment.sum()),control_fragments=int(g.control_fragment.sum()),case_full_feature_records=int(g.case_feature_record_available.sum()),control_full_feature_records=int(g.control_feature_record_available.sum()),explicit_family_pairs=int((g.family_support=='explicit_cached_family').sum()),different_fold_pairs=int((g.case_fold!=g.control_fold).sum()),descriptive_only=True)
 r['features']={f:dict(n_finite=int(g['diff_'+f].notna().sum()),mean_case_minus_control=float(g['diff_'+f].mean()) if g['diff_'+f].notna().any() else None,n_case_higher=int((g['diff_'+f]>0).sum()),n_case_lower=int((g['diff_'+f]<0).sum())) for f in features+['sg_close_edge_fraction','sg_min_distance','sg_proxy_arc_crossings']}
 ids=list(g.case_id)+list(g.control_id);r['cached_primary_pmids']=sorted({p['pmid'] for s in ids for p in meta[s]['cached_papers']})
 r['documented_chemical_descriptions']=sorted({f.get('description','') for s in ids for f in meta[s]['chemical_feature_records']})
 groups.append(r)
save('class_descriptive_summary.json',groups)
lines=['# 全类对照结果（探索性补充）','', '本表覆盖27个不复用序列组的候选配对。按原标签与家族/名称类别分层；21对来自不同折模型，只有3对有两侧明确缓存家族支持，24对为名称类别代理，不能当成27对实验终点匹配的金标准。没有新增显著性检验。', '', '|家族/名称类|原标签|对数|错题/正确对照片段数|错题/正确对照完整特征缓存数|空间电荷—疏水接触差均值|疏水接触差均值|', '|---|---:|---:|---|---|---:|---:|']
for g in groups:
 def fmt(f):
  v=g['features'][f]['mean_case_minus_control'];return '未知' if v is None else f'{v:.3f}'
 lines.append(f"|{g['family']}|{g['original_label']}|{g['n_pairs']}|{g['case_fragments']}/{g['control_fragments']}|{g['case_full_feature_records']}/{g['control_full_feature_records']}|{fmt('positive_hydrophobic_contact_excess')}|{fmt('hydrophobic_contact_excess')}|")
lines+=['', '差值均为错题减正确对照；原标签0/1分开，不将不同标签混在一起。单类配对很少，数值只描述，不能据此给某种特征排因果重要性。名称类别可以比真实同源家族宽。','', '## 每一对都保留的证据','', '|类别|原标签|错题→正确对照|序列比对相同率|身份注释与文献缓存|','|---|---:|---|---:|---|']
for p in evidence:
 m,n=p['case_evidence'],p['control_evidence'];q=p['pair'];papers={x['pmid']:x for s in [m,n] for x in s['cached_papers']}
 links='；'.join('['+x['pmid']+']('+x['url']+')' for x in papers.values()) or '原始摘要未缓存；仅数据库线索'
 links+='；'+' / '.join('['+a+'](https://www.uniprot.org/uniprotkb/'+a+'/entry)' for a in m['accessions']+n['accessions'])
 lines.append(f"|{p['family_group']}|{q['label']}|{q['case_id']} → {q['control_id']}|{q['identity']:.1%}|{links}|")
(O/'CLASS_COMPARISONS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('class strata',len(groups),'pairs',len(pairs),'groups with >=3 pairs',[(g['family'],g['original_label'],g['n_pairs']) for g in groups if g['n_pairs']>=3])
