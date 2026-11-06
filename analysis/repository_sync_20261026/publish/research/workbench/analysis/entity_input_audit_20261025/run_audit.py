"""Finite, read-only entity audit. Historical probabilities are never refitted."""
import csv, hashlib, json
from pathlib import Path
import torch

ROOT = Path('D:/dachuang2026/GHXTox')
OUT = Path(__file__).resolve().parent
CACHE = Path('D:/dachuang2026/GHXTox-ziliao/train_chemical_sites_final_esm2_multisource_1_30_full2795.pt')
ATLAS = ROOT/'analysis/expert_disagreement_20261025/training_expert_atlas.csv'
PROTO = ROOT/'analysis/ghxtox_error_atlas_20261025/protocol.json'
LEDGER = ROOT/'analysis/ghxtox_error_atlas_20261025/evidence_phase2/error_accession_evidence_ledger.csv'
FAMILY = ROOT/'analysis/label_ambiguity_map_20261025/ectatomin_family_panel.json'
def readcsv(p):
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
inputs=[ATLAS,PROTO,LEDGER,FAMILY,CACHE,ROOT/'src/ghxtox/chemical_sites.py',ROOT/'src/ghxtox/preprocess.py']
before={str(p):sha(p) for p in inputs}
atlas=readcsv(ATLAS); byid={r['sample_id']:r for r in atlas}
family=json.loads(FAMILY.read_text(encoding='utf-8'))
inter=json.loads((ROOT/'analysis/label_ambiguity_map_20261025/interchain_error_panel.json').read_text(encoding='utf-8'))
ids=list(dict.fromkeys([r['sample_id'] for r in family]+[r['sample_id'] for r in inter]+['peptide_309']))
protocol=json.loads(PROTO.read_text(encoding='utf-8'))
w=protocol['fusion_weight_sequence']; threshold=protocol['fusion_threshold']
payload=torch.load(CACHE,map_location='cpu',weights_only=False,mmap=True)
records={r['sample_id']:r for r in payload['records'] if r['sample_id'] in ids}
evidence=readcsv(LEDGER)
subunits={'peptide_201','peptide_74','peptide_255','peptide_212','peptide_569','peptide_4096'}
rows=[]
for sid in ids:
    a=byid[sid]; r=records[sid]
    assert r['sequence']==a['sequence'] and int(r['label'])==int(a['label'])
    n=len(a['sequence']); assert list(r['coords'].shape)==[n,3]
    ps,pt,pf=map(float,[a['p_seq'],a['p_struct'],a['p_fusion']])
    assert abs(w*ps+(1-w)*pt-pf)<1e-8
    dims={k:list(v.shape) for k,v in r.items() if torch.is_tensor(v)}
    sulfur=r['chemical_site_types'][:,:,6]
    ev=[{k:x[k] for k in ['accession','protein_name','chain_boundary_features','documented_ptm_features','source_url']} for x in evidence if x['sample_id']==sid]
    fam=next((x for x in family if x['sample_id']==sid),{})
    row=dict(sample_id=sid,accession=fam.get('accession',ev[0]['accession'] if ev else ''),sequence=a['sequence'],length=n,cys_count=a['sequence'].count('C'),label=int(a['label']),fold=int(a['fold']),group_id=a['group_id'],
        p_seq=ps,p_struct=pt,p_fusion=pf,sequence_result=a['sequence_category'],structure_result=a['structure_category'],fusion_result=a['fusion_category'],
        fusion_margin_signed=pf-threshold,struct_probability_needed_at_fixed_seq=(threshold-w*ps)/(1-w),seq_probability_needed_at_fixed_struct=(threshold-(1-w)*pt)/w,
        cached_mean_plddt=sum(r['plddt'].tolist())/n,cached_structure_source=str(r.get('structure_source')),tensor_shapes=dims,
        explicit_chain_id_field=any('chain' in k.lower() for k in r),explicit_covalent_connectivity_field=any(any(t in k.lower() for t in ['ssbond','disulfide','bond_index','covalent']) for k in r),
        explicit_terminal_chemistry_field=any(any(t in k.lower() for t in ['acetyl','amidat','formyl','n_termin','c_termin']) for k in r),
        sulfur_site_count=int(sulfur.sum().item()),subunit_candidate=sid in subunits,evidence=ev)
    rows.append(row)
save('panel.json',rows)
save('cache_contract.json',dict(cache=str(CACHE),sha256=before[str(CACHE)],payload_metadata={k:v for k,v in payload.items() if k!='records'},record_keys=list(next(iter(records.values()))),
    caveat='辅助输入核查缓存来自full2795，不是重跑旧OOF；旧概率独立保留。原始PDB未在已检索资料目录恢复，无法核实其链/SSBOND记录。'))
lines=['# 分子对象与输入核查','',f'本轮只核查固定的{len(rows)}条训练样本。保留原标签、模型和旧OOF概率；未读取测试预测。','',
    '## 实际输入能确认什么','',
    '已逐条核对缓存序列、标签与既有训练表一致，Cα坐标数均等于输入序列长度。Ectatomin A、B分别作为独立样本进入；缓存没有额外伴侣链的残基坐标。不能把这份缓存称为完整A+B复合物。',
    '缓存有硫原子位点信息，但没有显式链ID、二硫键配对或末端化学状态字段。坐标可能隐含空间接近；“有硫位点”不等于“已知道哪个Cys与哪条链连接”。',
    '现有chemical_sites.py解析器只读取ATOM，按链号/残基号分组后返回残基列表；它不解析SSBOND、LINK、CONECT或HETATM修饰。此为当前代码事实，不能据此反推所有历史PDB内容。原始PDB未恢复。',
    '此核查使用full2795辅助缓存；概率仍取历史结构OOF。两者不作为同一检查点的重新推理结果，不能从缓存变化推断旧预测变化。','',
    '## 一维、三维和融合发生了什么','',
    '|样本|长度|原标签|一维概率|三维概率|融合概率|一维/三维/融合|折|',
    '|---|---:|---:|---:|---:|---:|---|---:|']
for r in rows:
    lines.append(f"|{r['sample_id']} ({r['accession']})|{r['length']}|{r['label']}|{r['p_seq']:.4f}|{r['p_struct']:.4f}|{r['p_fusion']:.4f}|{r['sequence_result']}/{r['structure_result']}/{r['fusion_result']}|{r['fold']}|")
A=next(r for r in rows if r['sample_id']=='peptide_201'); B=next(r for r in rows if r['sample_id']=='peptide_74')
lines+=['',f"固定融合公式为0.72×一维概率+0.28×三维概率，阈值{threshold:.4f}。一维自身参考阈值另为{protocol['sequence_reference_threshold']:.4f}，三维另为{protocol['structure_reference_threshold']:.4f}，因此不能仅按分支TP/FN推断融合结果。",'',
    f"Et1a A链（201）的一维仅{A['p_seq']:.4f}，虽然超过一维参考阈值，但三维仅{A['p_struct']:.4f}，融合后降至{A['p_fusion']:.4f}而漏判。B链（74）的一维为{B['p_seq']:.4f}，三维为{B['p_struct']:.4f}，融合为{B['p_fusion']:.4f}。B链正确不能当成三维识别了完整毒素的证据。",'',
    f"保持一维不动，A链三维概率需达到{A['struct_probability_needed_at_fixed_seq']:.4f}才能让融合翻转。这只是公式诊断，不是重新设阈值的建议，也不能证明缺失伴侣链造成了漏判。",'',
    'A链201在折0，B链74在折3；Et1b A链255在折0、B链212在折4，均属于不同分组。因此它们的OOF概率来自不同折模型，不是同一模型下的配对干预实验。实际缓存中212为36残基，以序列逐字符核对结果为准。','',
    '## 生物学含义与边界','',
    '已有原文追溯表明Et1a实验效应对应二硫键连接的A+B整体。当前输入却拆成两条链。这支持“分子对象不一致需要单独标记”，还不足以断言原标签错误，更不足以要求单独A链一定有同样毒性。原文：[1999年Ectatomin研究](https://doi.org/10.1046/j.1432-1327.1999.00426.x)。',
    'Et1b、Eb1a与Myrmexin只是亚基候选，不能直接借用Et1a实验；Ponericin Q42/Q49/Q50仅名称相关，并不是本轮确认的A/B亚基。Pleuturegin的28残基片段也不能直接等同原文测定的完整38kDa蛋白。详见panel.json已缓存来源与此前证据报告。','',
    '## 下一步计算优先级','',
    '1. 主线转回4条有直接效应记录的漏判候选（249、463、94、1434）：区分一维本来就低和融合压低两类，再找同终点的正确对照。重点检验局部电荷/疏水排列是否有跨折稳定信息。',
    '2. 亚基与片段保留为适用范围旁表。需要完整分子和对应标签才能公平训练组装模型，当前不把A+B直接串接送入单链网络。',
    '3. H1/H2的修饰冲突暂不当作已确认纠错机制。公开同序列末端对照可作辅助验证，暂不加权、不微调大模型。','',
    '本轮是病例诊断，不是新模型性能验证。旧OOF含校准与历史选择限制；少数病例不能证明普遍机制。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
after={str(p):sha(p) for p in inputs}
assert before==after
save('verification.json',dict(panel_size=len(rows),input_hashes_before=before,input_hashes_after=after,inputs_unchanged=True,sequence_label_and_coordinate_counts_match=True,fusion_formula_verified=True,model_trained=False,test_predictions_read=False,raw_pdb_recovered=False))
print(json.dumps(dict(n=len(rows),a=A['p_fusion'],b=B['p_fusion'],b_seq=B['p_seq'],b_struct=B['p_struct'],unchanged=True),ensure_ascii=False))
