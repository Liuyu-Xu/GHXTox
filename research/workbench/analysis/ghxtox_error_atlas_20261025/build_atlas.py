"""Read-only GHXTox error discovery. Run with the retained pytorch_env Python.

No external test inputs, toxicity training, threshold fitting, or model changes.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

OUT = Path(__file__).resolve().parent
ROOT = Path('D:/dachuang2026/GHXTox')
ASSETS = Path('D:/dachuang2026/GHXTox-ziliao')
P = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
KD = dict(zip('ACDEFGHIKLMNPQRSTVWY', [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
TEST_FEATURES = [
    'hydrophobic_contact_excess', 'like_charge_contact_excess',
    'positive_hydrophobic_contact_excess', 'charge_hydrophobic_separation',
    'hydrophobic_patch_fraction', 'local_positive_excess',
    'local_hydrophobic_excess', 'ideal_alpha_hydrophobic_moment',
    'ideal_beta_hydrophobic_moment', 'nonlocal_contact_density',
    'mean_plddt', 'low_plddt_fraction', 'sequence_seed_std',
    'max_otherfold_kmer_cosine'
]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def save(name, df):
    df.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.17g')

def metrics(y,p,t):
    y=np.asarray(y);p=np.asarray(p); pred=p>=t
    tp=int(np.sum(pred&(y==1)));tn=int(np.sum(~pred&(y==0)))
    fp=int(np.sum(pred&(y==0)));fn=int(np.sum(~pred&(y==1)))
    return dict(n=len(y),TP=tp,TN=tn,FP=fp,FN=fn,errors=fp+fn,
        accuracy=float(np.mean(pred==y)),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
        mcc=float(matthews_corrcoef(y,pred)),auroc=float(roc_auc_score(y,p)),
        auprc=float(average_precision_score(y,p)),brier=float(np.mean((p-y)**2)))

def category(y,p,t):
    return np.where(p>=t,np.where(y==1,'TP','FP'),np.where(y==1,'FN','TN'))

def seq_features(s):
    n=len(s);h=np.array([KD[a] for a in s]);q=np.array([1. if a in 'KR' else -1. if a in 'DE' else 0 for a in s])
    hyd=(h>0).astype(float);pos=(q>0).astype(float)
    w=min(7,n); kernel=np.ones(w)/w
    return dict(length=n,net_charge_proxy=q.sum(),charge_density=q.mean(),mean_hydropathy=h.mean(),
        cys_count=s.count('C'),cys_fraction=s.count('C')/n,aromatic_fraction=sum(s.count(a) for a in 'FWY')/n,
        proline_fraction=s.count('P')/n,glycine_fraction=s.count('G')/n,
        local_positive_excess=float(np.max(np.convolve(pos,kernel,'valid'))-pos.mean()),
        local_hydrophobic_excess=float(np.max(np.convolve(hyd,kernel,'valid'))-hyd.mean()),
        ideal_alpha_hydrophobic_moment=float(abs(np.mean(h/4.5*np.exp(1j*np.arange(n)*np.deg2rad(100))))),
        ideal_beta_hydrophobic_moment=float(abs(np.mean(h/4.5*np.exp(1j*np.arange(n)*np.pi)))))

def structure_features(s,xyz):
    n=len(s); h=np.array([KD[a]>0 for a in s]);q=np.array([1 if a in 'KR' else -1 if a in 'DE' else 0 for a in s]);pos=q>0
    d=cdist(xyz,xyz); sep=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    eligible=(sep>2)&np.triu(np.ones((n,n),dtype=bool),1); contacts=eligible&(d<8.)
    def excess(pair):
        return float(pair[contacts].mean()-pair[eligible].mean()) if contacts.any() else 0.
    hh=h[:,None]&h[None,:]
    qq=q[:,None]*q[None,:]
    ph=(pos[:,None]&h[None,:])|(h[:,None]&pos[None,:])
    centered=xyz-xyz.mean(0);rg=np.sqrt(np.mean(np.sum(centered**2,axis=1)))
    separation=float(np.linalg.norm(xyz[h].mean(0)-xyz[pos].mean(0))/max(rg,1e-8)) if h.any() and pos.any() else np.nan
    # Largest C-alpha hydrophobic proximity component; not solvent exposure or a membrane-bound patch.
    hydro_idx=np.where(h)[0];seen=set();largest=0
    for start in hydro_idx:
        if int(start) in seen:continue
        stack=[int(start)];seen.add(int(start));count=0
        while stack:
            u=stack.pop();count+=1
            for v in hydro_idx[(d[u,hydro_idx]<8.)&(d[u,hydro_idx]>0)]:
                if int(v) not in seen:seen.add(int(v));stack.append(int(v))
        largest=max(largest,count)
    return dict(hydrophobic_contact_excess=excess(hh),like_charge_contact_excess=excess(qq),
        positive_hydrophobic_contact_excess=excess(ph),charge_hydrophobic_separation=separation,
        hydrophobic_patch_fraction=largest/max(int(h.sum()),1),nonlocal_contact_density=float(contacts.sum()/max(eligible.sum(),1)),
        ca_radius_gyration=float(rg))

def build():
    print('Audit identities and frozen probabilities',flush=True)
    paths={
        'contract':ROOT/'runs/unified_inference_oof_v1/contract.csv',
        'folds':ROOT/'data/folds/train_cdhit080_fallback_folds.csv',
        'calibrated_sequence':ROOT/'runs/lora_mamba_full/oof_calibrated.json',
        'archived_structure_oof':ROOT/'runs/conformer_ensemble_residual_oof/full_calibrated_predictions.csv',
        'fusion_spec':ROOT/'runs/lora_mamba_full/fusion_refreeze.json',
        'structure_cache':ASSETS/'train_chemical_sites_final_esm2_multisource_1_30_full2795.pt',
        'historical_selection_provenance':ROOT/'configs/provenance/historical_selected_base_v1/fold0_historical_selected_base.provenance.json'
    }
    for seed in [42,123,2025]:paths[f'seed{seed}']=ROOT/f'runs/lora_mamba_full/oof_seed{seed}.json'
    # Exact round-trip parsing matters: peptide_601 is exactly on the frozen threshold.
    df=pd.read_csv(paths['contract'],float_precision='round_trip');manifest=pd.read_csv(paths['folds']);struct=pd.read_csv(paths['archived_structure_oof'],float_precision='round_trip')
    assert len(df)==6387 and df.sample_id.is_unique
    for key in ['source_index','sample_id','label','sequence','fold','group_id']:
        assert np.array_equal(df[key],manifest[key]),key
    for key in ['source_index','sample_id','label','sequence','fold']:
        assert np.array_equal(df[key],struct[key]),key
    assert manifest.groupby('group_id').fold.nunique().max()==1
    assert df.groupby('sequence').label.nunique().max()==1
    for key in ['p_seq','p_struct','p_fusion']:
        assert np.isfinite(df[key]).all() and df[key].between(0,1).all()
    one=json.loads(paths['calibrated_sequence'].read_text());seeds=[]
    for seed in [42,123,2025]:
        x=json.loads(paths[f'seed{seed}'].read_text())
        for key,col in [('sample_ids','sample_id'),('labels','label'),('folds','fold')]:assert np.array_equal(x[key],df[col])
        seeds.append(x['oof'])
    seeds=np.array(seeds).T;raw=seeds.mean(1)
    a=one['platt']['slope'];b=one['platt']['intercept']
    # Retained calibration script uses EPS=1e-7 (public inference uses a different clip).
    raw=np.clip(raw,1e-7,1-1e-7);cal=1/(1+np.exp(-(a*np.log(raw/(1-raw))+b)))
    assert np.max(np.abs(cal-df.p_seq))<1e-10
    assert np.max(np.abs(struct.toxicity_probability-df.p_struct))<1e-10
    assert np.max(np.abs(.72*df.p_seq+.28*df.p_struct-df.p_fusion))<1e-10
    spec=json.loads(paths['fusion_spec'].read_text());baseline=metrics(df.label,df.p_fusion,P['fusion_threshold'])
    for k,v in spec['fusion_oof_metrics'].items():
        name={'balanced_accuracy':None}.get(k,k)
        if name:assert abs(baseline[name]-v)<1e-10,k
    df['sequence_seed_std']=seeds.std(1,ddof=1)
    # Per-seed classifications are reference diagnostics, not independently calibrated seed models.
    clipped=np.clip(seeds,1e-7,1-1e-7);seedcal=1/(1+np.exp(-(a*np.log(clipped/(1-clipped))+b)))
    for j,seed in enumerate([42,123,2025]):df[f'p_sequence_seed{seed}_raw']=seeds[:,j]
    df['seed_reference_disagreement']=(seedcal>=P['sequence_reference_threshold']).any(1)&~(seedcal>=P['sequence_reference_threshold']).all(1)
    df['all_seed_reference_wrong']=np.all((seedcal>=P['sequence_reference_threshold'])!=df.label.to_numpy()[:,None],axis=1)
    for key,col,t in [('fusion','p_fusion',P['fusion_threshold']),('sequence','p_seq',P['sequence_reference_threshold']),('structure','p_struct',P['structure_reference_threshold'])]:
        df[f'{key}_category']=category(df.label,df[col],t)
        df[f'{key}_error']=df[f'{key}_category'].isin(['FP','FN'])
    df['expert_decision_disagreement']=(df.p_seq>=P['sequence_reference_threshold'])!=(df.p_struct>=P['structure_reference_threshold'])
    df['fusion_corrects_sequence']=df.sequence_error&~df.fusion_error
    df['fusion_harms_sequence']=~df.sequence_error&df.fusion_error
    df['fusion_margin']=np.abs(df.p_fusion-P['fusion_threshold'])
    band=float(df.fusion_margin.quantile(.1))
    df['near_boundary_10pct']=df.fusion_margin<=band
    df['analysis_partition']=np.where(df.fold.isin(P['discovery_folds']),'discovery','internal_replication')
    seqdf=pd.DataFrame([seq_features(s) for s in df.sequence])
    for col in seqdf:df[col]=seqdf[col]
    for col in ['mean_plddt','min_plddt','low_plddt_fraction','global_gate']:df[col]=struct[col].to_numpy()
    print('Extract single-structure geometry (no conformer features)',flush=True)
    import torch
    payload=torch.load(paths['structure_cache'],map_location='cpu',weights_only=False)
    records=payload['records'];assert len(records)==len(df)
    extra=[]
    for i,rec in enumerate(records):
        assert rec['sequence']==df.sequence.iloc[i] and rec['sample_id']==df.sample_id.iloc[i] and int(rec['label'])==int(df.label.iloc[i])
        xyz=np.asarray(rec['coords'].tolist(),dtype=float)
        x=structure_features(rec['sequence'],xyz)
        x['cached_mean_plddt']=float(np.mean(rec['plddt'].tolist()))
        x['structure_available']=bool(np.isfinite(xyz).all())
        # No endpoint/source evidence can be inferred from a generic peptide|label header.
        extra.append(x)
    del records,payload
    for col in extra[0]:df[col]=[x[col] for x in extra]
    df['endpoint_evidence']='unknown_in_retained_records'
    df['assay_concentration']='unknown'
    df['original_database']='unknown'
    df['terminal_modifications']='unknown'
    print('Compute cross-fold sequence-neighborhood proxy',flush=True)
    vec=TfidfVectorizer(analyzer='char',ngram_range=(3,4),lowercase=False,dtype=np.float32)
    matrix=vec.fit_transform(df.sequence)
    sims=np.empty(len(df));nn=np.empty(len(df),dtype=int)
    for fold in range(5):
        ix=np.where(df.fold.to_numpy()==fold)[0];train=np.where(df.fold.to_numpy()!=fold)[0]
        scores=(matrix[ix]@matrix[train].T).toarray()
        best=scores.argmax(1);sims[ix]=scores[np.arange(len(ix)),best];nn[ix]=train[best]
    df['max_otherfold_kmer_cosine']=sims
    df['nearest_otherfold_kmer_sample_id']=df.sample_id.to_numpy()[nn]
    df['nearest_otherfold_kmer_label']=df.label.to_numpy()[nn]
    # This is NOT alignment identity and must not be used for identity-stratified claims.
    save('all_training_atlas.csv',df)
    save('errors_only.csv',df[df.fusion_error].sort_values(['fusion_category','fusion_margin'],ascending=[True,False]))
    save('fusion_changes.csv',df[df.fusion_corrects_sequence|df.fusion_harms_sequence])
    source_manifest=[dict(role=k,path=str(v),sha256=sha(v),bytes=v.stat().st_size) for k,v in paths.items()]
    save('input_manifest.csv',pd.DataFrame(source_manifest))
    summary=dict(status=P['status'],inputs=source_manifest,n=len(df),unique_groups=df.group_id.nunique(),
        group_crossfold_overlap=False,duplicate_sequences=int(df.sequence.duplicated().sum()),
        baseline=baseline,sequence=metrics(df.label,df.p_seq,P['sequence_reference_threshold']),
        structure=metrics(df.label,df.p_struct,P['structure_reference_threshold']),
        fusion_corrects_sequence=int(df.fusion_corrects_sequence.sum()),fusion_harms_sequence=int(df.fusion_harms_sequence.sum()),
        near_boundary_band=band,near_boundary_errors=int((df.near_boundary_10pct&df.fusion_error).sum()),
        seed_disagreement_errors=int((df.seed_reference_disagreement&df.fusion_error).sum()),
        max_cached_vs_archived_mean_plddt_difference=float(np.max(np.abs(df.cached_mean_plddt-df.mean_plddt))),
        external_test_inputs_read=False,new_model_trained=False,limitations=P['limitations'])
    for name in ['FP','FN','TP','TN']:
        block=df[df.fusion_category==name]
        summary[name]=dict(n=len(block),cys_ge4=int((block.cys_count>=4).sum()),
            mean_plddt=float(block.mean_plddt.mean()),mean_margin=float(block.fusion_margin.mean()),
            reference_seed_disagreement=int(block.seed_reference_disagreement.sum()),
            all_seed_reference_wrong=int(block.all_seed_reference_wrong.sum()))
    save('fold_metrics.csv',pd.DataFrame([dict(fold=f,**metrics(g.label,g.p_fusion,P['fusion_threshold'])) for f,g in df.groupby('fold')]))
    dump('audit_summary.json',summary)
    return df,summary

def matched(df):
    pairs=[]
    matchcols=['length','net_charge_proxy','mean_hydropathy','cys_count','aromatic_fraction']
    scale=np.array([3,2,.5,1,.1])
    for err,correct in [('FP','TN'),('FN','TP')]:
        used=set()
        for fold in range(5):
            cases=df[(df.fusion_category==err)&(df.fold==fold)]
            controls=df[(df.fusion_category==correct)&(df.fold==fold)]
            eligible={}
            for i,case in cases.iterrows():
                delta=np.abs(controls[matchcols].to_numpy()-case[matchcols].to_numpy(dtype=float))
                valid=np.all(delta<=scale+1e-12,axis=1)&(controls.group_id.to_numpy()!=case.group_id)
                eligible[i]=[(int(j),float(np.sum((dv/scale)**2))) for j,dv in zip(controls.index[valid],delta[valid])]
            for i in sorted(eligible,key=lambda ix:(len(eligible[ix]),int(df.source_index.iloc[ix]))):
                cg=df.group_id.iloc[i]
                if cg in used:continue
                options=[(j,dist) for j,dist in eligible[i] if df.group_id.iloc[j] not in used]
                if not options:continue
                j,dist=min(options,key=lambda x:(x[1],x[0]));used.update([cg,df.group_id.iloc[j]])
                pairs.append(dict(error_type=err,fold=fold,case_index=i,control_index=j,
                    case_id=df.sample_id.iloc[i],control_id=df.sample_id.iloc[j],
                    case_group=cg,control_group=df.group_id.iloc[j],distance=dist))
    pairs=pd.DataFrame(pairs)
    save('matched_pairs.csv',pairs)
    # Pair groups cannot repeat within an error-type comparison.
    for err,g in pairs.groupby('error_type'):
        ids=list(g.case_group)+list(g.control_group);assert len(ids)==len(set(ids))
    tests=[];balance=[];rng=np.random.default_rng(P['seed'])
    for err,g in pairs.groupby('error_type'):
        ci=g.case_index.to_numpy();ti=g.control_index.to_numpy()
        for col in matchcols:
            delta=df[col].to_numpy()[ci]-df[col].to_numpy()[ti]
            balance.append(dict(error_type=err,feature=col,n=len(delta),case_mean=float(df[col].to_numpy()[ci].mean()),control_mean=float(df[col].to_numpy()[ti].mean()),mean_difference=float(delta.mean())))
        for col in TEST_FEATURES:
            vals=df[col].to_numpy(dtype=float);diff=vals[ci]-vals[ti]
            valid=np.isfinite(diff);d=diff[valid];folds=g.fold.to_numpy()[valid]
            if len(d)<5:continue
            boots=d[rng.integers(0,len(d),(P['statistical_test']['bootstrap_replicates'],len(d)))].mean(1)
            perm=(d[None,:]*rng.choice([-1,1],(P['statistical_test']['sign_flip_replicates'],len(d)))).mean(1)
            p=(1+int((np.abs(perm)>=abs(d.mean())).sum()))/(len(perm)+1)
            entry=dict(error_type=err,feature=col,n=len(d),mean_pair_difference=float(d.mean()),
                ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),p=p,
                discovery_difference=float(d[np.isin(folds,[0,1,2])].mean()),
                replication_difference=float(d[np.isin(folds,[3,4])].mean()))
            fmeans=[float(d[folds==f].mean()) if np.any(folds==f) else np.nan for f in range(5)]
            for f,m in enumerate(fmeans):entry[f'fold{f}_difference']=m
            entry['folds_same_direction']=sum(np.sign(m)==np.sign(d.mean()) for m in fmeans if np.isfinite(m))
            entry['discovery_replication_same_direction']=bool(entry['discovery_difference']*entry['replication_difference']>0)
            tests.append(entry)
    tests=pd.DataFrame(tests);order=np.argsort(tests.p.to_numpy());pv=tests.p.to_numpy()[order]
    q=np.minimum.accumulate((pv*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1]
    tests['q_bh']=0.;tests.loc[order,'q_bh']=np.minimum(q,1.)
    save('matched_feature_tests.csv',tests.sort_values('q_bh'))
    save('matching_balance.csv',pd.DataFrame(balance))
    save('matching_coverage.csv',pd.DataFrame([dict(error_type=e,fold=f,total_errors=int(((df.fusion_category==e)&(df.fold==f)).sum()),matched=int(((pairs.error_type==e)&(pairs.fold==f)).sum())) for e in ['FP','FN'] for f in range(5)]))
    return pairs,tests

def report(df,s,pairs,tests):
    sig=tests[(tests.q_bh<.05)&(tests.folds_same_direction>=4)&tests.discovery_replication_same_direction]
    lines=['# GHXTox 首轮错误分析','',
        '这是历史训练 OOF 的探索性分析，不是新模型的独立验证。未读取 Test1/Test2，未修改模型、校准、融合或阈值。','',
        '## 核心结果','',
        f"- 共 {s['n']} 条训练记录，{s['unique_groups']} 个序列组；现有组标签不跨折。",
        f"- 融合错分 {s['baseline']['errors']} 条：假阳性 {s['baseline']['FP']}，假阴性 {s['baseline']['FN']}。",
        f"- 相对参考阈值下的序列分支，融合纠正 {s['fusion_corrects_sequence']} 条，同时引入 {s['fusion_harms_sequence']} 条错误。阈值不同，因此不是纯结构贡献或因果效应。",
        f"- 距阈值最近约 10% 的样本包含 {s['near_boundary_errors']} 条错误；种子参考分类分歧覆盖 {s['seed_disagreement_errors']} 条错误。",
        f"- 严格卡钳与不复用组的组成匹配得到 {len(pairs)} 对；未匹配错误仍保留在总表，不能忽略。",'',
        '## 配对小实验','',
        '同标签、同折、不同序列组，控制长度、近似净电荷、平均疏水性、Cys 数和芳香残基比例。每个组在一种错误比较内最多使用一次。置信区间来自配对 bootstrap，p 值来自配对符号置换；对 28 项比较统一做 BH 校正，仅作探索。来源、真实家族和实验条件缺失，不能作因果解释。','',
        '|错误类型|指标|错题减正确对照的差值|95%区间|BH q|同方向折数|',
        '|---|---|---:|---|---:|---:|']
    for _,r in tests.sort_values('q_bh').head(12).iterrows():
        lines.append(f"|{r.error_type}|{r.feature}|{r.mean_pair_difference:.4f}|[{r.ci_low:.4f}, {r.ci_high:.4f}]|{r.q_bh:.4g}|{int(r.folds_same_direction)}/5|")
    lines+=['',f'通过 q<0.05、至少四折同向且发现/内部复核方向一致的关联共有 {len(sig)} 项。不能把这些关联称为机制证据。','',
        '## 已知限制','']
    lines+=['- '+x for x in [
        '旧 OOF 的校准和融合参数使用了全体 OOF 标签；部分历史结构 checkpoint 的审计记录确认留出折参与 checkpoint 选择，当前预测流完整上游谱系未获认证。不能把后续在这些输出上交叉拟合的纠错分数当成严格泛化性能。',
        '当前错误地图使用能复现既有 fusion_refreeze 的 archived operating-point OOF；它与另一个 full2795 OOF 并非同一数值流。不能混用或声称已重现当前完整单结构推理。',
        '错误稳定性只对三个序列种子有诊断；没有三个独立完整融合模型，不能声称融合错误跨完整模型种子稳定。',
        '保留数据只给出通用 peptide|label 标识，原始数据库、实验终点、浓度和末端修饰均未知。未自动补机制标签，也未重标样本。',
        'k-mer 余弦仅为序列邻域代理，不是序列一致性；不得据此报告 <=40% 等一致性分层。',
        '几何指标基于预测 C-alpha 坐标，疏水斑块是距离图代理，不是实测溶剂暴露表面；理想螺旋疏水矩不是实际二级结构证据。',
        '缓存/归档 pLDDT 差异在 audit_summary 中单独记录；如果数值不同，不能把两者静默互换。',
        '发现折与内部复核折并非盲测；配对置换所需交换性也不由匹配自动保证。显著性只用于排序待验证假设。']]
    lines+=['','## 文件','',
        '- all_training_atlas.csv：全部样本、概率、错误属性、序列与空间特征。',
        '- errors_only.csv：错题清单。fusion_category 区分 FP/FN。',
        '- fusion_changes.csv：融合纠正和引入错误的逐样本记录。',
        '- matched_pairs.csv / matching_coverage.csv / matching_balance.csv：配对、覆盖率和匹配平衡。',
        '- matched_feature_tests.csv：全部探索性统计结果。',
        '- input_manifest.csv / audit_summary.json / protocol.json：输入哈希、审计及固定分析规则。',
        '- build_atlas.py：可复现分析脚本。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    df,s=build();print('Match error cases and run fixed association tests',flush=True)
    pairs,tests=matched(df);report(df,s,pairs,tests)
    print(json.dumps(dict(baseline=s['baseline'],pairs=len(pairs),top=tests.sort_values('q_bh').head(8).to_dict('records')),ensure_ascii=False),flush=True)
