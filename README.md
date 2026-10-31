# GHXTox

GHXTox is a peptide toxicity prediction framework that combines a sequence
submodel with a structural submodel. The sequence submodel uses a
LoRA-adapted ProtT5 encoder and a bidirectional Mamba classifier. The
structural submodel integrates residue semantics, Cα-level geometry, and
full-atom chemical-site information with pLDDT-aware message weighting and
gating.

The default model combines the two independently estimated toxicity
probabilities using fixed weights derived from out-of-fold (OOF) predictions:

```text
p(GHXTox) = 0.72 p(sequence) + 0.28 p(structure)
```

The binary decision threshold is `0.4702225653`. Default structural inference
uses one ESMFold structure per peptide. The exploratory multi-conformer
residual extension is not enabled during default inference.

## Installation

GHXTox requires Python 3.9 or later.

```bash
git clone https://github.com/Liuyu-Xu/GHXTox.git
cd GHXTox
python -m pip install -e .
```

Optional dependencies can be installed for evaluation, protein language
models, ESMFold structure generation, and atom-level graph construction:

```bash
python -m pip install -e ".[metrics,plm,esmfold,atom-graph]"
```

## Data

The repository uses the following names consistently:

| Dataset | File | Records | Role |
| --- | --- | ---: | --- |
| Training set | `dataset/training_set.fasta` | 6,387 | Model development and OOF parameter selection |
| Test set 1 | `dataset/test_set_1.fasta` | 1,126 | Fixed retrospective benchmark |
| Test set 2 | `dataset/test_set_2.fasta` | 582 | Fixed retrospective benchmark |

Test set 1 and Test set 2 are benchmark sets retained from preceding studies;
they are not newly collected external cohorts. FASTA headers follow the format
`>sample_id|label`, where `1` denotes a toxic peptide and `0` denotes a
non-toxic peptide.

## Inference

The default command accepts processed structural features and sequence records:

```bash
ghxtox-predict predict \
  --processed data/processed/test_set_1.pt \
  --sequence-records data/processed/test_set_1_sequence_records.pt \
  --output-csv results/test_set_1_predictions.csv \
  --output-metrics results/test_set_1_metrics.json \
  --device cuda
```

The command uses the checkpoint paths and OOF calibration files specified in
`configs/default_multidimensional_fusion.json`. Model checkpoints and processed
structure files are large binary assets and are not stored in the source tree.

If calibrated sequence and structural probabilities have already been
generated, the fixed fusion can be applied without loading either neural
network:

```bash
ghxtox-predict assemble \
  --cal-1d sequence_predictions.csv \
  --structure-csv structure_predictions.csv \
  --output-csv ghxtox_predictions.csv \
  --output-metrics ghxtox_metrics.json
```

The sequence prediction file must contain `sample_id`, `label`, and
`toxicity_probability`. The structural prediction file must contain
`sample_id`, `label`, and `structure_probability`. Sample identifiers, labels,
and sequences, when provided, must be aligned between the two files.

## Training

The structural submodel can be trained from a processed dataset as follows:

```bash
ghxtox-train \
  --train data/processed/training_set.pt \
  --config configs/default.json \
  --output-dir results/structural_submodel \
  --device cuda
```

Sequence-submodel training options are available through:

```bash
python -m ghxtox.lora_mamba_1d --help
```

## Exploratory multi-conformer analysis

The multi-conformer residual reported in the article is an exploratory
analysis and is not part of default inference. Its fixed protocol, cohort
manifests, configuration, evaluation script, and reported values are provided
in [`research/multiconformer/`](research/multiconformer/README.md).

## Repository structure

- `src/ghxtox/`: model, preprocessing, training, and inference code
- `configs/`: default model configurations
- `dataset/`: labeled peptide sequences used in the study
- `tests/`: tests for the public inference interface
- `research/multiconformer/`: reproducibility materials for the exploratory residual

## Citation

Please cite the associated GHXTox article when using this code or the provided
datasets. Full citation information will be added after publication.


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

### 2026-10-30：Cys、对比学习与残基排列进展

- B/C/D 三种子五折条件融合 MCC 为 0.8614/0.8680/0.8743；历史模型 0.8733，D 的差异区间跨零，不能证明提升。历史与新模型训练不完全匹配。详见 [报告](analysis/official_bcd_oof_20261030/REPORT.md)。
- 融合权重探索中 D 五折都选择原一维权重 0.72，指标不变；结果仅有汇总，服务端脚本尚未独立审计。详见 [审计](analysis/fusion_weight_crossfit_20261030/AUDIT.md)。
- 两路共同错误：低 Cys 有毒 107/262，高 Cys 无毒 37/240；C→D 共纠正 64 条、新错 48 条。见 [报告](analysis/expert_cys_cd_family_20261030/REPORT.md)。
- 共同错误的排列配对及轻量探针仅提供探索线索；探针对原完整模型纠正 90 条、新错 448 条，尚不适合直接融合。见 [配对分析](analysis/both_wrong_order_matched_20261030/REPORT.md) 和 [探针对照](analysis/residue_order_probe_20261030/GHX_COMPARISON.md)。
- 下一步：先审计原三维分支已有残基、化学位点和局部交互特征，再设计残基信息补充的匹配消融；同折同种子比较一维、三维和完整融合，关注两类 Cys 错误与整体 MCC/AUPRC。既有 OOF 校准非完全嵌套，历史测试集已参与开发，均不能视为盲测。


### 2026-10-30：三维残基与本地训练进展

- 历史三维五折候选权重已核验，第2折化学位点残差为零；上游权重谱系不完整，相关结果为探索性。残基位置探针、化学位点遮蔽及匹配对照见 [三维残基审计](analysis/3d_residue_audit_20261030/REPORT.md) 和 [输入及配对分析](analysis/3d_residue_audit_20261030/INPUT_AND_MATCHED_REVIEW.md)。
- 局部及空间统计特征的轻量增量对照没有显示稳定收益；本地单折冻结主干的普通残基/Cys残差头没有改善重点群体。减弱惩罚虽增大修正但验证错误增加；用基础模型未见样本训练修正头的探索亦未改善分类。详见 [本地训练报告](analysis/local_residue_interaction_20261030/REPORT.md)。
- 这些试验均非最终完整 GHXTox 模型对比；上游OOF/历史测试已有开发使用、单折单种子及标签不确定性限制结论。下一步研究残基级三维交互，设置匹配训练与等参数对照，并评价三维与完整融合。


### 2026-10-31：近期研究结论与下一步

|研究|当前结果|结论与限制|
|---|---|---|
|Cys加权/分专家/监督对比|特定加权和专家方案未稳健改善；D三种子历史条件融合MCC0.8743，历史参考0.8733|同环境A三种子对照尚未补齐，不能称稳定提升|
|三维残基输入与敏感性|原三维已有残基化学、位置、近邻；历史候选fold2化学残差为零|配置/可解码特征不代表毒性机制，历史上游谱系不全|
|内部残基交互|冻结主干两折试验分类未提高；普通校准更好降低概率误差|不支持据此晋级模块，非完整1D+3D比较|
|部分联合训练|fold3普通微调MCC0.8068，普通交互0.8102，Cys交互0.8072|新增模块相对普通微调仅净少错1，低Cys漏判未改善，单fold探索|
|组成纠偏小试|普通小头MCC0.7764；难例加权0.6931；组成组合损失0.6634|低Cys漏判略少、反向无毒误报增加；非完整网络，停止扩大当前配置|

研究结果与入口：

- [三维输入、探针和遮蔽审计](analysis/3d_residue_audit_20261030/REPORT.md)
- [冻结头、内部消息、联合训练与Platt对照](analysis/local_residue_interaction_20261030/REPORT.md)
- [组成规律纠偏初筛](analysis/composition_debias_pilot_20261031/REPORT.md)
- [10月研究复盘与策略；含重复对比学习建议的纠正](analysis/research_strategy_20261031/REPORT.md)

现阶段优先补足已有候选的同条件比较，并寻找终点、分子形式明确的新辅助监督；不重复已做的Cys组内对比、严格配对可行性统计和同数据上的权重搜索。原主数据集和已发布模型不因探索结果自动改动。历史Test1/Test2均已参与开发，Test2是Test1子集；不作为两次独立盲测。多个近期结果来自冻结表示、单折或小样本，不能直接与原完整三种子融合模型排名，也不能据此否定所有同类方法。

### 研究资料归档说明

本次归档研究报告、分析代码、JSON/CSV结果表、训练包文本；近期src/configs/scripts/tests保存于[研究源码快照](research_snapshot_20261031/)，未替换默认应用入口。快照脚本有本地依赖和路径，需按报告配置环境和原始输入，不能保证单独clone即可复现。

[上传清单](research_snapshot_20261031/upload_manifest.json)记录复制数量、未上传的二进制或大文件及排除目录。模型权重、PT/NPZ缓存、压缩包、单文件≥5MB、第三方_vendor/依赖目录、sources/handoff快照未纳入本次文本归档；这不是全部本地文件的灾备副本。原始权重与大型数据仍在本地GHXTox及GHXTox-ziliao。
