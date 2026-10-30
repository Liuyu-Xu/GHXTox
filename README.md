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
