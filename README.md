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
