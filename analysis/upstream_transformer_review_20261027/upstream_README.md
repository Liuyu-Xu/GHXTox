# GHXTox

GHXTox is a peptide toxicity prediction framework that combines a sequence
submodel with a structural submodel. The sequence submodel uses a
LoRA-adapted ProtT5 encoder and a bidirectional Mamba classifier. The
structural submodel integrates residue semantics, Cα-level geometry, and
full-atom chemical-site information with pLDDT-aware message weighting and
gating.

The primary model is the validated extra-Transformer removal variant. Its
structural submodel bypasses its extra two-layer sequence
Transformer (`num_sequence_layers=0`). The three residue EGNN layers,
chemical-site message passing, cross-attention, and pLDDT gates remain.
ProtT5 and ESM2 retain their pretrained architectures.

All default training, structural prediction, evaluation, and public fusion
entrypoints select this variant.

The default model combines the two independently estimated toxicity
probabilities using fixed weights derived from out-of-fold (OOF) predictions:

```text
p(GHXTox) = 0.76 p(sequence) + 0.24 p(structure)
```

The binary decision threshold is `0.5063534555`. Default structural inference
uses one ESMFold structure per peptide. The exploratory multi-conformer
residual extension is not enabled during default inference.

The matching structural checkpoint is
`runs/default_no_inner_transformer_20261023/best_model.pt`: a seed42 refit on
all 6,387 training records for six fixed epochs (the median of the five
validation-selected epochs). Fusion parameters were fitted to aligned
training OOF predictions for deployment. This refit and parameter selection
are not a new independent performance evaluation.

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

## Data and checkpoints

Datasets, processed structures, and model checkpoints are not distributed in
the source repository. After obtaining authorized copies, place them under the
following paths:

| Dataset | File | Records | Role |
| --- | --- | ---: | --- |
| Training set | `dataset/training_set.fasta` | 6,387 | Model development and OOF parameter selection |
| Test set 1 | `dataset/test_set_1.fasta` | 1,126 | Fixed retrospective benchmark |
| Test set 2 | `dataset/test_set_2.fasta` | 582 | Fixed retrospective benchmark |

Test set 1 and Test set 2 are benchmark sets retained from preceding studies;
they are not newly collected external cohorts. Test set 2 is an exact
sequence-and-label subset of Test set 1. FASTA headers follow the format
`>sample_id|label`, where `1` denotes a toxic peptide and `0` denotes a
non-toxic peptide.

Processed feature files belong under `data/`, and checkpoints belong under
`runs/`, using the paths declared by the public configuration files.

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
Use structural probabilities generated with the current checkpoint; CSV
columns alone cannot establish which architecture produced them.

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

To reproduce the final structural refit, use the full processed training cache
with `configs/default.json` and `--full-data-epochs 6`. Without that flag, the
training command uses its usual validation and early-stopping procedure.

## Exploratory multi-conformer analysis

The multi-conformer residual reported in the article is an exploratory
analysis and is not part of default inference. Its fixed protocol, cohort
manifests, configuration, evaluation script, and reported values are provided
in [`research/multiconformer/`](research/multiconformer/README.md).

## Repository structure

- `src/ghxtox/`: model, preprocessing, training, and inference code
- `configs/`: default model configurations
- `dataset/`: local-only labeled peptide sequences used in the study
- `tests/`: tests for the public inference interface
- `research/multiconformer/`: reproducibility materials for the exploratory residual

## Citation

Please cite the associated GHXTox article when using this code or the provided
datasets. Full citation information will be added after publication.

