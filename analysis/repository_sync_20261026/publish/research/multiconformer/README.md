# Exploratory multi-conformer residual

This directory documents the exploratory residual analysis reported in the
GHXTox article. It is separate from the canonical model: default inference
uses one ESMFold structure per peptide and the single-structure base logit.

## Protocol

The analysis used eight conformers for each covered peptide. PepFlow supplied
ensembles for 761 training peptides of 11–15 residues. AlphaFold2 predictions
generated through ColabFold supplied ensembles for 2,034 training peptides of
16–30 residues. The latter protocol generated 20 candidates from five models
and four seeds, then retained eight diverse structures after applying a mean
pLDDT threshold of 50.

Each ensemble was summarized by eight residue-level features and twelve
peptide-level features. A three-category vector encoded the conformer source.
The single-structure structural submodel remained frozen; only the residual
encoders, source encoder, residual multilayer perceptron, and residual scale
were trained. An availability mask set the correction exactly to zero for
peptides without conformer data.

Five fixed sequence-group-aware folds were used. The residual comparison kept
the sequence probability, fusion weights (0.72 and 0.28), and fusion threshold
(0.4702225653) unchanged. Paired uncertainty analysis used 5,000
class-stratified bootstrap samples and 5,000 within-sample randomizations with
seed 20260808.

## Files

- `config.json`: residual-head architecture and training parameters.
- `manifests/`: fixed folds and the exact conformer-covered cohorts.
- `evaluate.py`: comparison of the residual-enabled and base-only outputs.
- `reported_results.json`: machine-readable values reported in the article.

Generated conformer arrays, processed tensors, and checkpoints are excluded
from Git because of their size. They must be generated locally or downloaded
from the project data archive. The following asset layout is assumed:

```text
data/processed/training_set_structural.pt
data/conformers/pepflow/<sample_id>/ensemble_summary.npz
data/conformers/colabfold/<sample_id>/ensemble_summary.npz
checkpoints/structural_base_fold0.pt ... structural_base_fold4.pt
checkpoints/structural_residual.pt
```

The data-archive URL and checkpoint checksums must be added to the release
record before publication. Source code alone cannot reproduce the numerical
results without these generated assets.

Install the repository and the metric dependency before running the protocol:

```bash
pip install -e ".[metrics]"
```

## Generate the conformer caches

The supplied FASTA files define the exact peptide cohorts. PepFlow ensembles
were generated with eight samples per peptide and converted with an imputed
confidence of 0.7:

```bash
python -m ghxtox.pepflow_batch \
  --fasta research/multiconformer/manifests/training_pepflow.fasta \
  --pepflow-root /path/to/PepFlow \
  --full-model /path/to/full_model.pth \
  --output-root data/conformers/pepflow/raw \
  --num-samples 8 --chunk-size 4 --min-length 11 --max-length 15 \
  --sample-id-mode header

python -m ghxtox.conformer_cache \
  --fasta research/multiconformer/manifests/training_pepflow.fasta \
  --pdb-root data/conformers/pepflow/raw \
  --output-root data/conformers/pepflow \
  --source pepflow --confidence-mode imputed --imputed-confidence 0.7 \
  --max-conformers 8 --sample-id-mode header
```

The AlphaFold2/ColabFold protocol used shallow MSAs, five models, four seeds,
and seed 2026. It generated 20 candidates and retained eight diverse
conformers per peptide:

```bash
colabfold_batch \
  research/multiconformer/manifests/training_colabfold.fasta \
  data/conformers/colabfold/msa \
  --msa-only --msa-mode mmseqs2_uniref_env

colabfold_batch \
  data/conformers/colabfold/msa data/conformers/colabfold/results \
  --model-type alphafold2_ptm --num-models 5 --num-seeds 4 \
  --random-seed 2026 --max-seq 16 --max-extra-seq 32 \
  --num-recycle 3 --num-relax 0 --rank plddt --stop-at-score 100

python -m ghxtox.colabfold_ensemble_cache \
  --fasta research/multiconformer/manifests/training_colabfold.fasta \
  --results-root data/conformers/colabfold/results \
  --raw-root data/conformers/colabfold/raw \
  --cache-root data/conformers/colabfold \
  --max-conformers 8 --min-conformers 8 \
  --selection-mode diversity --min-mean-plddt 50
```

The same commands and parameters apply to the two `test_set_1_*.fasta`
cohorts. Test set 2 is an exact-sequence subset of Test set 1 and therefore
reuses the Test set 1 caches through verified sequence matching.

## Prepare the training tensors

The cohort manifests contain one row per peptide. Each cache directory must
contain an `ensemble_summary.npz` produced by `ghxtox.conformer_cache` or
`ghxtox.colabfold_ensemble_cache`.

```bash
python -m ghxtox.conformer_features \
  --processed data/processed/training_set_structural.pt \
  --conformer-manifest \
    research/multiconformer/manifests/training_pepflow.csv \
    research/multiconformer/manifests/training_colabfold.csv \
  --cache-root data/conformers/pepflow data/conformers/colabfold \
  --output data/processed/training_set_multiconformer.pt \
  --pilot-output data/processed/training_set_multiconformer_subset.pt \
  --pilot-fold-manifest data/processed/multiconformer_subset_folds.csv \
  --source-fold-manifest research/multiconformer/manifests/training_folds.csv
```

The resulting audit must report 2,795 covered records. The supplied
`training_subset_folds.csv` can be compared with the generated subset fold
file to verify sample order and fold membership.

## Train the residual head

Run each fold with the corresponding frozen base checkpoint:

```bash
for fold in 0 1 2 3 4; do
  ghxtox-train \
    --train data/processed/training_set_multiconformer_subset.pt \
    --fold-manifest research/multiconformer/manifests/training_subset_folds.csv \
    --fold "$fold" \
    --config research/multiconformer/config.json \
    --initial-checkpoint "checkpoints/structural_base_fold${fold}.pt" \
    --output-dir "runs/multiconformer_fold${fold}" \
    --device cuda
done
```

The final residual checkpoint was refitted for six epochs on the complete
covered subset, initialized from the frozen full-data structural checkpoint:

```bash
ghxtox-train \
  --train data/processed/training_set_multiconformer_subset.pt \
  --config research/multiconformer/config.json \
  --initial-checkpoint checkpoints/structural_base.pt \
  --output-dir runs/multiconformer_refit \
  --full-data-epochs 6 \
  --device cuda
```

## Evaluate the reported comparison

Attach the Test set 1 caches before evaluation:

```bash
python -m ghxtox.conformer_features \
  --processed data/processed/test_set_1_structural.pt \
  --conformer-manifest \
    research/multiconformer/manifests/test_set_1_pepflow.csv \
    research/multiconformer/manifests/test_set_1_colabfold.csv \
  --cache-root data/conformers/test1_pepflow data/conformers/test1_colabfold \
  --output data/processed/test_set_1_multiconformer.pt \
  --match-by-sequence
```

For Test set 2, replace the input and output paths, retain the Test set 1
manifests and caches, and add `--allow-unused-manifest-records`.

For a labeled processed test set carrying conformer features, compare the
active residual with the base logit from the same checkpoint:

```bash
python research/multiconformer/evaluate.py checkpoint \
  --processed data/processed/test_set_1_multiconformer.pt \
  --sequence-predictions results/test_set_1_sequence.csv \
  --checkpoint checkpoints/structural_residual.pt \
  --output-dir results/multiconformer_test_set_1 \
  --device cuda
```

Use the same command for Test set 2. The output contains aligned per-sample
probabilities, full-set metrics, the number of conformer-covered peptides, and
the number of changed binary decisions. For training OOF analysis, first
generate aligned base and residual structural OOF predictions, then run:

```bash
python research/multiconformer/evaluate.py predictions \
  --sequence-predictions results/training_sequence_oof.csv \
  --base-structure-predictions results/training_structure_base_oof.csv \
  --residual-structure-predictions results/training_structure_residual_oof.csv \
  --coverage-manifest research/multiconformer/manifests/training_subset_folds.csv \
  --output-dir results/multiconformer_training_oof \
  --bootstrap-iterations 5000 \
  --permutation-iterations 5000 \
  --seed 20260808
```

The analysis is retrospective. Its output supports an exploratory sensitivity
comparison and must not be described as independent external validation.
