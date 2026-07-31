# Task Plan: Quality-Aware 1D/3D Coupling Upgrade

## Goal
Upgrade the GHXTox 1D/3D fusion pipeline with stronger conformer inputs, quality-aware gating, anti-regression selection, nested cross-validation, and reproducible evaluation.

## Current Phase
Phase 26 in progress

## Phases

### Phase 1: Audit Inputs and Existing Coupler
- [x] Inspect current coupling and stacking implementations
- [x] Audit ProtT5, 3D-v2, conformer, and full-model OOF prediction schemas
- [x] Confirm sample alignment, fold integrity, and available quality features
- **Status:** complete

### Phase 2: Design Stronger Coupling Protocol
- [x] Define conformer-aware quality features
- [x] Define nested hyperparameter selection without held-out leakage
- [x] Define anti-regression acceptance criteria against the 1D anchor and full baseline
- [x] Identify a residue/conformer-level interaction that adds information beyond rejected sample-level gates
- **Status:** complete

### Phase 3: Implement
- [x] Extend the retained residue interaction adapter with conformer residue/global inputs
- [x] Preserve existing nested selection and retention gates
- [x] Preserve frozen checkpoint application and exact missing-structure fallback
- [x] Add CLI attachment/training controls and reproducibility metadata
- [x] Build the combined 2,795-sample conformer processed artifact and OOF adapter cache
- **Status:** complete

### Phase 4: Tests and OOF Screening
- [x] Add focused unit tests
- [x] Run syntax and targeted test suite
- [x] Run the upgraded OOF coupling experiment
- [x] Compare against 1D, previous dynamic coupling, and full default
- **Status:** complete

### Phase 5: Documentation and Handoff
- [x] Document commands and fair reporting rules in README
- [x] Review changed files and generated artifacts
- [x] Summarize whether the upgraded coupler qualifies as the main model
- **Status:** complete

### Phase 6: Implement Required Training Controls
- [x] Add threshold-neighborhood asymmetric margin loss
- [x] Add source/length-aware low-rank conformer residual
- [x] Add fold-local class/source/length balanced sampling
- [x] Preserve exact missing-conformer and frozen-base fallback
- **Status:** complete

### Phase 7: Fixed OOF Ablation Matrix
- [x] Run baseline-compatible control
- [x] Run margin-loss-only candidate
- [x] Run source/length-residual-only candidate
- [x] Run combined candidate with balanced sampling
- [x] Apply unchanged retention gates without Test1/Test2
- **Status:** complete

### Phase 8: Tests and Documentation
- [x] Add focused unit tests for all new controls
- [x] Run Ruff, regression tests, and Git whitespace checks
- [x] Document commands, results, and final decision in README
- **Status:** complete

### Phase 9: Import and Audit 31-50 aa Pilot
- [x] Copy the server result archive and checksum into the project
- [x] Verify SHA256, sample alignment, class balance, and conformer counts
- [x] Extract only into the dedicated 31-50 data/run namespaces
- **Status:** complete

### Phase 10: Fixed 31-50 aa Pilot OOF Evaluation
- [x] Add an independent reproducible evaluation script
- [x] Build the 400-sample augmented/subset artifacts and group-aware folds
- [x] Train five OOF residual checkpoints
- [x] Compare covered-cohort and full-training OOF metrics with paired statistics
- [x] Apply the prespecified retention gate before requesting full generation
- **Status:** complete

### Phase 11: Regression Checks and Documentation
- [x] Verify outputs, scripts, and no active training processes
- [x] Run focused regression and whitespace checks
- [x] Document pilot results and the full-generation decision in README
- **Status:** complete

### Phase 12: Audit Latest Suiyumanbu/GHXTox
- [x] Fetch the latest upstream main without modifying the dirty worktree
- [x] Compare commits, files, model code, tests, configs, and documented results
- [x] Identify improvements absent locally and conflicts with local 31-50 work
- **Status:** complete

### Phase 13: Integrate Upstream Improvements
- [x] Import the selected upstream code and reproducibility artifacts
- [x] Preserve local coupling controls, 31-50 pilot, and exact fallback behavior
- [x] Resolve README conflicts without losing user outputs
- **Status:** complete

### Phase 14: Verify and Document
- [x] Run focused and regression tests for merged model paths
- [x] Run Ruff, compile, CLI parsing, and Git whitespace checks
- [x] Document the merged status and final model decision
- **Status:** complete

### Phase 15: Design Conservative Bidirectional Coupling
- [x] Audit existing sample-level and residue-level 1D/3D coupling
- [x] Define a bidirectional branch that preserves the current default and exact fallback
- [x] Fix the ablation protocol before any new OOF result is viewed
- **Status:** complete

### Phase 16: Implement and Test
- [x] Add reverse 3D-to-1D attention and agreement-gated local interaction
- [x] Add an independent zero-initialized bounded residual and frozen-base training mode
- [x] Add diagnostics, checkpoint compatibility, CLI controls, and focused tests
- **Status:** complete

### Phase 17: Reproducibility and Decision Boundary
- [x] Add a resumable fixed OOF screening script
- [x] Document architecture, run command, retention gate, and non-promotion status
- [x] Run lint and regression checks without launching server-scale training
- **Status:** complete

### Phase 18: Integrate Bidirectional OOF Evidence
- [x] Import and verify the AutoDL result archive
- [x] Apply the frozen retention gate without Test1/Test2
- [x] Document metrics, diagnostics, and the stopping decision
- [x] Keep compact evidence while excluding fold checkpoints from Git
- **Status:** complete

### Phase 19: Length and Generator-Source OOF Diagnostics
- [x] Audit the frozen OOF inputs and existing cohort-analysis utilities
- [x] Report 1D, 3D, and retained joint OOF metrics for 11-15, 16-30, and 31-50 aa
- [x] Quantify source/length predictability, label balance, quality/divergence, and error types
- [x] Emit machine-readable JSON/CSV without reading Test1/Test2 labels
- **Decision:** generator source and length are perfectly confounded in the
  existing 3,195-row conformer coverage (Cramér's V 1.0); no threshold, weight,
  or default-model change is authorized.
- **Status:** complete

### Phase 20: Reproducible 200-Sample BioEmu Pilot Preparation
- [x] Build a deterministic train-only label/length-stratified 200-sample manifest
- [x] Add reproducible BioEmu environment and server launcher scripts
- [x] Record version, model, dependency, seed, filtering, resume, and storage protocol
- [x] Add smoke-only feasibility checks and a machine-readable decision schema
- **Status:** complete

### Phase 20A: Execute BioEmu on Linux GPU
- [x] Upload and verify the frozen server-input archive on AutoDL/Linux CUDA
- [x] Run the six-sample smoke and collect the machine-readable gate decision
- [x] Diagnose the initial infrastructure-only smoke failure and rerun the exact
  frozen smoke with cached BioEmu/AF2 weights
- [x] Run the authorized 200-sample pilot after every smoke gate passed
- [x] Verify and import the result archive and freeze the continue/switch decision
- **Decision:** the 200-row BioEmu pilot is feasible (198/200 strict usable,
  5,683 valid conformers, 3.062 GPU hours), but the protocol still does not
  authorize generation of all 6,387 rows.
- **Status:** complete

### Phase 21: Verification and README Handoff
- [x] Add focused tests for diagnostics, sampling, filtering, local cached-model
  execution, and infrastructure-failure decision logic
- [x] Run final lint, regression, archive-integrity, CLI, and whitespace checks
- [x] Document manifest counts, smoke protocol, feasibility criteria, and next decision
- **Status:** complete

### Phase 22: BioEmu Cross-Length Descriptor Evaluation
- [x] Convert the 200 imported BioEmu trajectories into aligned, reusable
  ensemble-summary descriptors without changing the frozen sample cohort
- [x] Audit same-sample overlap with PepFlow/AF2 sources and compare descriptor
  shifts under one unified generator
- [x] Measure cross-fitted length/source predictability and quality/divergence
  without reading Test1/Test2 labels
- [x] Emit compact JSON/CSV evidence and freeze the expand/stop decision
- **Decision:** retain the verified 200-row pilot, but hold full BioEmu
  generation. BioEmu descriptors contain significant train-only toxicity
  signal, while the same-sample source fingerprint is strong and fixed
  descriptor decisions are substantially weaker than the 1D OOF anchor.
- **Status:** complete

### Phase 23: Frozen 200-Row Bounded BioEmu Fusion Probe
- [x] Implement one leakage-aware candidate using the existing canonical folds,
  frozen 1D OOF probabilities, and exactly four BioEmu ensemble descriptors
- [x] Cross-fit a standardized L2 logistic structure score using only the other
  four folds; do not train a fusion/meta-model on 1D OOF probabilities
- [x] Apply the preregistered formula
  `logit(p_final)=logit(p_1d)+0.25*(valid_conformers/32)*clip(logit(p_3d),-1,1)`
  with the unchanged 1D decision threshold 0.45
- [x] Report all 200 rows as primary and the two strict-quality exceptions only
  as a prespecified sensitivity analysis
- [x] Freeze the retain/stop decision without reading Test1/Test2 labels
- **Fixed structure model:** `StandardScaler` plus L2 `LogisticRegression`,
  `C=1`, `class_weight=balanced`, `solver=liblinear`, seed `20260731`; no search
- **Retention gate:** candidate MCC strictly exceeds 1D; AUPRC is no more than
  0.002 lower; Brier is no more than 0.002 higher; ECE-10 is no more than 0.005
  higher; corrected decisions exceed newly incorrect decisions; and at least
  three of five fold MCC values are non-decreasing. All conditions must pass.
- **Decision:** `stop_bounded_bioemu_candidate`. The candidate slightly improves
  ranking/calibration but lowers MCC from 0.727161 to 0.718125 and changes one
  correct 1D decision into an error. Keep the frozen 1D anchor and do not expand
  BioEmu generation.
- **Status:** complete

### Phase 24: Standalone Source-Robust Multi-Conformer 3D Encoder
- [x] Audit the existing 3D tensor/cache schemas and reusable model/training code
- [x] Build one fold-safe dataset joining the 3,195 legacy conformer samples,
  200 BioEmu samples, and 143 same-sample cross-generator pairs
- [x] Freeze a lightweight variable-length contact-graph encoder and paired
  representation-consistency objective before viewing OOF metrics
- [x] Run smoke tests followed by canonical five-fold 3D-only OOF training
- [x] Export standalone 3D OOF predictions, embeddings, quality scores, missing
  masks, fold checkpoints, and a handoff manifest for the fusion teammates
- [x] Compare against the existing 3D-v2/multi-conformer branch and freeze the
  retain/stop decision without reading Test1/Test2 labels
- **Boundary:** this phase produces a standalone 3D deliverable only; it does
  not search 1D/2D/3D fusion weights and does not authorize full BioEmu generation.
- **Frozen input:** 21-channel residue identity, eight existing bounded ensemble
  residue features, normalized position/length, twelve global ensemble features,
  and six pair channels (distance mean/std plus 6/8/10 A occupancy and 8 A
  occupancy uncertainty). Maximum sequence length is 50.
- **Frozen encoder:** hidden/embedding dimension 32, two edge-gated message
  passing layers, mean/max pooling, dropout 0.10, and one standalone toxicity
  logit. Paired source views share every parameter.
- **Frozen optimization:** seed 20260731, 15 epochs, batch size 64, AdamW learning
  rate `3e-4`, weight decay `1e-4`, training-fold positive class weight, BCE on
  each available view, and cosine consistency weight 0.10 for the 143 paired
  legacy/BioEmu samples. No early stopping or hyperparameter search.
- **Primary threshold:** 0.50 for the new standalone branch. Existing 3D-v2 is
  compared at its already frozen threshold 0.677819 on the identical 3,252 rows.
- **Replacement gate:** new MCC strictly exceeds matched 3D-v2 MCC 0.764328,
  AUPRC is at least 0.885914, and at least three of five folds improve MCC.
- **Fusion-handoff gate:** new MCC is at least 0.70, AUPRC at least 0.85,
  cross-generator source-classifier BACC at most 0.65, paired embedding cosine
  at least 0.90, and paired mean absolute probability difference at most 0.10.
  This gate may authorize an experimental 3D handoff but never a default-model
  replacement or full BioEmu generation.
- **Decision:** `stop_multiconformer_graph_candidate`. Candidate MCC/AUPRC are
  0.594800/0.818271 versus matched 3D-v2 0.764328/0.887914, with zero fold MCC
  wins. Paired cosine and probability-delta gates pass, but the source probe
  remains high at BACC 0.814685 and the predictive handoff gate fails.
- **Status:** complete

### Phase 25: Retained 3D Team Handoff Package
- [x] Join the retained capability-aware 3D OOF stream to canonical folds and
  the train-only conformer/source diagnostics without changing probabilities
- [x] Export one 6,387-row prediction contract with fixed threshold, structure
  availability, generator source, quality fields, and exact missing mask
- [x] Export compact schema/provenance/checksum metadata for the fusion teammates
- [x] Verify alignment, labels, folds, probabilities, and no Test1/Test2 access
- **Boundary:** package only the already retained 3D model; do not recalibrate,
  retune thresholds, or mix any 1D/2D prediction into this artifact.
- **Decision:** deliver the retained capability-aware 3D OOF stream to the
  fusion team: 6,387/6,387 rows, fixed MCC 0.839018/AUPRC 0.928171, 3,195
  conformer-covered rows and 3,192 explicit missing-conformer fallbacks.
- **Status:** complete

### Phase 26: Preregistered 3D-v2 Multi-Seed Stability and Ensemble
- [x] **P0 — Reproducibility audit:** identify the exact retained 3D-v2 plus
  capability-aware conformer-residual training entrypoints, original seed,
  fold checkpoints, data hashes, environment, and expected GPU cost
- [ ] **P0 — Reference reproduction gate:** rerun the original seed first and
  require aligned 6,387-row OOF predictions and retained metrics within the
  documented serialization/training tolerance before launching more seeds
- [x] **P1 — Freeze the experiment:** before viewing any new-seed outcome,
  record exactly three seeds `42/123/2025`, unchanged canonical
  folds, architecture, preprocessing, training schedule, fallback behavior,
  threshold `0.677819`, and equal-probability ensemble rule
- [ ] **P1 — Execute three complete OOF runs:** produce five held-fold
  predictions for each seed; every seed must run the same full retained 3D
  path, including the capability-aware conformer residual rather than only the
  base 3D-v2 branch
- [ ] **P2 — Robustness report:** report each seed, mean plus standard deviation,
  fixed equal-weight ensemble, all five folds, three length strata, generator
  cohorts, calibration, decision disagreements, and runtime/storage
- [ ] **P2 — Frozen promotion gate:** promote only the equal-weight ensemble if
  aggregate MCC exceeds `0.839018`, AUPRC is at least `0.928171`, at least three
  of five fold MCC values do not decrease, and Brier/ECE plus every length
  stratum show no material regression; never promote the best single seed
- [ ] **P3 — Handoff regeneration:** if and only if the gate passes, rebuild the
  6,387-row 3D team handoff with new checksums; otherwise retain Phase 25 exactly
- **Boundary:** Test1/Test2 remain sealed. No seed, threshold, epoch count,
  architecture, loss, or ensemble weight may be chosen after viewing the new
  OOF outcomes.
- **Reference reproduction tolerance:** exact 6,387-row ID/fold/label/sequence
  alignment, probability Pearson correlation at least `0.995`, absolute MCC
  difference at most `0.003`, absolute AUPRC difference at most `0.001`, and
  absolute Brier difference at most `0.001` versus the retained seed-42 OOF.
- **Frozen no-material-regression tolerances:** ensemble Brier may increase by
  at most `0.001`, ECE-10 by at most `0.002`, and each length stratum may lose
  at most `0.005` MCC and `0.002` AUPRC versus the retained handoff. These are
  evaluated in addition to the stricter aggregate/fold promotion conditions.
- **Status:** in_progress

## Key Questions
1. Can the conformer residual OOF stream be aligned with all 6,387 training records?
2. Which available columns provide reliable per-sample structure quality signals?
3. Does nested selection improve MCC/AUPRC without degrading Brier/ECE?
4. Does the upgraded coupler exceed the full default OOF MCC of approximately 0.8712?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use cross-fitted/nested OOF metrics for model claims | Frozen fits trained on all OOF rows are not unbiased estimates |
| Keep the 1D ProtT5 stream as the anchor | It is robust and provides a safe fallback when structure quality is weak |
| Preserve unrelated dirty-worktree changes | They are existing user outputs and outside this code upgrade |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| PowerShell profile execution-policy warning | 1 | Treat as harmless because commands still execute successfully |
| Phase 26 wrapper parsed `$LASTEXITCODE:` as a drive-qualified variable | 1 | Delimit it as `${LASTEXITCODE}` and rerun the parser audit |
| Direct `.ps1` launch was blocked by Windows ExecutionPolicy | 1 | Use a one-process `powershell.exe -NoProfile -ExecutionPolicy Bypass -File` launcher; do not change machine policy |
| Historical fold inspection queried nonexistent `summary.json` files | 1 | Use the existing `config.json`/`history.json` artifacts and their timestamps instead |
| Phase 26 prepare reported AUPRC 0.927682 instead of retained 0.928171 | 1 | Preserve CSV probabilities as float64 to avoid ranking ties, then regenerate the frozen protocol |
| PowerShell audit command had an empty pipeline element | 1 | Collect objects in a variable before piping to `Format-List` |
| Repeated `foreach` followed by pipeline parse failure | 2 | Ban that shell pattern for this task; emit collected arrays only |
| Planning-file multi-hunk patch did not match | 1 | Re-read the files and apply smaller exact-context edits |
| PowerShell `$Matches` automatic variable collision | 1 | Use a task-specific collection variable name |
| Parallel audit aborted when `rg` returned no matches | 1 | Use `Get-ChildItem` for optional artifact discovery |
| Target PyTorch environment lacks `pytest` | 1 | Install the lightweight test runner, then run focused tests |
| OOF precompute exceeded the short command timeout | 1 | Launch it as a hidden background process with log files and monitor |
| Cache verification assumed nonexistent `token_values` field | 1 | Inspect the payload keys and validate the actual `token_embeddings` schema |
| OOF aggregation test compared serialized float32 values exactly | 1 | Use a numerical tolerance appropriate for CSV float serialization |
| Ruff found formatting drift and an unused import | 1 | Remove the import, run Ruff format/check, and rerun tests |
| Frozen base parameters still allowed training-mode Dropout | 1 | Force non-conformer modules to evaluation mode and rerun the fixed three-seed protocol |
| Three-seed foreground wrapper exceeded 240 seconds | 1 | Preserve completed outputs and rerun only the missing seed in the background |
| Target PyTorch environment has no `ruff` module | 1 | Use the standalone/default-environment Ruff command for linting |
| Zero-initialization test required exact probability round-trip equality | 1 | Assert exact zero residual and tolerance-based probability equality |
| Old checkpoint loader rejected the new zero-initialized source/length head | 1 | Add only the new conformer-head prefix to the allowed missing-key list and cover it in the initialization test |
| Upstream geometry module contained an unused `partial` import | 1 | Remove it and rerun Ruff |
| Full test suite lacked RDKit and used a Qt plotting backend | 1 | Run all non-RDKit tests with `MPLBACKEND=Agg`; 114 passed and 5 dependency-bound tests were deselected |
| PowerShell `Get-Content` displayed the UTF-8 README appendix as mojibake | 1 | Use explicit UTF-8 decoding for all subsequent README reads |
| Targeted pytest collection could not import `ghxtox` from the src layout | 1 | Rerun with the project `src` directory explicitly added to `PYTHONPATH` |
| Default Python lacks PyTorch and BioEmu pilot imported a nonexistent topology module | 1 | Use the documented PyTorch environment and import the existing topology helper from its actual module |
| Exact-margin quota backtracking allowed one fold option to overdraw a remaining margin | 1 | Check per-option length/label multiplicities before recursion |
| No local Linux runtime is available for the Linux-only BioEmu package | 1 | Finished and verified the AutoDL input package; real smoke awaits a Linux CUDA host |
| Python 3.10 validation environment lacks standard-library `tomllib` | 1 | Parse `pyproject.toml` with the default Python 3.13 interpreter instead |

## Required Upgrade Matrix
| Variant | Margin loss | Source/length low rank | Balanced sampler |
|---------|-------------|------------------------|------------------|
| Control | no | no | no |
| Margin | yes | no | no |
| Source-length | no | yes | no |
| Combined | yes | yes | yes |

The matrix is fixed before OOF execution. Test1 and Test2 remain untouched.

### Fixed Matrix Parameters
- Common: seed 42, 40 epochs, batch size 64, learning rate `3e-4`,
  weight decay `5e-4`, patience 7, threshold 0.60, structure dropout 0,
  conformer residual L1 0.01, frozen reproduced chemical-adapter fold checkpoint.
- Margin candidate: weight 0.10, logit margin 0.15, positive multiplier 1.5.
- Source-length candidate: rank 8.
- Combined candidate: the same margin and rank settings plus fold-local balanced
  sampling with maximum weight ratio 5.
- No value may be altered after viewing OOF outcomes.

## Notes
- Test1 and test2 remain untouched external evaluations until architecture and thresholds are frozen.
- Do not report all-OOF frozen-fit metrics as unbiased performance.
