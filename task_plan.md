# Task Plan: Quality-Aware 1D/3D Coupling Upgrade

## Goal
Upgrade the GHXTox 1D/3D fusion pipeline with stronger conformer inputs, quality-aware gating, anti-regression selection, nested cross-validation, and reproducible evaluation.

## Current Phase
Phase 23 complete

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
