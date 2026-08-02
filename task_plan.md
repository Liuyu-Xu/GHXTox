# Task Plan: Quality-Aware 1D/3D Coupling Upgrade

## Goal
Upgrade the GHXTox 1D/3D fusion pipeline with stronger conformer inputs, quality-aware gating, anti-regression selection, nested cross-validation, and reproducible evaluation.

## Current Phase
Phase 34 in progress

## Phases

### Phase 34: Fusion Contract Audit, Documentation, and GitHub Handoff
- [x] Audit the Phase33 3D handoff and fusion contract for OOF leakage,
  row/fold alignment, threshold/calibration misuse, fallback behavior, and
  prohibited Test1/Test2 access
- [x] Add a concrete leakage-safe 1D/2D/3D fusion protocol and ablation/reporting
  requirements to the latest README
- [x] Add or strengthen machine-checkable validation where the audit finds a
  material gap
- [x] Run focused tests, lint/compile checks, handoff integrity checks, and Git
  whitespace/status review
- [ ] Commit only the intended engineering/documentation artifacts and push the
  current branch to the configured GitHub remote
- **Boundary:** do not fit fusion weights on the same OOF rows used for the
  headline claim; do not read Test1/Test2 labels; do not overwrite the retained
  Phase25 handoff while auditing the promoted Phase33 handoff.
- **Status:** in_progress

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
- [x] **P0 — Reference reproduction gate:** rerun the original seed first and
  require aligned 6,387-row OOF predictions and retained metrics within the
  documented serialization/training tolerance before launching more seeds
- [x] **P1 — Freeze the experiment:** before viewing any new-seed outcome,
  record exactly three seeds `42/123/2025`, unchanged canonical
  folds, architecture, preprocessing, training schedule, fallback behavior,
  threshold `0.677819`, and equal-probability ensemble rule
- [x] **P1 — Execute three complete OOF runs:** produce five held-fold
  predictions for each seed; every seed must run the same full retained 3D
  path, including the capability-aware conformer residual rather than only the
  base 3D-v2 branch
- [x] **P2 — Robustness report:** report each seed, mean plus standard deviation,
  fixed equal-weight ensemble, all five folds, three length strata, generator
  cohorts, calibration, decision disagreements, and runtime/storage
- [x] **P2 — Frozen promotion gate:** promote only the equal-weight ensemble if
  aggregate MCC exceeds `0.839018`, AUPRC is at least `0.928171`, at least three
  of five fold MCC values do not decrease, and Brier/ECE plus every length
  stratum show no material regression; never promote the best single seed
- [x] **P3 — Handoff regeneration:** if and only if the gate passes, rebuild the
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
- **Decision:** `retain_seed42_3d_handoff`. Seed 42 passed the reference
  reproduction gate, seeds 123 and 2025 completed the same full path on AutoDL,
  and the equal-weight ensemble failed the frozen promotion gate: MCC
  `0.835649` and AUPRC `0.927630` versus the retained reference
  `0.839018/0.928171`; Brier/ECE improved slightly, but aggregate MCC/AUPRC
  and the 11-15 aa length stratum regressed. The Phase 25 3D team handoff
  remains the retained artifact; no best single seed, threshold retuning, or
  handoff regeneration is authorized.
- **Status:** complete

### Phase 27: Source-Robust Physical-Chemical 3D Descriptor Upgrade
- [x] **P0 — Input and implementation audit:** identify the retained train-only
  structure/conformer artifacts, reusable geometry utilities, canonical folds,
  and exact Phase 25 reference stream without changing probabilities
- [x] **P0 — Freeze a bounded descriptor contract:** define no more than 20
  physically interpretable sample-level descriptors spanning solvent exposure,
  hydrophobic/charge surface organization, secondary-structure/shape,
  contact order, compactness, and conformer uncertainty; record units,
  missingness, clipping, and aggregation before viewing candidate OOF metrics
- [x] **P1 — Source/length audit:** export fold-safe descriptor diagnostics by
  generator and 11-15/16-30/31-50 aa strata; report raw and fold-local
  source-standardized source probes, while keeping the frozen 20-feature set
  unchanged and forbidding post-hoc subset selection
- [x] **P1 — Bounded OOF candidate:** train one small fold-local regularized
  residual head on the frozen descriptors and retained 3D score, preserving the
  exact missing-structure fallback and fixed threshold `0.677819`
- [x] **P2 — Short-peptide decision:** the global candidate failed specifically
  in 11-15 aa. Do not retrofit a new rule on the viewed OOF; move any true
  short-peptide specialist to a separately frozen phase with fresh training
  provenance, while leaving the retained handoff unchanged here
- [x] **P2 — Frozen promotion gate:** promote only if MCC exceeds `0.839018`,
  AUPRC is at least `0.928171`, at least three of five fold MCC values do not
  decrease, Brier/ECE-10 do not materially regress, and no length/source cohort
  loses more than its preregistered tolerance
- [x] **P3 — Verification and handoff:** add focused tests, compact OOF evidence,
  README documentation, and regenerate the 3D team handoff only if every gate
  passes
- **Boundary:** Test1/Test2 remain sealed. No post-hoc descriptor, seed,
  threshold, residual weight, length rule, or checkpoint selection is allowed
  after candidate OOF outcomes are viewed. Phase 26 artifacts remain untouched.
- **Decision:** `retain_phase25_3d_handoff`. The frozen residual improves AUPRC
  and Brier slightly, but lowers MCC to `0.838166`, produces net `-2` decision
  corrections, and exceeds the 11-15 aa MCC regression tolerance. No feature,
  weight, threshold, source, or length post-selection is authorized.
- **Status:** complete

### Phase 28: All-Atom Short-Peptide Surface-Chemistry Specialist
- [x] **P0 — SASA feasibility audit:** verify a deterministic Biopython
  Shrake-Rupley implementation on PepFlow multi-model PDBs, freeze heavy-atom
  handling, benchmark representative 11-15 aa files, and estimate the 761-row
  CPU/storage cost before any candidate OOF is computed
- [x] **P0 — Fresh protocol freeze:** define at most 12 surface-only descriptors
  (total/nonpolar/polar/charged/sulfur exposure, exposed hydrophobic/electrostatic
  patches, SASA-weighted spatial moments, and conformer variability), one
  fold-local short-only residual, exact fallback for all other rows, and strict
  no-search promotion gates
- [x] **P1 — Implement and test:** add the all-atom parser/extractor, manifest
  alignment, deterministic cache, short-only five-fold evaluator, and focused
  geometry/missing-fallback tests
- [x] **P2 — Execute one frozen OOF:** evaluate only the preregistered candidate,
  report global/short/fold metrics and decision transitions, and stop without
  feature/weight/threshold search if any gate fails
- [x] **P3 — Document and hand off:** retain Phase 25 unless all gates pass;
  document reproducibility, costs, metrics, and the next 3D direction
- **Boundary:** this is a new prospective phase, not a rescue of Phase 27.
  Test1/Test2 remain sealed; medium/long and all non-PepFlow rows must preserve
  the retained Phase 25 probability exactly.
- **Decision:** `retain_phase25_3d_handoff`. Surface features improve short and
  overall AUPRC plus calibration, but lower short MCC to `0.645344`, lower
  overall MCC to `0.838649`, and introduce the only changed decision as an
  error. No weight, feature, threshold, or conformer post-selection is allowed.
- **Status:** complete

### Phase 29: End-to-End Fixed-Threshold Short-Peptide 3D-v2 Specialist
- [x] **P0 — Existing-control audit:** inspect the Phase 6/7 margin loss,
  retained fold-specific 3D-v2 checkpoints, processed PepFlow geometry, and
  trainable model paths so this experiment adds residue-level 3D evidence
  instead of repeating a tested sample-level residual
- [x] **P0 — Prospective protocol freeze:** before viewing any new candidate
  OOF result, freeze the balanced 250-row PepFlow pilot cohort, canonical outer folds,
  fold-specific initialization, fixed epoch/seed/margin settings, exact Phase
  25 fallback for the remaining 6,137 rows, threshold `0.677819`, and strict
  global/short/fold promotion gates
- [x] **P1 — Implement and test:** add leakage-safe short-cohort preparation,
  fold training/evaluation wrappers, fixed-threshold margin diagnostics, exact
  fallback checks, and focused tests
- [x] **P2 — Local smoke and cost gate:** run only schema/smoke validation
  locally, estimate RTX 4090 time/storage, and package a self-contained AutoDL
  handoff if the full frozen five-fold run is not economical locally
- [x] **P3 — Execute one frozen five-fold OOF:** run the preregistered candidate,
  aggregate all 6,387 rows, report global/short/fold metrics plus threshold
  transitions, and stop without epoch/head/margin/seed/threshold post-selection
- [x] **P4 — Document and hand off:** retain Phase 25 unless every gate passes;
  document commands, hashes, costs, result, and the next authorized 3D direction
- **Boundary:** Test1/Test2 remain sealed. The held outer fold must never enter
  specialist training, model selection, or calibration. Non-short rows must
  preserve the retained Phase 25 probability exactly, and no best-seed or
  best-epoch selection may be performed from candidate OOF outcomes.
- **Decision:** `retain_phase25_3d_handoff`. The candidate preserves all 6,137
  fallback probabilities but lowers global MCC to `0.838611`, lowers pilot250
  MCC to `0.564547`, changes one correct decision to an error, and has a
  `-0.038184` fold-1 MCC regression. No threshold/epoch/seed/margin rescue is
  authorized.
- **Status:** complete

### Phase 30: Existing-Data Multi-Conformer Domain-Invariant 3D Pretraining
- [x] **P0 — Non-duplication and artifact audit:** compare the completed
  single-structure geometry SSL, ESMFold-to-ensemble cross-view screen, and
  supervised Phase 24 graph against the available 3,195 legacy conformer rows,
  200 BioEmu rows, and 143 matched legacy/BioEmu pairs; identify one genuinely
  new residue-level label-free path without modifying Phase 25
- [x] **P0 — Prospective protocol freeze:** before viewing candidate toxicity
  OOF metrics, freeze fold-pure pretraining data, two-conformer residue/pair
  objectives, matched cross-generator invariance, initialization, fine-tuning
  scope, fixed epochs/seed, threshold `0.677819`, exact fallback, and promotion
  gates
- [x] **P1 — Implement and verify:** add the existing-data pretrainer,
  alignment-safe embedding attachment, fold-specific fine-tuning/evaluation,
  source-leakage diagnostics, resumable launchers, and focused synthetic tests
- [x] **P2 — Execute one frozen five-fold OOF:** run only the preregistered
  candidate, report global/fold/length/source metrics, paired representation
  diagnostics, calibration, and fixed-threshold decision transitions
- [x] **P3 — Retain or promote:** regenerate the 3D handoff only if every gate
  passes; otherwise retain Phase 25 and stop without architecture, loss, seed,
  epoch, threshold, or subset rescue
- **Boundary:** use only existing local training structures and BioEmu pilot
  artifacts. Test1/Test2 remain sealed. The earlier single-structure SSL,
  ESMFold-anchor consistency, Phase 24 from-scratch graph, scalar descriptors,
  random-seed search, and Phase 29 margin experiment may not be repeated under
  new names.
- **Status:** complete

### Phase 31: Cross-Fitted Shared-Slope Length-Intercept 3D Calibration Screen
- [x] **P0 — Freeze the minimal calibrator:** use the Phase 25 OOF score only,
  with `logit(p_cal)=a*logit(p_phase25)+b0+bs*I(11-15)+bl*I(31-50)`,
  `a=exp(alpha)>0`, 16-30 aa as the reference stratum, no generator, pLDDT,
  label-ratio, Phase 30, 1D, or 2D covariates
- [x] **P0 — Freeze optimization and evaluation:** each held fold receives
  parameters fitted on the other four folds only; minimize mean logistic loss
  plus `0.01*alpha^2+0.25*(bs^2+bl^2)` by deterministic L-BFGS-B, bound
  `alpha` to `[-3,3]`, and classify calibrated probabilities at exactly `0.5`
- [x] **P1 — Implement and test:** add the calibrator, frozen-input verifier,
  per-fold coefficient/convergence records, two-column fusion handoff,
  probability correlation, global/fold/length metrics, confusion matrices,
  decision transitions, launchers, and synthetic fold-safety tests
- [x] **P2 — Execute one CPU OOF candidate:** run the single frozen candidate
  on all 6,387 Phase 25 OOF rows without reading Test1/Test2 or searching any
  regularization, threshold, feature, seed, or alternative parameterization
- [x] **P3 — Retain or advance:** advance to a strict nested Phase 25 retraining
  only if global MCC strictly improves,
  AUPRC/calibration and every length stratum remain within frozen tolerances,
  at least three folds have non-decreasing MCC, short-peptide MCC does not
  decrease, and fixed-threshold net corrections are positive; otherwise keep
  the Phase 25 handoff and stop this calibration route
- **Boundary:** this is calibration of the retained 3D stream only. The raw
  Phase 25 probability must be preserved as a separate output column for the
  fusion team, and the existing handoff must not be overwritten unless every
  advance condition passes. This screen cannot directly replace the official
  handoff because the Phase 25 scores for the four calibration-training folds
  were generated by upstream models that may include the current outer fold.
- **Decision:** `retain_phase25_and_stop_length_calibration`. The exploratory
  screen improves Brier/ECE substantially but lowers MCC to `0.835319`, AUPRC
  to `0.924213`, short-peptide MCC to `0.632614`, and produces net `-11`
  fixed-threshold corrections. Do not spend GPU time on strict nested Phase 25
  retraining for this calibrator.
- **Status:** complete

### Phase 32: Fold-Pure Partial-Unfreeze Ensemble 3D Evidence
- [x] **P0 - Audit and freeze:** verify the five Phase 30 fold-pure encoder
  checkpoints, identify `encoder.layers.1` as the only trainable encoder block,
  and freeze every input hash, training parameter, fusion coefficient, and
  promotion gate before computing candidate OOF probabilities
- [x] **P1 - Implement and test:** add packed multi-conformer batching, a
  standalone residue-attention toxicity head, last-layer-only fine-tuning,
  exact missing-structure fallback, source diagnostics, resumable launchers,
  and focused fold-safety/freezing tests
- [x] **P2 - Execute one fixed five-fold OOF:** initialize each outer fold from
  its corresponding Phase 30 encoder, train on toxicity labels from the other
  four folds only, and apply a fixed bounded evidence term to the held-out
  Phase 25 logit only after training
- [x] **P3 - Retain or promote:** report global/fold/length/source metrics and
  fixed-threshold decision transitions; regenerate the 3D handoff only if every
  preregistered promotion condition passes, otherwise retain Phase 25 exactly
- **Leakage boundary:** neither Phase 25 probabilities nor logits may enter the
  supervised training dataset, loss, checkpoint selection, or optimizer. The
  fixed formula `logit(p_candidate)=logit(p_phase25)+0.15*tanh(z_3d)` is applied
  only to each held row after its independent 3D head has been trained without
  that row's outer fold. This avoids the Phase 31 second-level OOF provenance
  problem.
- **Boundary:** use only the existing 3,252 ensemble-covered rows and five
  Phase 30 fold encoders; all other 3,135 rows must be exact Phase 25 fallbacks.
  Test1/Test2 remain sealed, with no architecture, seed, epoch, threshold,
  evidence-weight, subset, checkpoint, or length/source post-selection.
- **Decision:** `retain_phase25_3d_handoff`. The bounded candidate slightly
  improves AUPRC/Brier/ECE but lowers MCC from `0.839018` to `0.838129`, with
  four corrected versus six regressed decisions. All three length-stratum MCC
  values decrease, so the Phase 25 handoff remains the official 3D output.
- **Status:** complete

### Phase 33: Unified-BioEmu Decision-Boundary 3D Specialist
- [x] **P0 - Freeze label-blind cohort selection:** rank all 6,387 retained
  Phase 25 OOF rows only by absolute probability distance from the fixed
  threshold `0.677819`, select the closest 600 rows without using labels,
  preserve canonical fold/sequence alignment, and reuse any already generated
  BioEmu rows that fall inside the frozen cohort
- [x] **P0 - Freeze generation and evaluation protocol:** use one BioEmu version,
  checkpoint, sample count, seed rule, filtering rule, and resumable output layout
  across all three length strata; keep Phase 25 probabilities exact outside the
  selected cohort and keep Test1/Test2 sealed
- [x] **P1 - Prepare and verify AutoDL handoff:** emit CSV/FASTA manifests,
  overlap/reuse audit, expected runtime/storage projection, checksums, a resumable
  AutoDL launcher, and a compact upload bundle
- [x] **P2 - Import and audit generated structures:** verify archive SHA256,
  sample/sequence identity, conformer usability, residue counts, and exact cohort
  coverage before any toxicity candidate is trained
- [x] **P3 - Pretrain and evaluate one topology/dynamics-aware specialist:** train
  fold-pure on the other four folds only, apply a bounded residual only to held
  boundary rows, and report global/fold/length metrics plus fixed-threshold
  decision transitions
- [x] **P4 - Retain or promote:** regenerate the team handoff only if global MCC
  strictly exceeds `0.839018`, AUPRC is at least `0.928171`, at least three fold
  MCC values do not decrease, every length layer stays within its frozen
  tolerance, and net decision corrections are positive
- **Boundary:** cohort membership is determined without labels. No random-seed,
  threshold, subset, generator, topology-feature, checkpoint, epoch, or residual
  weight search is allowed after candidate OOF results are viewed. Phase 25 is
  the exact fallback for every non-selected row.
- **Decision:** `promote_phase33_3d_handoff`. The fixed boundary-only candidate
  improves MCC from `0.839018` to `0.839758`, AUPRC from `0.928171` to
  `0.928204`, Brier from `0.059399` to `0.059378`, and ECE-10 from `0.051109`
  to `0.051094`. It changes four fixed-threshold decisions, correcting three
  and regressing one, while all 5,787 non-boundary rows remain exact Phase 25
  fallbacks.
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
| Phase 33 completion planning patch repeated the earlier leading-space-before-header typo | 1 | Reapplied as valid contiguous multi-file patch headers; artifact bytes and experiment protocol were unaffected |
| Phase 33 full-manifest dry-run command included a recursive temporary-directory cleanup and was rejected before execution | 1 | Rerun as a non-destructive TEMP-only validation and leave the isolated temporary output for the operating system to clean; no project or experiment file was created by the rejected command |
| Phase 33 parallel preflight returned nonzero because Ruff found formatting-only drift in two Python files | 1 | Apply the existing mechanical Ruff formatter, then rerun semantic checks, focused tests, Bash parsing, and whitespace checks independently |
| Phase 33 multi-file planning patch accidentally prefixed a patch header with a space | 1 | Reapplied the same intended edits with valid patch headers; no project code or experiment artifact had been changed |
| Phase 33 AutoDL runner stopped after each first failing BioEmu sample, causing retries to advance from only 3 to 5 statuses | 1 | Add an explicit `--continue-on-error` runner mode, enable it in the AutoDL launcher, and verify all manifest rows are visited while failures stay recorded for repair |
| Phase 32 final artifact audit reused the known PowerShell `foreach { } |` parse trap | 1 | No files or results changed; collect the fold rows in a task-specific array before formatting and keep subsequent checks independent |
| Phase 32 first fold reported NaN loss before producing predictions/checkpoints | 1 | Stopped the exact background PID, reproduced non-finite last-layer gradients on the first real batch, and traced them to the derivative of conformer standard deviation at zero variance; add a fixed `1e-6` variance epsilon and require finite backward gradients before restarting from an empty fold directory |
| Phase 32 synthetic ensemble fixture assumed the historical 21-channel residue width | 1 | Import the current `RESIDUE_FEATURE_DIM` constant (42) in the test fixture; production data loading and the frozen protocol were unchanged |
| Phase 32 Ruff format check found mechanical layout drift in the new module | 1 | Applied the existing Ruff formatter and reran semantic/format checks without changing the frozen experiment |
| Phase 29 final focused test used a minimal synthetic protocol without an `evaluation` block | 1 | Made the newly added full-evaluation hash checks optional for backward-compatible synthetic protocols while keeping them mandatory in the real frozen protocol |
| Phase 29 pytest could not parse `pyproject.toml` because the new CLI entry was duplicated | 1 | Removed the duplicate identical key before test collection; no candidate training or result was produced |
| Phase 29 `rg` audit command used an unterminated PowerShell double quote | 1 | No files changed; reran with a single-quoted search expression |
| Phase 29 parallel preflight stopped when Ruff observed a transient duplicate TOML script key | 1 | Inspected the settled file, confirmed one key, removed duplicated `.gitignore` exceptions, and reran checks independently |
| Phase 29 Bash syntax check could not resolve `bash` from `PATH` | 1 | Used the installed Git-for-Windows absolute Bash path; syntax passed without changing host configuration |
| Phase 29 frozen-input verifier found a config SHA256 mismatch before training | 1 | Confirmed no candidate had started and no config parameter changed; refreshed the pre-candidate hash to the actual finalized file bytes, then rerun verification independently |
| Phase 29 completion patch used stale pre-compaction text twice | 2 | Re-read the canonical plan and applied smaller exact-context updates; no experiment artifact changed |
| Phase 30 default Python artifact inspection could not import PyTorch | 1 | Switched read-only schema inspection to the documented `D:\\anaconda\\envs\\pytorch_env\\python.exe`; no environment or artifact changed |
| Phase 30 wildcard `rg` audit used a Windows-invalid `runs/geometry_*` path | 1 | Read the known geometry summaries by explicit path and keep wildcard matching in `--glob` only; no artifact changed |
| Phase 30 combined protocol/planning patch used stale mojibake checkbox text | 1 | Split the protocol creation and UTF-8 planning updates into small exact-context patches; the protocol settings were unchanged |
| Phase 30 test-helper audit named nonexistent `conformer_ensemble.py` | 1 | Used the actual `conformer_cache.py` discovered by `rg`; the failed read-only command changed no artifact |
| `pytorch_env` has no importable Ruff module | 1 | Used the already installed `D:\\anaconda\\Scripts\\ruff.exe` for linting; no package installation or environment mutation |
| Full feature-building preflight exceeded the 120-second command limit | 1 | The process was terminated before training and produced no candidate output; split preflight into fast schema validation and leave one-time feature construction inside the resumable formal run |
| Final focused pytest command named nonexistent `test_geometry_pretrain.py` | 1 | Retain the already passed launcher syntax checks and rerun the two discovered related test files only; no code or result artifact changed |
| Phase 31 Ruff format check found mechanical layout drift in the new module | 1 | Apply the existing Ruff formatter, then rerun semantic and formatting checks; the frozen model/protocol is unchanged |
| Phase 31 test file required Ruff's mechanical formatting | 1 | Both tests and semantic lint already passed; format only the test and rerun all focused checks without changing protocol or candidate code |
| Phase 27 parallel artifact audit stopped on an optional `rg` no-match exit code | 1 | Split the audit into independent commands and explicitly allow optional no-match results |
| Phase 27 reused the known PowerShell `foreach { } |` parse trap | 1 | No files changed; ban this construct for the remainder of Phase 27 and use independent commands or a parenthesized collected array |
| Windows `rg` rejected unexpanded `requirements*.txt` / `environment*.yml` path arguments | 1 | The relevant `pyproject.toml` audit completed; use `rg --glob` for any later wildcard search |
| `pytorch_env` has no Ruff module, aborting collection of parallel check outputs | 1 | Re-run compile/tests independently and use an available Ruff executable only if discovered |
| Focused pytest collection could not import `ghxtox` | 1 | Set the repository-standard `PYTHONPATH=src` for test and CLI commands; do not mutate the environment |
| Phase 27 synthetic tests found a directional patch-test assumption and empty-cohort reporting failure | 1 | Test for geometry sensitivity without assuming exposure direction; skip absent report cohorts while retaining all real frozen strata |
| Phase 28 synthetic PDB test found Biopython residue-map key case incompatibility | 1 | Resolve uppercase keys first and retain title-case fallback; no protocol or descriptor change |
| PowerShell profile execution-policy warning | 1 | Treat as harmless because commands still execute successfully |
| Phase 26 wrapper parsed `$LASTEXITCODE:` as a drive-qualified variable | 1 | Delimit it as `${LASTEXITCODE}` and rerun the parser audit |
| Direct `.ps1` launch was blocked by Windows ExecutionPolicy | 1 | Use a one-process `powershell.exe -NoProfile -ExecutionPolicy Bypass -File` launcher; do not change machine policy |
| Historical fold inspection queried nonexistent `summary.json` files | 1 | Use the existing `config.json`/`history.json` artifacts and their timestamps instead |
| WSL executable exists but no Linux distribution is installed | 1 | Use Git for Windows Bash for `bash -n`; do not alter the host installation |
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
