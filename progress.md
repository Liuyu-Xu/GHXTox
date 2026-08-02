# Progress Log

## Session: 2026-07-29

### Phase 1: Audit Inputs and Existing Coupler
- **Status:** complete
- **Started:** 2026-07-29
- Actions taken:
  - Restored the current repository state and identified existing uncommitted changes.
  - Created persistent planning files for the coupling upgrade.
  - Confirmed that the prior dynamic coupling module and conformer artifacts are present.
  - Confirmed full 6,387-row alignment across ProtT5, 3D-v2, and conformer residual OOF files.
  - Audited non-constant conformer quality columns and existing nested CV utilities.
  - Discovered that the latest repository already completed and rejected sample-level quality gate, stacking, and signed residual variants.
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 2: Design Stronger Coupling Protocol
- **Status:** complete
- Actions taken:
  - Reframed the next change to require new residue/conformer-level information rather than another sample-level OOF fusion search.
  - Selected residue-level dynamics injection plus global conformer conditioning with a hard availability mask.
- Files created/modified:
  -

### Phase 3: Implement
- **Status:** complete
- Actions taken:
  - Added `attach-conformer` support to the cross-modal feature cache.
  - Added per-residue conformer dynamics gating before cross-attention.
  - Added global conformer/source conditioning of the bounded adapter residual.
  - Added explicit OOF/full-training flags while keeping old checkpoints compatible.
  - Built the 2,795-sample multi-source conformer processed artifact.
  - Regenerated the 6,387-sample strict OOF ProtT5/spatial feature cache.
  - Attached and validated chemical-site and conformer tensors.
- Files created/modified:
  - `src/ghxtox/cross_modal_adapter.py`
  - `tests/test_cross_modal_adapter.py`

### Phase 4: Tests and OOF Screening
- **Status:** complete
- Actions taken:
  - Ran syntax checks and focused adapter/conformer/coupling tests.
  - Reproduced the immutable frozen baseline at MCC 0.869346 and AUPRC 0.953598.
  - Screened joint chemical/conformer retraining: MCC 0.870828, below the current full default.
  - Refactored the conformer route into a zero-initialized residual over a frozen chemical adapter.
  - Reproduced the chemical adapter OOF MCC exactly at 0.871184 and saved all five missing fold checkpoints.
  - Detected and corrected training-mode Dropout in the parameter-frozen base; base modules now remain in evaluation mode while conformer modules train.
  - Screened the final function-frozen conformer residual: seed-42 MCC 0.871583 with exact uncovered-sample fallback, one fold win, and four ties.
  - Ran fixed seeds 42/123/2025 and aggregated them without seed selection; MCC 0.871583 with one fold win, four ties, and slightly lower AUPRC.
- Files created/modified:
  -

### Phase 5: Documentation and Handoff
- **Status:** complete
- Actions taken:
  - Added commands, metrics, coverage, fallback guarantees, and the stop decision to README.
  - Added a row-alignment-checked OOF seed aggregation command.
  - Ran Ruff formatting/linting, focused regressions, and Git whitespace checks.
- Files created/modified:
  -

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Initial repository audit | `git status`, `rg --files` | Locate relevant code and preserve user changes | Relevant files found; dirty outputs left intact | Pass |
| Cross-modal adapter tests | `pytest tests/test_cross_modal_adapter.py -q` | Old behavior and new conformer branch pass | 5 passed | Pass |
| Conformer/coupling regression tests | `pytest tests/test_coupled_fusion.py tests/test_conformer_features.py -q` | No regressions | 9 passed, 3 expected transformer warnings | Pass |
| Combined conformer processed artifact | Three local multi-conformer manifests | 2,795 aligned records | 2,795 / 6,387 (43.7608%) | Pass |
| Strict OOF adapter precompute | Fold-excluded checkpoints | 6,387 samples with no in-fold spatial state | 6,387 samples, 201,994 residues | Pass |
| Attached tensor integrity | OOF cache | Aligned, finite, exact fallback for missing conformers | All finite; uncovered global/source tensors exactly zero | Pass |
| Joint conformer adapter OOF | 5 group-aware folds | Beat frozen baseline and current full default | MCC 0.870828; beats frozen baseline but not full default | Partial |
| Chemical adapter reproduction | Combined cache, seed 42 | Match current default | MCC/ACC/SN/SP exactly matched | Pass |
| Function-frozen conformer residual OOF | Frozen chemical fold checkpoints | Improve aggregate and at least 3 fold MCCs | MCC +0.000399, 1 win/4 ties/0 losses | Partial |
| Fixed three-seed residual ensemble | Seeds 42/123/2025 | Reduce variance and pass retention gates | MCC +0.000399, AUPRC -0.000011, 1 win/4 ties/0 losses | Fail |
| Final focused regression suite | Adapter, coupling, conformer tests | No regressions | 16 passed, 3 known Transformer warnings | Pass |
| Ruff and whitespace checks | Modified Python files and Git diff | Clean | Ruff format/check and `git diff --check` passed | Pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-07-29 | PowerShell profile cannot run under execution policy | 1 | Ignored because command payloads complete successfully |
| 2026-07-29 | CSV audit command failed with `EmptyPipeElement` | 1 | Rewrote the loop to collect results before formatting |
| 2026-07-29 | Quality-column audit repeated `EmptyPipeElement` | 2 | Stopped using direct `foreach ... |` constructs |
| 2026-07-29 | Planning-file patch verification failed | 1 | Re-read files and applied smaller exact-context changes |
| 2026-07-29 | CSV search reused PowerShell automatic `$Matches` | 1 | Renamed the collection variable and avoided automatic names |
| 2026-07-29 | Optional processed-artifact search returned exit code 1 | 1 | Switched optional discovery to non-failing PowerShell enumeration |
| 2026-07-29 | `D:\anaconda\envs\pytorch_env` has no pytest module | 1 | Install pytest before running the focused suite |
| 2026-07-29 | OOF feature precompute exceeded a 10-second shell timeout | 1 | Use a hidden background process and inspect PID/log/output |
| 2026-07-29 | Cache verification expected `token_values`, but the schema uses `token_embeddings` | 1 | Read actual keys and rerun checks against the serialized format |
| 2026-07-29 | OOF aggregation unit test used exact decimal equality for float32 CSV output | 1 | Replaced with `torch.allclose` |
| 2026-07-29 | Ruff reported two unformatted files and an unused `math` import | 1 | Removed the import and ran Ruff format/check |
| 2026-07-29 | Frozen parameters still had base Dropout in training mode | 1 | Keep every non-conformer child module in evaluation mode during residual fitting and rerun all fixed seeds |
| 2026-07-29 | Sequential three-seed wrapper exceeded the 240-second command timeout | 1 | Keep completed seeds 42/123 and rerun only missing seed 2025 as a monitored background process |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5 complete |
| Where am I going? | Preserve the default and carry the conformer residual forward as an ablation |
| What's the goal? | Improve fair OOF performance without calibration regression |
| What have I learned? | Sample-level conformer gates were already tested and rejected |
| What have I done? | Implemented, screened, documented, and stopped the conformer upgrade under predeclared gates |

## Session: Required Coupling Completion

### Phase 6: Implement Required Training Controls
- **Status:** complete
- Actions taken:
  - Diagnosed coverage, source/class balance, length domains, and threshold decision changes.
  - Fixed a four-variant OOF ablation matrix before implementation.
  - Added asymmetric threshold-neighborhood margin loss.
  - Added a zero-initialized source/length-aware low-rank conformer residual.
  - Added fold-local label/source/length-bin balanced conformer sampling.
  - Added strict argument validation while preserving disabled-by-default compatibility.
  - Added focused tests for zero initialization, source-specific correction,
    fold-local sampling, unavailable-record exclusion, and asymmetric margin loss.
  - Ran the focused adapter suite: 10 passed.
  - Ran Ruff formatting and lint checks successfully.
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `src/ghxtox/cross_modal_adapter.py`
  - `tests/test_cross_modal_adapter.py`

### Required-Control Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-07-29 | Target PyTorch environment has no `ruff` module | 1 | Use standalone/default-environment Ruff for final linting |
| 2026-07-29 | Exact equality failed after `sigmoid(logit(p))` floating-point round trip | 1 | Check exact zero residual and probability equality with tolerance |
| 2026-07-29 | Old checkpoint loader rejected the new zero-initialized source/length head | 1 | Allow that conformer-only prefix and extend the initialization compatibility test |

### Phase 7: Fixed OOF Ablation Matrix
- **Status:** complete
- Actions taken:
  - Froze all common training settings and the three treatment settings before
    viewing any new OOF result.
  - Reproduced the control at MCC 0.871583 with one fold win.
  - Ran Margin-only at MCC 0.871982 with two fold wins.
  - Ran Source-length-only at MCC 0.871583 with one fold win.
  - Ran Combined at MCC 0.871658 with three wins, one tie, and one loss; Brier regressed.
  - Applied unchanged retention gates; no variant passed, so seed expansion,
    full-data refit, and Test1/Test2 evaluation were stopped.
  - Generated `matrix_summary.json` and `matrix_summary.csv`.

### Phase 8: Tests and Documentation
- **Status:** complete
- Actions taken:
  - Added a reusable `summarize-ablation` CLI and focused report-generation test.
  - Updated README with fixed settings, exact results, stopping rule, and summary command.
  - Ran compile checks, Ruff formatting/linting, and the full focused regression suite.
  - Final result: 20 tests passed with three known Transformer warnings.
  - Ran Git whitespace checks successfully and confirmed no adapter training process remains.

## Required-Control Final Decision
- Keep the reproduced chemical cross-attention adapter as the paper default.
- Retain Margin-only and Combined as informative ablations, not promoted models.
- Do not run extra seeds, full-data refit, or Test1/Test2 because no candidate passed
  every predeclared OOF retention gate.

## Session: 31-50 aa AF2 Pilot

### Phase 9: Import and Audit
- **Status:** complete
- Actions taken:
  - Located the server input and result archives in `D:\dachuang2026\ziliao`.
  - Verified both SHA256 files against the downloaded archives.
  - Audited 400 balanced pilot samples, 3,200 conformers, 400 summaries,
    eight conformers per sample, and zero sequence mismatches.
  - Copied the result archive/checksum into the project and extracted only the
    dedicated 31-50 data and run namespaces.

### Phase 10: Fixed OOF Evaluation
- **Status:** complete
- Actions taken:
  - Added resumable `scripts/evaluate_colabfold_31_50_oof.ps1`.
  - Built the full augmented tensor, 400-sample subset, and fixed fold manifest.
  - Trained all five residual folds without touching Test1/Test2.
  - Ran covered-cohort and 6,387-sample OOF predictions plus 1,000-iteration
    paired bootstrap/randomization comparisons.
  - Observed AUROC/AUPRC/Brier gains but BACC/MCC regressions.
  - Applied the unchanged gate and stopped before refit, ProtT5 fusion, or
    generation of the remaining 3,192 structures.

### Phase 11: Regression and Documentation
- **Status:** complete
- Actions taken:
  - Fixed the cohort length buckets for 31-50 aa and the multisource long-length
    reporting omission.
  - Recomputed the structure cohort report with 31-35, 36-40, 41-45, and 46-50
    strata.
  - Added a boundary regression test; focused conformer suite passes 16 tests.
  - Ruff passes on all modified Python files.
  - Updated README with exact pilot metrics, confidence intervals, stopping
    decision, and the resumable reproduction command.

## Session: Latest Upstream Reference Integration

### Phase 12: Upstream Audit
- **Status:** complete
- Actions taken:
  - Confirmed the local `origin` is `Suiyumanbu/GHXTox`.
  - Opened the current GitHub main page and found a newer geometry-SSL/three-level
    fusion experiment not present in the local README.
  - Preserved the dirty local worktree and prepared a fetch-and-compare workflow
    rather than replacing files wholesale.
  - Fetched upstream main: local baseline `8cb7da2`, upstream `907acc9`, one
    upstream-only commit.
  - Classified the 206-file commit as mostly generated output and narrowed the
    integration audit to the actual algorithm, config, tests, and compact summaries.
  - Read the geometry SSL implementation and confirmed its OOF label isolation,
    inference-embedding path, and optional cross-view extension.
  - Audited upstream 31-50 and geometry-screen decisions; both preserve the current
    default after transfer/retention failures.
  - Fixed the integration scope to reusable algorithm code and compact evidence,
    excluding generated logs, PIDs, checkpoints, predictions, and temporary files.

### Phase 13: Selective Integration
- **Status:** complete
- Actions taken:
  - Added fold-pure, label-free geometry SSL with optional cross-view refinement.
  - Merged geometry gating and residue/motif/peptide hierarchical fusion into the
    locally extended conformer-aware adapter.
  - Preserved exact missing-structure and missing-conformer fallback behavior.
  - Added checkpoint shape-aware initialization for old adapter checkpoints.
  - Added exact chunked edge-bias evaluation for long dense chemical graphs.
  - Added confidence/length-balanced pilot selection without overwriting the local
    shared long-length bucket implementation.
  - Imported only three compact machine-readable result summaries.

### Phase 14: Verification and Documentation
- **Status:** complete
- Actions taken:
  - Added geometry alignment, zero-fallback, hierarchical-head, SSL-loss,
    cross-view-loss, checkpoint-compatibility, graph-chunking, and pilot tests.
  - Ruff passed on all touched Python files.
  - Focused suite: 28 passed.
  - Full suite with `MPLBACKEND=Agg` and only five unavailable RDKit tests excluded:
    114 passed, 5 deselected.
  - Updated README with protocol, commands, OOF/historical metrics, artifact links,
    and the unchanged default-model decision.

## Session: Bidirectional 1D-3D Coupling

### Phase 15: Fixed Design and Protocol
- **Status:** complete
- Actions taken:
  - Kept the retained ProtT5-query chemical interaction adapter immutable.
  - Predeclared a frozen-base bidirectional residual screen before viewing results.
  - Fixed the OOF gate to MCC > 0.871184, AUPRC >= 0.953639,
    Brier <= 0.044464, at least three fold MCC wins, and at most one fold loss.

### Phase 16: Implementation
- **Status:** complete
- Actions taken:
  - Added reverse 3D-query/1D-key-value attention.
  - Added confidence-weighted cosine agreement and a learned residue mutual gate.
  - Added an independent zero-initialized bounded bidirectional logit residual.
  - Added old-checkpoint compatibility, frozen-extension training, a
    bidirectional-only loss mode, and per-sample coupling diagnostics.
  - Added fold checkpoint resume support to the OOF trainer.

### Phase 17: Verification and Reproducibility
- **Status:** complete
- Actions taken:
  - Added `scripts/evaluate_bidirectional_coupling_oof.ps1` with a fixed protocol,
    result-independent gate, and `-Resume` support.
  - Added focused fallback, gate-bound, checkpoint, freeze, and training tests.
  - Completed an ephemeral two-fold CPU OOF/resume smoke test.
  - Ruff and PowerShell syntax checks passed.
  - Focused adapter suite: 18 passed.
  - Full non-RDKit suite: 116 passed, 5 deselected; the five excluded tests require
    the unavailable local RDKit package.
  - Documented the candidate as implemented but not yet OOF-evaluated; the current
    engineering default remains unchanged.

### Phase 18: Server OOF Result Integration
- **Status:** complete
- Actions taken:
  - Imported five fold checkpoints, histories, OOF predictions, summary, and the
    predeclared decision from the AutoDL result archive.
  - Confirmed candidate MCC 0.870001 versus baseline 0.871184, with zero fold wins.
  - Confirmed small Brier/ECE improvements but lower BACC, MCC, AUROC, and AUPRC.
  - Stopped before full-data refit and Test1/Test2 evaluation as required by the
    frozen gate.
  - Updated README and retained the branch as a calibration-positive,
    classification-negative ablation.

## Session: Unified-Conformer Pilot Preparation

### Phase 19: Length and Generator-Source OOF Diagnostics
- **Status:** in_progress
- Actions taken:
  - Restored the completed coupling and upstream-integration context.
  - Read the latest README navigation, priority index, Appendix B stopping
    decisions, and Appendix C handoff instruction.
  - Froze the next scope to train-OOF diagnostics plus a 200-sample BioEmu
    pilot package; full generation and Test1/Test2 access remain prohibited.
  - Confirmed the exact three length strata and fixed BioEmu continuation gate.
  - Located reusable cohort/multisource analysis and pilot-selection utilities;
    no existing BioEmu launcher or evaluator was found.
  - Identified aligned 6,387-row 1D, 3D-v2, and retained-joint OOF prediction files
    plus the canonical train fold manifest.
  - Identified a sampling edge case: 200 is not divisible by 30 fold/label/length
    cells, so the BioEmu selector needs deterministic near-balanced quotas.
  - Verified the current official BioEmu package/API and froze package 1.4.1,
    model `bioemu-v1.2`, deterministic `base_seed`, default physical filtering,
    and resumable per-sequence output directories for the server protocol.
  - Audited the four local ensemble cache roots and confirmed perfect generator-
    length confounding between full PepFlow and AF2 16-30 coverage.
  - Implemented a train-only OOF diagnostic with exact length strata, fixed model
    thresholds, error-type tables, source summaries, Cramer's V, and five-fold
    cross-fitted feature predictability.
  - Implemented deterministic near-balanced 200-row BioEmu selection, six-row
    smoke selection, version/seed/model-pinned commands, resumable execution,
    structure/diversity/cost assessment, and machine-readable decisions.
  - Added Linux/AutoDL environment and guarded smoke/pilot launch scripts. The
    launcher intentionally exposes no full-dataset mode.
  - First targeted test collection found the default interpreter does not install
    the src-layout package; lint passed. The rerun will set `PYTHONPATH=src`.

### Phase 20: Reproducible 200-Sample BioEmu Pilot Preparation
- **Status:** complete
- Actions taken:
  - User explicitly reprioritized the GPU 200-sample BioEmu pilot ahead of the
    CPU OOF diagnostic.
  - Preserved the frozen six-sample smoke gate before authorizing all 200 rows.
  - Confirmed the local RTX 4060/CUDA runtime is visible, but BioEmu is Linux-only
    and no AutoDL SSH connection is configured; checking CUDA-enabled WSL next.
  - Generated the first 200-row manifest and found the quota recursion checked
    cells individually rather than grouped multiplicities; fixing it before
    freezing the final manifest.
  - Froze the corrected 200-row allocation at exact 100/100 labels, 40 rows per
    fold, 67/67/66 rows per length stratum, and 6-7 rows per primary cell.
  - Generated the six-row balanced smoke manifest and machine-readable selection
    summary.
  - Added AutoDL installation, guarded smoke/pilot execution, trajectory assessment,
    projected cost gates, checksums, and a pending machine-readable decision.
  - Built and verified `bioemu_pilot_200_server_inputs_20260731.tar.gz`.
  - Updated README with exact counts, frozen versions, upload/run commands, gates,
    current pending status, and the explicit no-full-generation boundary.
  - Verification: 9 related tests passed; Ruff, formatting, Bash syntax, archive
    SHA256, and Git whitespace checks passed.
  - Python 3.10 compile succeeded; moved the independent TOML syntax check to the
    default Python 3.13 interpreter because `tomllib` is unavailable in 3.10.

### Phase 20A: Linux GPU Execution
- **Status:** pending
- Blocking environment:
  - BioEmu is Linux-only.
  - The workstation has no WSL, Docker, or Podman runtime.
  - Connected to the signed-in AutoDL console through Chrome and inventoried five
    healthy, powered-off RTX 4090 x1 instances.
  - Selected instance `074e4a9aca-106dc2e1` (machine 253) as the tentative target
    because it combines low data-disk use (28.03%) with the longest release window
    (14 days 22 hours).
  - No instance has been powered on and no billable action has been taken.
  - Automatic control of the AutoDL tab timed out. Recovery checks found Chrome
    closed and the ChatGPT Chrome Extension unavailable in its selected profile;
    the native-messaging host is valid.
  - The user chose to operate AutoDL manually. Prepared an exact UI/terminal
    handoff using the frozen archive, checksum, installer, six-row smoke gate,
    conditional 200-row command, monitoring, and result-return steps.
  - User-provided runtime evidence confirms BioEmu 1.4.1 and CUDA-enabled RTX 4090
    operation on instance `074e4a9aca-106dc2e1`.
  - `tmux` is absent from the image; switched the manual handoff to a `nohup`
    background job with persistent console log and PID.
  - The first monitoring screenshot shows the RTX 4090 idle with no GPU process.
    This is not yet classified as failure because first-run model downloads and
    MSA preparation can be CPU/network-only; requested PID and log evidence next.
  - The smoke launcher completed and reported
    `decision=switch_to_esmflow_small_pilot`; the 200-row BioEmu run is therefore
    not authorized.
  - Root cause is not yet frozen: the feasibility JSON and per-sample log are
    still needed to distinguish a genuine structure-quality failure from a
    recoverable download, MSA, RAM, or runtime error.
  - The failing sample traceback identified a recoverable checkpoint-download
    failure in Hugging Face Hub/HTTPX before BioEmu inference began.
  - Updated the AutoDL installer to pin `huggingface-hub==0.36.0` and record that
    version in the runtime report. The same six-row smoke must be rerun after
    prefetching the official `microsoft/bioemu` checkpoint and config.
  - Rebuilt the server input archive with the dependency fix. Verified SHA256:
    `2e3812743c48bafdca4fb68d657bc89cfaf7d3a406bc143f00b9509e157d8cff`.
  - The pinned prefetch now fails with a plain connection timeout to
    `huggingface.co:443`, proving that the dependency-client error is resolved
    but AutoDL outbound connectivity remains blocked.
  - Routed the manual retry through AutoDL's documented academic-resource proxy
    and moved Hugging Face cache storage to `/root/autodl-tmp/cache/huggingface`.
  - User evidence confirms the proxy path is working: Hugging Face returned
    `200 Connection established` followed by `200 OK` through AutoDL's cache.
  - Successfully prefetched the official 143 MB BioEmu v1.2 checkpoint and
    config into `/root/autodl-tmp/cache/huggingface`.
  - The frozen smoke retry is now running under `nohup` with PID 3155 and log
    `runs/bioemu_pilot_200/smoke_retry2_console.log`.
  - A later GPU snapshot again shows no active CUDA process. This remains
    inconclusive while AlphaFold2 weights, MSA, or embeddings may be prepared;
    requested PID and per-sample log checks rather than inferring from GPU idle.
  - The retry reached AlphaFold2 embedding setup, then failed because the AutoDL
    proxy returned HTTP 500 for the official Google Storage AF2 parameter tarball.
  - Next: disable the proxy, verify direct Google Storage access, back the fixed
    `~/.cache/colabfold` path with the data disk, prefetch AF2 weights, and rerun
    smoke with the already cached Hugging Face checkpoint in offline mode.
  - Direct single-stream AF2 download works but only at about 34.3 KB/s, implying
    a roughly 30-hour transfer for 3.47 GB and unacceptable billable idle time.
  - Switched the next retry to a resumable 16-connection `aria2c` download of the
    same official Google Storage artifact; do not resume smoke until extraction
    and the completion marker are verified.
  - The first aria2 invocation exited before download because it was launched
    from `/root`, so the relative `runs/bioemu_pilot_200` log path did not exist.
    Aria2 is installed; retry from the project root using absolute log/PID paths.
  - The AF2 archive subsequently downloaded and extracted successfully, but the
    next smoke attempt stopped on its first row before sampling because BioEmu
    queried the Hugging Face model listing and hit a connection timeout.
  - Diagnosed the reported `switch_to_esmflow_small_pilot` result as invalid:
    one row had return code 1 and the other five rows were never attempted.
  - Added automatic local BioEmu checkpoint/config discovery to the AutoDL
    launcher so a cached `bioemu-v1.2` model bypasses the remote listing call.
  - Changed feasibility handling so any failed or unattempted row authorizes
    only a retry of the same frozen six-row smoke, with no cost extrapolation
    from failed rows.
  - Added focused local-model and infrastructure-failure regressions; the BioEmu
    test module now passes 6 tests, and the AutoDL shell script passes Bash syntax
    validation.
  - Reran the frozen smoke with the cached Hugging Face model and extracted AF2
    parameters; all six rows passed and authorized only the 200-row pilot.
  - Completed the authorized pilot with 200/200 zero return codes, 198/200 strict
    usable samples, 5,683 valid conformers, and 3.062 actual GPU hours.
  - Downloaded the 17,469,312-byte result archive and checksum to the workstation;
    local SHA256 verification matches
    `f1c10dbba383c9df3bd62ac0614bb1cd87520d18834175883ba14acd63339cfc`.
  - Audited all 1,430 archive paths before extraction: zero unsafe/duplicate
    paths and exactly 200 trajectories, topologies, run statuses, and batch NPZs.
  - Confirmed the archived manifest/selection summary are byte-identical to the
    frozen local inputs, then extracted into an isolated staging directory.
  - Independently aligned all 200 manifest/report/directory/status rows, confirmed
    200 non-empty core output sets and 200 zero return codes, and identified the
    two strict-quality failures as `peptide_9` (RMSD just below 0.05 nm) and
    `peptide_876` (short-peptide radius just above 1.0 nm).
  - Imported 72,340,756 bytes of verified raw results and four runtime
    reproducibility records into `runs/bioemu_pilot_200/`.
  - Replaced the tracked pending decision with the completed pilot evidence while
    preserving `full_generation_authorized=false`.
  - Updated README with the real AutoDL environment, infrastructure recovery,
    200-row metrics, two strict-quality exceptions, runtime/storage projection,
    archive integrity, and the frozen no-full-generation boundary.
  - Final BioEmu verification passed: archive/import assertions, Bash syntax,
    Python compilation, Ruff format/check, six focused BioEmu tests, and a
    40-test related regression suite with three known Transformer warnings.
  - Completed Phases 20A and 21. Returned to Phase 19, the deferred train-only
    length/generator-source OOF diagnostic, as the next priority.
  - Committed the cached-model/runtime-failure fix as `63d274c` and the verified
    pilot import/README decision as `5156554`.
  - GitHub push retried after local completion but the HTTPS connection reset;
    local `main` is safely two commits ahead of `origin/main`. Continue Phase 19
    locally and retry one consolidated push after the next bounded phase.
  - Ran the fixed Phase 19 diagnostic over all 6,387 training OOF rows and 3,195
    conformer-covered rows; emitted `summary.json` and a 6,387-row audit CSV.
  - Added aggregate probability-divergence and corrected/newly-incorrect
    comparisons without changing any model threshold or weight; Ruff and the
    focused diagnostic test passed and outputs were regenerated.
  - A one-off reporting snippet had an unmatched bracket. No artifact was
    modified; replace it with simple sequential loops rather than repeating the
    failed expression.
  - Completed Phase 19: existing conformer source and length are perfectly
    confounded (Cramér's V 1.0); the balanced long-AF2 pilot remains negative
    versus 1D despite a small aggregate 31-50 joint advantage.
  - Added the fixed metrics, decision-disagreement transitions, machine-readable
    decision, and no-retuning boundary to README.
  - Opened Phase 22 to convert and evaluate the imported 200-row unified BioEmu
    cohort across all three length strata.
  - Installed fixed `mdtraj==1.10.3` into the existing analysis environment
    without changing NumPy/PyTorch, so filtered XTC trajectories can be read
    rather than substituting unfiltered raw batches.
  - Added a BioEmu XTC-to-ensemble-summary converter with explicit nm-to-Angstrom
    conversion and frame-count/sequence/C-alpha alignment checks.
  - Converter/cache tests passed (3 total), and all 200 trajectories converted
    into 200 summaries containing exactly 5,683 filtered conformers.
  - A test-expansion patch initially missed because Ruff had reformatted the
    target context. No file changed; reread the current test and applied a
    smaller exact patch.
  - A report-only inspection used the obsolete key `paired_feature_deltas` and
    raised `KeyError` after printing the new predictability results. No artifact
    changed; inspect the current `by_length` and `by_legacy_source` keys instead.
  - A combined permutation-test patch again missed Ruff-formatted test context.
    No file changed; split the source and test/progress edits into exact patches.
  - Added a deterministic 1,000-permutation toxicity-signal check that preserves
    canonical fold and length-stratum label counts; focused lint/tests passed.
  - Regenerated the 200-row source analysis with the frozen 1D OOF anchor. The
    BioEmu descriptor classifier corrects 9 anchor errors but introduces 57,
    while the same-sample generator fingerprint remains strong (BACC 0.870629).
  - Completed Phase 22 with `hold_bioemu_full_generation`; no external test
    labels, model weights, thresholds, or GPU jobs were touched.
  - The first final-suite command referenced a nonexistent
    `tests/test_bioemu_selection.py`, so pytest collected no tests in that
    invocation. Enumerated the actual BioEmu tests and reran the suite with the
    valid paths.
  - The broader related suite hit a pre-existing collection mismatch:
    `test_multisource_conformer_analysis.py` imports `_bucket`, which the current
    committed module does not expose. This is outside the Phase 22 changes;
    excluded that single incompatible test and reran the remaining related
    regression suite.
  - Final related regression suite: 47 passed with three known Transformer
    warnings. Ruff, JSON/CSV consistency assertions, CLI help, and Git
    whitespace checks also passed.
  - Opened Phase 23 at the user's request. Froze a single CPU-only bounded
    BioEmu residual, its cross-fitting protocol, and an anti-regression gate
    before computing any fused result; Test1/Test2 remain sealed.
  - Implemented the bounded-fusion CLI and a synthetic fold-integrity/
    anti-regression test. The structure classifier never consumes 1D OOF
    probabilities during fitting; the fixed 1D combination happens only after
    cross-fitted structure predictions are complete.
  - The first synthetic test omitted the 31-50 stratum, so reporting its empty
    subset emitted two NumPy mean warnings despite passing. Added the missing
    stratum to exercise the full frozen cohort layout and remove the warnings.
  - Ran the frozen Phase 23 candidate on all 200 rows. It changed one decision,
    newly misclassifying `peptide_800`; MCC decreased 0.009037 while AUPRC,
    Brier, and ECE changed only slightly in the favorable direction.
  - The primary and strict-usable sensitivity analyses agree. Froze
    `stop_bounded_bioemu_candidate`; no second weight, threshold, feature set,
    Test1/Test2 label, or GPU generation job was tried.
  - Final Phase 23 verification passed: Ruff, CLI help, JSON/CSV/frozen-formula
    assertions, Git whitespace, and 48 related regression tests with three
    known Transformer warnings.
  - Opened Phase 24 after the user clarified ownership of the 3D workstream.
    Scope is a standalone multi-conformer contact-graph representation plus
    paired source-consistency, with fusion left to the user's teammates.
  - An initial matched-baseline audit incorrectly unioned raw manifest
    `source_index` values and obtained 2,734 rows. Diagnosed AF2 indices as
    subset-local; Phase 24 will derive canonical indices from `original_id`
    and rerun the audit before freezing the performance gate.
  - Reran the matched audit with canonical IDs: 3,252 rows, matched 3D-v2 MCC
    0.764328/AUPRC 0.887914. Froze the Phase 24 graph architecture, fixed-epoch
    optimization, source-robustness measurements, replacement gate, and weaker
    experimental fusion-handoff gate before training.
  - Implemented the variable-length edge-gated contact encoder, fold-safe
    legacy/BioEmu sample builder, paired-view training objective, source-probe
    evaluation, OOF/checkpoint export, and full 6,387-row fusion handoff schema.
  - Added a synthetic end-to-end test covering paired folds, checkpoints,
    primary/BioEmu OOF views, full missing mask, and fixed-size embeddings.
  - First lint pass found three unused imports in the new standalone module;
    removed them before running the end-to-end test.
  - Synthetic end-to-end graph OOF test passed. The real-data preflight then
    resolved all 3,252 samples and all four source tensor variants with node
    shape `L x 31`, edge shape `L x L x 6`, global dimension 12, five complete
    folds, 143 pairs, and maximum length 50.
  - Completed fixed 15-epoch training for all five folds on the local RTX 4060
    in about 78 seconds; losses were finite and declined in every fold.
  - Candidate failed both frozen gates: MCC 0.594800, AUPRC 0.818271, zero fold
    wins, and source-probe BACC 0.814685. Paired cosine/probability-delta gates
    passed, but no replacement or experimental fusion handoff is authorized.
  - Froze `stop_multiconformer_graph_candidate` without a second architecture,
    consistency weight, threshold, or Test1/Test2 inspection. Opened Phase 25 to
    package the already retained strong 3D stream for the fusion teammates.
  - Implemented the Phase 25 handoff packager and a synthetic alignment test.
    It preserves all retained probabilities, joins only by canonical index,
    exports conformer missingness separately from base-structure availability,
    and records SHA256 provenance plus explicit no-retuning rules.
  - Generated the real 6,387-row 3D team package. Recomputed fixed metrics match
    the retained decision exactly (MCC 0.839018/AUPRC 0.928171); counts are
    3,195 conformer-available and 3,192 explicit fallbacks.
  - Added a teammate README and strengthened the contract to require nested OOF
    rather than allowing a biased meta-model fit/score on the same OOF table.
  - Completed Phase 25 without changing any 3D probability, threshold, model,
    Test1/Test2 artifact, or fusion parameter.
  - Regenerated the handoff contract after the leakage-rule clarification.
    Verified all 6,387 folds and labels, zero numerical probability drift,
    3,195 conformer-available rows, and 3,192 explicit conformer fallbacks.
  - Verified the stopped graph branch exports 6,387 x 32 embeddings with 3,252
    structure-available rows and preserves `external_test_labels_read=false`.
  - Final focused tests passed (2/2), Ruff and Python compilation passed, and
    the dependency-compatible regression suite passed 128 tests. Five legacy
    RDKit-bound smoke tests were deselected because neither local Conda
    environment contains RDKit; `MPLBACKEND=Agg` avoids the unrelated Qt crash.
  - The first module-help command used a login shell without `PYTHONPATH`, so
    the blocked PowerShell profile and src-layout import both failed. Reran with
    a non-login shell and explicit `PYTHONPATH=src`; CLI help passed.
  - The PyTorch environment also lacks a Ruff module; the existing standalone
    `D:\anaconda\Scripts\ruff.exe` completed the same four-file check.
  - An initial artifact assertion compared CSV probability strings exactly and
    rejected harmless formatting differences. Parsed numeric values instead;
    the maximum absolute probability difference is exactly zero.
  - User requested the next-stage priority order and asked whether retained
    3D-v2 could benefit from random seeds. Added planned Phase 26: first audit
    and reproduce the exact retained pipeline, then freeze a three-seed full-OOF
    equal-probability ensemble, evaluate it without best-seed selection, and
    regenerate the team handoff only if the preregistered gate passes.
  - The first combined 3D-v2 provenance inspection command had an unmatched
    PowerShell quote in its final `rg` pattern and did not run. Split the query
    into quote-safe file/README inspection and a separate fixed-string search.
  - User supplied the earlier nine-item 3D priority table for reassessment
    against completed Phases 19–25. Began a status crosswalk so completed,
    rejected, conditional, and still-actionable items are not mixed together.
  - Visual crosswalk confirms most original high-priority items are now closed:
    BioEmu-200 and diagnostics completed; full BioEmu remains blocked by the
    evidence gate; standalone contact graph, bounded residual/gating, and
    geometry-consistency paths did not qualify. The retained descriptor and
    handoff portions remain useful.
  - The only old architectural item not tested in its exact form is the
    Mac-Diff-inspired high-contact-neighborhood local 1D–3D reader. Classified
    it as a later fusion-team/new-validation ablation, not an immediate pure-3D
    task and not a reason to reopen failed global bidirectional coupling.
  - Started Phase 26 after user authorization. Work begins with a read-only
    provenance/data/runtime audit and a deterministic seed-42 reproduction
    launcher; seeds 123/2025 remain blocked until the reference gate passes.
  - The first Phase 26 artifact-audit command repeated the known PowerShell
    `foreach { ... } | Format-Table` parse failure and produced no audit data.
    Corrective rule: assign every loop result to a named array before piping;
    do not reuse the failed inline-loop pattern.
  - Corrected Phase 26 preflight found all ten required fold checkpoints and all
    processed/fold/OOF artifacts locally. The train CLI cannot override seed,
    so the implementation must generate per-seed config snapshots; it must also
    use tolerance-based reproduction because CUDA determinism is not enforced.
  - Froze seeds `42/123/2025`, arithmetic probability averaging, exact alignment
    requirements, numerical reproduction tolerances, and aggregate/fold/length/
    calibration promotion limits before launching any Phase 26 training.
  - Initial synthetic aggregation test failed only the frozen ECE tolerance:
    its supposedly improved probabilities were less calibrated than the
    reference despite correcting decisions. Kept the production gate unchanged
    and corrected the synthetic fixture to be both more accurate and calibrated.

  - Phase 26 focused tests now pass (`2 passed`). The first Windows-wrapper
    parser audit then found an invalid interpolated variable reference in the
    error message (`$LASTEXITCODE:`). Delimited the variable as
    `${LASTEXITCODE}` and flattened checkpoint argument arrays explicitly before
    rerunning the cross-platform entry-point checks.
  - Real Phase 26 preparation verified and hashed every required input plus all
    five frozen control checkpoints. Its first reference summary exposed a
    float32-only AUPRC drift (`0.927682` versus retained `0.928171`); traced this
    to artificial ranking ties inside the training-oriented metric helper and
    switched Phase 26 to the handoff's exact float64 metric implementation
    before regenerating the protocol.
  - Regenerated protocol now exactly matches the retained 3D handoff baseline:
    MCC `0.839018`, AUPRC `0.928171`, Brier `0.05939949`, and ECE-10
    `0.05110903`. Marked the provenance audit and experiment freeze complete.
  - Local execution preflight found an idle RTX 4060 Laptop GPU with about
    6.3 GB free VRAM and 107 GB free on drive D. No seed-42 Phase 26 outputs
    exist yet. Windows/Python checks pass; local Bash is unavailable, so the
    AutoDL shell entry point still requires a Linux-side `bash -n`/prepare run.
  - Direct invocation of the new `.ps1` was blocked by the machine's execution
    policy. Used a process-scoped `powershell.exe -ExecutionPolicy Bypass`
    launcher without changing the system policy; the real prepare stage passed.
  - Started the complete seed-42 reproduction in the background with resume
    enabled. Base fold 0 loaded its frozen control checkpoint, exposed only the
    chemical-site branch, and completed epoch 1 without error; driver PID 32368
    and Python PID 25452 were active at the first health check.
  - While seed 42 trains, audited the aggregate report against the frozen Phase
    26 checklist. Seed/fold/length/calibration fields exist, but generator
    cohorts, decision-disagreement counts, and artifact-based runtime/storage
    still need implementation before final aggregation.
  - An optional inspection command ended with code 1 because historical fold
    directories contain `history.json` rather than the queried `summary.json`.
    No artifacts were changed; use config/history timestamps for runtime spans.
  - Added generator-cohort, decision-disagreement, per-seed subgroup, and
    artifact runtime/storage fields to the Phase 26 aggregate report plus an
    exact float64-ranking regression test. The first expanded test run passed
    but revealed single-class synthetic cohorts and one Ruff formatting drift;
    adjusted only the test fixture and formatted the new module.
  - The corrected Phase 26 report suite passes without warnings (`3 passed`),
    and the focused multi-seed/handoff/conformer/OOF compatibility suite passes
    `9/9`. README now documents the frozen protocol plus Windows and AutoDL
    commands. Seed-42 base fold 0 reached epoch 10 with a valid best checkpoint
    while these CPU-only checks ran.
  - Git boundary audit found the two new launchers were hidden by the repository's
    scripts allowlist, while fold configs and driver logs were unintentionally
    visible as untracked files. Updated `.gitignore` to retain both launchers and
    the compact frozen protocol/reports while excluding per-fold weights and
    runtime logs.
  - Ran the final aggregate CLI against three identical copies of the retained
    6,387-row OOF as a production-schema dry run. It emitted all four generator
    cohorts, 100% seed-decision agreement, zero reference disagreements, and the
    expected non-promotion (`mcc_strictly_higher=false`). This confirms the
    final gate cannot promote an unchanged ensemble.
  - WSL syntax validation was unavailable because the Windows host has no Linux
    distribution installed. Located Git for Windows Bash instead; the AutoDL
    launcher passes a real `bash -n` syntax check. No system installation or
    policy change was made.
  - Seed-42 base fold 0 early-stopped normally after epoch 19 and saved both
    checkpoint and history; the wrapper automatically advanced to base fold 1.
    At handoff, fold 1 had reached epoch 4 and the background driver remained
    healthy. No new seed is authorized until the later 6,387-row reference gate.

### GitHub Project Snapshot
- **Status:** complete
- Actions taken:
  - User requested that the current engineering state be uploaded to the existing
    `https://github.com/Suiyumanbu/GHXTox.git` remote on `main`.
  - Audited 246 staged paths. The largest file is 5.4 MB, and strict GitHub/Hugging
    Face token-prefix scanning found no credentials.
  - Confirmed the apparent AWS-key matches are false positives inside peptide
    sequences.
  - Ran 38 focused BioEmu, OOF-diagnostic, fusion, conformer, and adapter tests;
    all passed.
  - Fetched remote commit `907acc9`, resolved the seven overlapping integration
    conflicts, and reran the same focused suite; all 38 tests still passed.
  - Rebased the engineering snapshot without force and pushed commit `bf1f1e2`
    to remote `main`.
  - Verified local `main` and `origin/main` have zero divergence and resolve to
    the same commit.
  - Next: await completion and inspect the regenerated feasibility JSON before
    any 200-row authorization.
  - Next: import `smoke_feasibility.json` and the failing sample logs. Keep the
    200-row BioEmu command blocked unless a diagnosed, recoverable infrastructure
    error is fixed and the same frozen six-row smoke is rerun successfully.

## Session: Phase 26 AutoDL Result Import

### Phase 26: Preregistered 3D-v2 Multi-Seed Stability and Ensemble
- **Status:** complete
- Actions taken:
  - Located the user-downloaded AutoDL result archive and checksum in
    `D:\dachuang2026\ziliao` by timestamp around 12:00.
  - Verified SHA256 for
    `ghxtox_3d_v2_multiseed_autodl_results_20260801.tar.gz`; computed hash
    `29e14673605efb208838e6454261a03d59d6377f5d47eef0d02f2cd0dc30b666`
    matches the downloaded `.sha256` file.
  - Inspected archive paths and extracted the normal Linux layout into
    `runs/3d_v2_multiseed/`, adding `seed_123`, `seed_2025`, `ensemble`, and
    AutoDL logs beside the local `seed_42` reproduction.
  - Verified all three residual OOF files contain exactly 6,387 rows.
  - Confirmed seed 42 reference reproduction passed, and seeds 123/2025 completed
    the same full retained 3D-v2 plus capability-aware conformer-residual path.
  - Applied the frozen promotion gate to the imported ensemble: reference
    MCC/AUPRC `0.839018/0.928171`, ensemble `0.835649/0.927630`; Brier/ECE-10
    improved to `0.058648/0.048849`, but MCC, AUPRC, and the 11-15 aa length
    stratum failed the predeclared promotion criteria.
  - Updated README, findings, and task plan to retain the Phase 25 seed42 3D
    handoff and close Phase 26 without threshold retuning, best-seed selection,
    handoff regeneration, or Test1/Test2 access.
  - Updated `.gitignore` so AutoDL runtime logs remain local while compact JSON
    summaries stay visible for version control.

### Phase 26 Import Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-08-01 | PowerShell rejected Linux heredoc syntax `python - <<'PY'` during a read-only JSON check | 1 | Reran the check with a PowerShell here-string piped to Python |
| 2026-08-01 | Planning-file patch failed because console mojibake hid the real UTF-8 em dash | 1 | Printed `unicode_escape` for the affected lines and applied the exact UTF-8 patch |
# Session: Phase 27 Physical-Chemical 3D Upgrade

## 2026-08-01
- User authorized continuing the 3D work in the previously recommended order.
- Restored `task_plan.md`, `progress.md`, and `findings.md`; ran the planning
  session catch-up helper and inspected the dirty worktree.
- Confirmed the uncommitted Phase 26 documents and compact JSON evidence are
  present and must be preserved while Phase 27 is added.
- Started Phase 27 with P0 input/code audit and protocol freeze. Test1/Test2
  remain sealed and the retained Phase 25 MCC/AUPRC reference is unchanged.
- The first combined artifact audit encountered an optional `rg` no-match exit
  code. Logged it and switched to independent, no-match-tolerant queries.
- A later read-only inventory accidentally reused the repository's known
  PowerShell `foreach { } |` parse trap. It failed before execution and changed
  no files; all subsequent inventories use independent commands.
- Confirmed exact raw PDB availability by retained source: AF2-16-30 `2034`,
  AF2-31-50 `400`, and PepFlow `761`. Full heavy atoms and multiple models are
  available, so Phase 27 can proceed without new structure generation.
- Dependency audit found no project-level SASA/DSSP package. Froze the design
  preference for deterministic pure-NumPy residue/atom geometry descriptors.
- Validated manifest uniqueness before protocol freeze: AF2-16-30 has 2,034
  unique samples/sequences across 16,272 conformer rows; AF2-31-50 has 400
  across 3,200 rows; PepFlow has 761 across 6,088 rows.
- Wrote `runs/physical_3d_descriptors/protocol.json` before candidate metrics
  were computed. It freezes exactly 20 descriptors, per-source outer-fold
  standardization, L2 logistic residual settings, residual weight `0.15`, fixed
  threshold `0.677819`, missing fallback, and the full promotion gate.
- Implemented `src/ghxtox/physical_3d_descriptors.py`: deterministic 20-feature
  conformer extraction, canonical manifest alignment, source-wise outer-fold
  standardization, one fixed L2 residual candidate, raw/normalized source
  probes, cohort gates, and compact provenance outputs.
- Added focused descriptor/fallback tests and the
  `ghxtox-physical-3d-descriptors` CLI entry point. Candidate OOF metrics have
  not yet been run or inspected.
- First parallel verification collection was interrupted because
  `D:\anaconda\envs\pytorch_env` has no Ruff module. No code/data changed; the
  compile and tests are being rerun independently so their results are visible.
- The independent pytest rerun reached collection but lacked `PYTHONPATH=src`,
  so `ghxtox` was not importable. Logged the command-environment issue and will
  rerun with the repository-standard source path.
- With the source path fixed, two of four focused tests initially failed. One
  test assumed compact geometry must increase an *exposed* patch; the other
  report path did not skip a synthetic empty length cohort. Corrected both
  without changing any frozen descriptor definition or candidate setting.
- Pre-OOF verification now passes: four focused tests, Python compilation,
  Ruff, protocol JSON audit, and `git diff --check`. The frozen protocol still
  reports 20 descriptors, residual weight `0.15`, and no external labels read.
- Executed the one frozen real-data run over 3,195 conformer summaries in 87
  seconds on CPU. It produced `descriptors.csv`, `oof_predictions.csv`,
  `summary.json`, and `decision.json` under
  `runs/physical_3d_descriptors/`.
- Result: retain Phase 25. MCC changed `0.839018 -> 0.838166`, AUPRC
  `0.928171 -> 0.928202`, Brier `0.059399 -> 0.059270`, and ECE-10
  `0.051109 -> 0.051295`. Eight decisions changed (3 corrected, 5 regressed).
- Source standardization reduced the held-fold source probe BACC from
  `0.917099` to `0.333333`, but the 11-15 aa MCC loss (`0.650427 -> 0.643650`)
  exceeded tolerance. No post-hoc rerun is authorized.
- Documented the complete Phase 27 protocol, metrics, cohort failure, stop
  decision, and reproduction command in README C.15. The retained Phase 25
  handoff remains unchanged.
- Started Phase 28 P0 as a separate prospective all-atom surface-chemistry
  short-peptide specialist. The immediate task is only SASA dependency,
  heavy-atom consistency, and runtime feasibility; no new OOF has been viewed.
- Benchmarked Biopython Shrake-Rupley on representative PepFlow PDBs: all have
  eight models; after deterministic hydrogen removal, 100-point SASA takes
  about `0.0085-0.0132` seconds per model. Estimated 761-row extraction is only
  one to three CPU minutes, so the all-atom short specialist is feasible.
- Froze `runs/surface_3d_short_specialist/protocol.json` before OOF. It fixes
  heavy-atom 100-point Shrake-Rupley SASA, exactly 12 surface descriptors, only
  the 761 PepFlow short rows, L2 `C=0.1`, residual weight `0.10`, threshold
  `0.677819`, exact fallback for all other rows, and strict global/short gates.
- Implemented the Phase 28 all-atom surface extractor, deterministic PDB/model
  alignment, SASA/patch/moment descriptors, short-only outer-fold residual,
  strict fallback/gates, provenance outputs, focused tests, and CLI entry point.
  Real Phase 28 OOF has not yet been run or inspected.
- First Phase 28 synthetic run found only a Biopython three-letter residue-map
  key case mismatch (`ALA` versus `Ala`). Added a version-compatible uppercase
  lookup with title-case fallback; the frozen surface contract is unchanged.
- Phase 28 pre-OOF verification passes: six focused tests, Python compilation,
  Ruff, protocol audit, and `git diff --check`. Protocol remains frozen at 12
  descriptors, 761 target rows, eight models each, residual weight `0.10`, and
  no external labels read.
- Executed the one frozen Phase 28 run in 101.5 CPU seconds. All 761 target rows
  have eight-model surface descriptors and all 5,626 fallback rows are exactly
  unchanged.
- Phase 28 decision: retain Phase 25. Overall MCC changed
  `0.839018 -> 0.838649`; short MCC `0.650427 -> 0.645344`; short AUPRC
  `0.692577 -> 0.693980`; overall Brier/ECE improved slightly. One decision
  changed, with `0` corrections and `1` regression. No rerun is authorized.
- Final Phase 27/28 verification passes: six focused tests, Ruff, JSON decision
  audits, row counts (`3195/6387` and `761/6387`), exact Phase 28 fallback, and
  `git diff --check`. Compact protocol/summary/decision JSONs are visible to
  Git; large per-row CSVs remain ignored by existing repository rules.

# Session: Phase 29 End-to-End Short-Peptide 3D Specialist

## 2026-08-01

- User authorized the next prioritized 3D improvement after the Phase 27/28
  scalar-descriptor candidates failed the fixed-threshold MCC gate.
- Started Phase 29 with a leakage-safe end-to-end 11-15 aa specialist design.
  The current P0 task is to audit the existing threshold-neighborhood margin
  implementation and fold-specific 3D-v2 path before freezing a single new OOF
  candidate. Test1/Test2 remain sealed and Phase 25 remains the reference.
- The prospective boundary requires fold-specific training without held-fold
  access, exact Phase 25 fallback for all 5,626 non-short rows, the unchanged
  threshold `0.677819`, and no post-hoc epoch/head/margin/seed search.
- P0 search confirms the existing decision-margin implementation is confined
  to the cross-modal adapter, while `train.py` has no threshold-aware loss.
  Phase 29 can therefore add the control at the core 3D-v2 training level
  without repeating the earlier experiment.
- Verified all five fold-specific Phase 25 conformer checkpoints and their base
  chemical-site checkpoints. The prospective candidate will retain the same
  small conformer trainable prefix set; full 3D-v2 unfreezing is rejected as
  unnecessarily high-variance for 761 short rows.
- Audited the actual tensors: only the balanced pilot250 carries the retained
  conformer inputs; the additional 511 short PDBs are negative-only and are not
  authorized for renewed supervised fitting. Froze the candidate training
  cohort to pilot250 and the epoch budget to the retained fold median, 12.
- Confirmed the replacement scope: Phase 29 changes only 250 PepFlow pilot OOF
  rows inside the retained 6,387-row handoff. All remaining rows, including the
  other generator sources and non-generated fallback, must remain exact.
- Froze `runs/short_margin_3d/protocol.json` before computing any candidate
  OOF metric. It fixes the pilot250 cohort, five folds, seed 42, 12 epochs,
  four conformer-only trainable prefixes, margin weight/width `0.10/0.15`,
  positive multiplier `1.5`, threshold `0.677819`, input/checkpoint hashes,
  exact 6,137-row fallback, and the full promotion gate.
- Added the matching fixed configuration
  `configs/conformer_ensemble_short_margin_fixed12.json`. The candidate starts
  from the same fold-specific include-same-residue chemical-site checkpoints as
  the retained Phase 25 conformer branch; no candidate metric has been run.
- Implemented core-trainer asymmetric fixed-threshold margin support and a
  `fixed_epoch` checkpoint policy. Added `short_margin_3d.py` for frozen-input
  hash verification, alignment-safe 250-row replacement, exact fallback,
  global/pilot/fold metrics, decision transitions, and the preregistered gate.
- Added Linux/AutoDL and Windows five-fold launchers plus focused tests for the
  logit-boundary loss, frozen hashes, and replacement-only aggregation. The
  real candidate OOF still has not been run.
- Preflight passes: three Phase 29 tests, Python compile, Ruff, all frozen
  hashes, `git diff --check`, and Git-Bash syntax. A one-epoch fold-0 GPU smoke
  completed in 9.8 seconds on the local RTX 4060 with all 15 intended tensors
  trainable and no missing/unexpected checkpoint parameters.
- The measured smoke cost makes the complete frozen five-fold run economical
  locally (roughly 10-15 minutes including full OOF inference), so P3 is now
  authorized without using Test1/Test2 or changing any frozen setting.
- The first protocol verification correctly blocked before training because the
  finalized config byte hash differed from the hash captured immediately after
  file creation. No setting or candidate output changed; updated the frozen
  pre-candidate hash to the actual finalized bytes and will rerun checks
  independently.
- The first focused pytest command stopped at configuration parsing because the
  new console-script key appeared twice in `pyproject.toml`. Removed the
  duplicate identical line; test collection had not started and no experiment
  output was affected.
- Consolidated the restored Phase 29 work onto the original single output root
  `runs/short_margin_3d/`; removed the redundant post-compaction protocol copy.
  The canonical verifier now also hashes the exact config, full evaluation
  tensor, and canonical fold manifest. OOF evaluation explicitly disables
  threshold search and uses only `0.677819`.
- Executed the only frozen Phase 29 run on the local RTX 4060. Hash verification,
  five 12-epoch folds, full 6,387-row inference, and aggregation completed in
  173.1 seconds; every formal checkpoint records `epoch=12`.
- Phase 29 failed promotion. Global MCC changed `0.839018 -> 0.838611`, global
  AUPRC `0.928171 -> 0.928100`, pilot250 MCC `0.572184 -> 0.564547`, and
  pilot250 AUPRC `0.824843 -> 0.823187`. All 6,137 non-pilot probabilities are
  exact; fold 1 loses `0.038184` MCC and the only changed decision is a
  correct-to-wrong transition.
- Documented Phase 29 in README C.17. The canonical OOF evaluator records the
  externally frozen threshold `0.677819` and explicitly disables threshold
  search. Phase 25 remains retained; no post-hoc rescue is authorized.
- The first final focused-test rerun exposed only a compatibility issue in the
  hash-verifier test fixture: its minimal protocol intentionally omits the real
  `evaluation` block. Made those extra checks optional for fixtures; the real
  protocol includes and verifies both full-evaluation hashes.
- Final Phase 29 verification passes: 12 focused/regression tests, Python
  compilation, Ruff, Git-Bash launcher syntax, PowerShell verify entrypoint,
  all 11 frozen hashes, fixed-threshold OOF schema, five epoch-12 checkpoints,
  6,387-row aggregation, exact 6,137-row fallback, and `git diff --check`.

# Session: Phase 30 Existing-Data Multi-Conformer Pretraining

## 2026-08-01

- User confirmed that no new labeled or generated dataset is available and
  authorized the highest-priority existing-data 3D experiment.
- Opened Phase 30 with a strict non-duplication audit. The candidate must be
  label-free during pretraining, fold-pure during OOF evaluation, and distinct
  from the already rejected single-structure geometry SSL, ESMFold-anchor
  cross-view refinement, supervised Phase 24 contact graph, scalar residuals,
  seed search, and short-margin fine-tuning.
- Preserved the dirty worktree and all completed Phase 26-29 artifacts. Phase 25
  remains the immutable reference at MCC/AUPRC `0.839018/0.928171` and threshold
  `0.677819`; Test1/Test2 remain sealed.
- Initial code audit confirms `geometry_pretrain.py` already implements masked
  residue, pair-distance/contact, and ESMFold-to-one-conformer objectives. Phase
  30 therefore needs an ensemble-only two-conformer objective plus invariance on
  the 143 matched legacy/BioEmu pairs, not a rerun of that entrypoint.
- A read-only `rg` audit referenced three nonexistent expected filenames and
  returned code 1 after printing the valid `geometry_pretrain.py`/`train.py`
  matches. No files changed; subsequent discovery will enumerate actual paths
  with `rg --files` before querying them.
- Enumerated the actual reusable implementation and tests:
  `geometry_pretrain.py`, `multiconformer_graph_3d.py`,
  `models/dual_modal.py`, `data.py`, and
  `test_multiconformer_graph_3d.py`.
- Confirmed the prior Phase 24 model trained directly on toxicity labels after
  collapsing each ensemble to mean/std/occupancy edges. Phase 30 will instead
  sample two real aligned conformers, pretrain residue embeddings without labels,
  and use the retained Phase 25 score as the immutable base for one small
  fold-local residue head.
- Located all four existing conformer manifests and cache roots locally; no
  external acquisition or BioEmu regeneration is required.
- Direct schema inspection with the default Python failed because that
  interpreter has no PyTorch. Repeated the read-only inspection with the
  documented `pytorch_env`; no package was installed and no file changed.
- Verified every source summary exposes aligned per-conformer C-alpha arrays:
  PepFlow and both AF2 sources contain eight conformers per sample, while
  BioEmu contains the filtered variable count (30 in the inspected row).
  Pair-distance mean/std and 6/8/10 A occupancy targets are also present.
- Froze `runs/ensemble_domain_ssl_3d/protocol.json` before any Phase 30
  toxicity OOF metric was computed. It fixes four SSL epochs, a 64-dimensional
  two-layer encoder, at most eight conformers per source, matched-source weight
  `0.25`, a frozen encoder, one 15-epoch bounded residue head, maximum absolute
  delta `0.35`, threshold `0.677819`, exact fallback, input hashes, and all
  promotion/source-robustness gates.
- The protocol explicitly disables an all-source adversarial classifier and
  restricts invariance to the 143 same-sequence pairs, preventing source/length
  confounding from being optimized away as if it were nuisance-only variation.
- An optional wildcard audit used a Windows-invalid `runs/geometry_*` path.
  The valid parser defaults were already printed; explicit geometry summary
  paths were then read successfully and no artifact changed.
- Added the Phase 30 implementation module
  `src/ghxtox/ensemble_domain_ssl_3d.py`. Its first syntax audit found two
  JSON-style lowercase booleans in a Python decision dictionary; both were
  corrected to `False`, and `py_compile` now passes in `pytorch_env`.
- Batched every sample's conformers into one encoder call, replacing the
  initial per-conformer loop without changing the frozen representation
  definition. Added Windows and AutoDL launchers plus the package entry point.
- One read-only test-helper audit named the nonexistent
  `conformer_ensemble.py`; the actual helper is in `conformer_cache.py` and no
  artifact was changed by the failed command.
- Added a synthetic two-fold end-to-end Phase 30 test covering 12 paired
  ensemble rows and two baseline-only rows. It passed on CPU in 8.56 seconds
  and verified fold execution, paired diagnostics, and probability-exact
  fallback. `pytorch_env` lacks the Ruff module, so linting uses the already
  installed standalone Ruff executable instead of changing the environment.
- Ruff semantic checks pass. Ruff initially reported only formatting drift in
  the new module; the mechanical formatter was applied and both source and test
  now pass `ruff check` and `ruff format --check`.
- The first production preflight attempted to build all 16-dimensional
  per-conformer feature tensors and exceeded the 120-second command limit. It
  was terminated before training and wrote no Phase 30 candidate artifact.
  Frozen hashes had already passed; schema validation is now separated from the
  one-time cached feature construction in the resumable formal process.
- Fast production schema validation passed: all 3,452 existing generator views
  across 3,252 canonical structure-covered samples expose the required aligned
  coordinates and ensemble targets; zero summaries are malformed. The frozen
  cohort remains 3,195 legacy, 200 BioEmu, and 143 matched pairs.
- Started the single frozen five-fold Phase 30 OOF job on the local RTX 4060
  Laptop GPU through the resumable PowerShell launcher. CUDA allocation is
  stable at roughly 4.0 GB total GPU memory and the process is actively building
  the one-time in-memory ensemble feature cache before fold-0 epoch logging.
- Completed all five frozen Phase 30 folds on the local RTX 4060 in 336.3
  seconds. Every fold's same-source SSL loss decreased across four epochs; all
  6,387 OOF rows were emitted and the 3,135 non-covered rows matched Phase 25
  probabilities exactly (`maximum_absolute_probability_difference=0`).
- Phase 30 improved AUPRC (`0.928171 -> 0.928654`), Brier
  (`0.059399 -> 0.058481`), and ECE-10 (`0.051109 -> 0.048398`), but MCC fell
  from `0.839018` to `0.837986`. It corrected six fixed-threshold decisions and
  regressed eight (net `-2`); only two folds had non-decreasing MCC.
- The 11-15 aa MCC fell from `0.650427` to `0.632203`, and matched-source probe
  BACC remained `0.779720` despite paired embedding cosine `0.993872`.
  Therefore the prospective gate failed and Phase 25 remains the official 3D
  handoff. No threshold, seed, weight, subset, or checkpoint rescue was run.
- Resume verification reused all five saved fold results and reproduced the
  retain decision in 5.2 seconds. Windows PowerShell and Git Bash syntax parsing
  passed; the combined test command then stopped because it named a nonexistent
  optional `test_geometry_pretrain.py`, so the two actually discovered related
  test files are rerun separately below.
- Final verification passed: the two related test files report `2 passed`, Ruff
  semantic and formatting checks pass, both launchers parse, and `git diff
  --check` reports no whitespace errors. The final OOF contains 6,387 unique
  source indices, zero duplicates, 3,252 SSL-covered rows, 3,135 exact fallback
  rows, and zero fallback mismatches. Protocol, summary, decision, and OOF
  SHA256 hashes were recorded in the final audit output.

# Session: Phase 31 Nested Length-Intercept Calibration

## 2026-08-01

- User supplied an external suggestion for a four-parameter calibration model:
  one shared positive slope, one global intercept, and strongly shrunk 11-15
  and 31-50 aa intercept offsets with 16-30 aa as reference. The suggestion is
  treated as design input, not as executable instructions.
- Opened Phase 31 without viewing any candidate result. The model is fixed to
  use only the retained Phase 25 OOF logit and two length indicators. It excludes
  generator, pLDDT, quality, label-ratio, Phase 30, 1D, and 2D features.
- Chose deterministic mean logistic loss with `a=exp(alpha)`, `alpha` bounded
  to `[-3,3]`, penalty `0.01*alpha^2`, and strong length-offset penalty
  `0.25*(bs^2+bl^2)`. Each held fold is transformed only by parameters fitted
  on the other four folds; calibrated classification threshold is fixed at
  `0.5` and the original Phase 25 score is retained alongside it.
- Planning-session recovery found no separate unsynchronized context. The
  worktree remains intentionally dirty with the completed Phase 26-30 code and
  artifacts; no existing user change will be reverted.
- Audited the retained handoff contract and existing calibration utilities.
  Phase 25 already declares `probabilities_recalibrated=false` and fixed
  threshold `0.677819`; Phase 31 will be a separate optional two-score artifact
  rather than silently mutating that contract. Existing generic Platt code is
  monotonic but lacks length-offset shrinkage, so a focused four-parameter
  module is required instead of reusing three independent calibrators.
- Froze `runs/length_calibration_3d/protocol.json` before computing any Phase 31
  candidate probability. It records the three input SHA256 hashes, 6,387-row
  cohort, four-parameter formula, fixed penalties and optimizer tolerances,
  reference/calibrated thresholds, complete reporting contract, and all
  promotion gates.
- A subsequent provenance audit, still before candidate computation, found the
  handoff's explicit warning that splitting an already generated OOF table is
  not a fully nested meta-model evaluation: Phase 25 predictions in the four
  fitting folds may come from upstream models trained with the current held
  fold. The available `train_nested_groupcv` roles do not include strict nested
  Phase 25 predictions. Phase 31 is therefore downgraded prospectively to an
  exploratory screen; passing gates may authorize expensive strict nested base
  retraining, but can never directly overwrite the official handoff.
- Added `length_conditioned_calibration_3d.py` with analytic gradients,
  deterministic L-BFGS-B fitting, positive shared slope, fold-pure parameter
  application, complete metric/gate reporting, and separate raw/calibrated
  handoff columns. Syntax and Ruff semantic checks pass; Ruff requested only a
  mechanical format pass before testing.
- Added Windows/AutoDL launchers, the package entry point, tracked compact CSV
  exceptions, and two focused tests. Both tests passed, including the direct
  proof that changing one held fold's labels leaves that fold's calibrated
  probabilities unchanged. Ruff semantic checks passed; only the test file
  required mechanical formatting.
- Final pre-candidate verification passed: both tests report `2 passed`, Ruff
  semantic/format checks pass, PowerShell and Bash launchers parse, all frozen
  input hashes match, and all 6,387 rows align with the canonical folds. The
  verifier confirms this remains a non-promotable exploratory screen and that
  Test1/Test2 labels are sealed.
- Executed the one frozen 6,387-row Phase 31 CPU screen in 4.9 seconds. All five
  L-BFGS-B fits converged in 11-12 iterations with positive slopes
  `0.3413-0.4227`; strong shrinkage kept the short/long offsets close to zero.
- Calibration improved Brier from `0.059399` to `0.056150` and ECE-10 from
  `0.051109` to `0.018028`, but MCC fell from `0.839018` to `0.835319` and
  AUPRC from `0.928171` to `0.924213`. It changed 27 decisions, corrected 8,
  and regressed 19 (net `-11`).
- The short layer again failed materially (`MCC 0.650427 -> 0.632614`); middle
  and long MCC also fell to `0.779625/0.888832`. Fold 0 improved, folds 3/4
  tied, and fold 1 fell by `0.016927`. The advance gate failed for global MCC,
  global/length AUPRC, short MCC, and net corrections. Phase 25 remains
  untouched, and strict nested upstream retraining is not authorized.
- Deterministic rerun reproduced byte-identical SHA256 hashes for summary,
  decision, both OOF/handoff CSVs, and fold coefficients. The two-column handoff
  has 6,387 unique IDs, exact fold/label/sequence alignment, and zero raw Phase
  25 probability mismatches; the official Phase 25 OOF hash remains
  `4a241180743ad9e7c68165e636b67f24dfb6751de07ebc4fbe6241c2d7845fda`.
- Final verification reports `6 passed` across the Phase 31 and generic
  probability-calibration tests. Ruff, Python compilation, PowerShell parsing,
  Bash parsing, and `git diff --check` all pass. Phase 31 is complete.
## 2026-08-01 - Phase 32 partial-unfreeze 3D evidence started
- Restored the persistent project context and audited the five Phase 30
  fold-pure encoder checkpoints before making a candidate prediction.
- Confirmed the two-layer encoder state schema and selected only
  `encoder.layers.1` for supervised unfreezing; input projection, RBF features,
  layer 0, final encoder norm, and both SSL decoders remain frozen.
- Froze `runs/partial_unfreeze_3d/protocol.json`, including all input/checkpoint
  hashes, fixed eight-epoch training, two stochastic training conformers, eight
  inference conformers, two optimizer learning rates, the held-row-only bounded
  fusion coefficient, exact fallback, and the complete promotion gate.
- No Phase 32 candidate probability has been computed yet. Test1/Test2 remain
  sealed and Phase 25 remains the official 3D handoff.
- Implemented the packed multi-conformer trainer, standalone evidence head,
  exact held-row-only fusion, post-fine-tune source diagnostics, fixed final
  checkpointing, resume support, Windows/AutoDL launchers, and two focused
  tests. The focused suite reports `2 passed`; Ruff, compilation, and both
  launcher parsers pass.
- The formal verifier confirms 3,252 ensemble rows, 6,387 reference rows, folds
  0-4, exact frozen hashes, and no external-test access. Candidate OOF execution
  is now authorized under the frozen protocol.
- The first formal launch was stopped during fold 0 because every epoch loss was
  NaN; it had not written a prediction or checkpoint. A real-batch backward
  audit showed finite inputs/forward values but non-finite gradients in the
  trainable encoder layer. The cause is the infinite derivative of `sqrt(0)` in
  exact zero conformer-variance channels, not corrupt coordinates or features.
  A fixed `1e-6` numerical epsilon was added under the square root, and restart
  is blocked until finite-gradient regression checks pass.
- The fixed numerical regression and a real-batch backward audit passed, after
  which the same frozen experiment restarted from empty Phase 32 fold outputs.
  The successful five-fold RTX 4060 run completed in about 267 seconds.
- The candidate reaches MCC/AUPRC `0.838129/0.928231` versus Phase 25
  `0.839018/0.928171`; Brier/ECE improve to `0.059289/0.050691`, but ten
  decisions change with four corrections and six regressions (net `-2`).
- Three folds are non-decreasing and all tolerance/source gates pass, but the
  required global MCC improvement and positive net correction gates fail. All
  three length-stratum MCC values decline. Decision:
  `retain_phase25_3d_handoff`; no handoff regeneration is authorized.
- A resume-only rerun reproduced byte-identical summary, decision, and 6,387-row
  OOF CSV hashes. Final alignment/finite-value/checkpoint-scope audit passes,
  exact fallback is `3,135/3,135`, and the focused regression set reports
  `5 passed` with Ruff and `git diff --check` clean.
## 2026-08-01 - Packaged useful 3D outputs for handoff
- Created `D:\dachuang2026\ziliao\GHXTox_3d_useful_outputs_20260801_154826`
  and `D:\dachuang2026\ziliao\GHXTox_3d_useful_outputs_20260801_154826.zip`
  with a SHA256 sidecar.
- Included the official Phase 25 3D team handoff predictions/quality table as
  the recommended downstream fusion input, plus compact Phase 26-32 evidence
  directories documenting the non-promoted 3D attempts.
- Excluded fold checkpoints, logs, caches, and large server archives. The
  package README explicitly warns that Phase 32 is evidence only and that the
  retained 3D probability column remains
  `runs/three_d_team_handoff/oof_3d_predictions.csv::toxicity_probability`.

## 2026-08-01 - Phase 33 boundary BioEmu specialist started
- User authorized the next 3D engineering phase after reviewing the ranked
  bottlenecks and the boundary-specialist proposal.
- Restored the persistent plan and audited the dirty worktree without reverting
  any existing Phase 26-32 changes. Phase 25 remains the immutable reference at
  MCC/AUPRC `0.839018/0.928171` and threshold `0.677819`.
- Opened Phase 33 with a label-blind selection contract: cohort membership may
  use only retained OOF probability/logit distance, canonical fold, sequence,
  and length. It may not use toxicity labels, correctness, Test1/Test2, or any
  post-candidate subset search.
- The first deliverable is a verified, resumable AutoDL generation package. No
  Phase 33 toxicity model or candidate OOF probability will be produced until
  the frozen BioEmu cohort has been generated and imported successfully.
- Audited the completed BioEmu pilot implementation and preserved its hard
  200-row limit. Phase 33 will use a new dedicated selector/runner instead of
  broadening the old pilot CLI, and its server-facing manifest will contain no
  toxicity label column.
- Reaffirmed the pre-existing selection proposal before implementation:
  globally select exactly 600 rows by absolute Phase 25 probability distance
  to `0.677819`, breaking ties by `source_index`. Rejected newly inspected
  logit-distance and forced-fold-quota alternatives rather than choosing among
  them using post-selection label/error diagnostics.
- Added the initial Phase 33 implementation module and CLI. It freezes the
  600-row label-blind selector, verifies exact reusable BioEmu outputs, writes
  separate full/missing/reuse manifests, projects cost from the verified pilot,
  runs only a label-free server manifest behind explicit authorization, resumes
  completed sample directories, and assesses the generated cohort before any
  specialist training can be authorized.
- Kept the historical 200-row runner unchanged except for allowing its sample
  assessor to represent a missing label as `null`; this is required for the
  label-free Phase 33 server manifest and does not change any BioEmu geometry or
  feasibility criterion.
- The first parallel preflight reached Ruff semantic success but returned
  nonzero because the existing formatter requested mechanical layout changes
  in `boundary_bioemu.py` and the one-line optional-label edit in
  `bioemu_pilot.py`. No cohort or experiment artifact has been generated yet.
- Applied the mechanical formatter, then passed Python compilation, nine
  focused Phase 20/33 tests, Ruff semantic/format checks, Git-for-Windows Bash
  parsing, and `git diff --check`.
- Executed the single frozen selector against the retained 6,387-row Phase 25
  handoff. It wrote `data/bioemu_boundary_600/` with protocol, full/missing/reuse
  CSV/FASTA manifests, checksums, and selection summary. Exactly 26 selected
  rows reuse verified pilot outputs and 574 require AutoDL generation; projected
  new cost is 8.059 GPU hours and 0.206 GB.
- Added narrow `.gitignore` exceptions for the compact Phase 33 protocol,
  manifests, checksums, and launcher while explicitly keeping future raw server
  trajectories under `runs/bioemu_boundary_600_server/` ignored.
- A combined full-manifest dry-run/recursive TEMP cleanup command was rejected
  by the shell safety policy before execution. No file was created or removed;
  validation will be rerun non-destructively in a unique system TEMP directory.
- The non-destructive full-manifest dry run completed all 574 server rows,
  created 574 isolated TEMP status directories, emitted no label field, and
  verified that every command includes `bioemu-v1.2` and
  `filter_samples=True`. The temporary output is outside the repository and is
  not part of the handoff.
- Added a Phase 33 AutoDL README with extraction, explicit authorization,
  background/resume, completion-gate, and result-packaging commands.
- Created `D:\dachuang2026\ziliao\GHXTox_phase33_boundary_bioemu_autodl.zip`
  with 17 POSIX-style entries and SHA256
  `56501fbcfc97f4843756b6b20da953a93e29332504704f00b524ad5570f1f689`.
  Its sidecar hash verifies, all required entries are present, and there are no
  backslash entry names.
- Extracted the finished zip into a unique system TEMP directory and reran the
  complete 574-row label-free dry run from the extracted code and manifest.
  All 574 statuses were emitted successfully; this verifies that the handoff is
  self-contained for the AutoDL generation stage.
- AutoDL execution exposed a launcher fragility: the first formal run completed
  only three status records and the retry only five because the runner stopped
  at the first nonzero BioEmu sample return code. The local implementation now
  supports `--continue-on-error`, the Phase 33 AutoDL launcher enables it by
  default, and a regression test verifies that all manifest rows are visited
  while failed samples remain explicitly recorded for later repair.
- Focused verification after the fix passed: `tests/test_boundary_bioemu.py`
  reports `4 passed`, Ruff semantic checks pass for the changed Python files,
  and Git-for-Windows Bash parses `scripts/run_boundary_bioemu_autodl.sh`.
  The completed 574-row AutoDL result archive was verified by SHA256, copied to
  `D:\dachuang2026\ziliao`, imported into `runs/bioemu_boundary_600_server/`,
  and converted into a 600-row local BioEmu conformer cache under
  `data/bioemu_boundary_600/cache_600/`.
- Added the Phase 33 boundary specialist entry point. It freezes a unified
  BioEmu-only protocol, verifies all input hashes, trains one fold-pure
  partially-unfrozen 3D evidence head per outer fold, applies the fixed
  `0.15*tanh(logit)` residual only to the frozen boundary rows, and preserves
  exact Phase 25 probabilities everywhere else.
- The Phase 33 five-fold local run completed in about 69 seconds and passed the
  predeclared promotion gate. MCC improved from `0.839018` to `0.839758`, AUPRC
  from `0.928171` to `0.928204`, Brier from `0.059399` to `0.059378`, and
  ECE-10 from `0.051109` to `0.051094`. Four decisions changed: three
  corrections and one regression.
- Packaged the useful Phase 33 outputs, fold models, OOF predictions, protocol,
  import manifest, and source entry point into
  `D:\dachuang2026\ziliao\phase33_boundary_specialist_3d_20260802_075400.zip`
  with a SHA256 sidecar.
- Because the frozen promotion gate passed, generated a separate promoted team
  handoff at `runs/three_d_team_handoff_phase33/` without overwriting the
  Phase25 baseline handoff. Its `toxicity_probability` column reproduces the
  Phase33 candidate metrics byte-for-metric, and it marks the 600 boundary rows
  plus the 5,787 exact fallback rows with audit columns.
- Packaged the promoted handoff for teammates as
  `D:\dachuang2026\ziliao\three_d_team_handoff_phase33_20260802_075400.zip`
  with a SHA256 sidecar.
- Added Windows and AutoDL/Linux reproduce wrappers for the Phase33 boundary
  specialist and documented the promoted result in README. Verification passed:
  Ruff format/check, Python compilation, PowerShell `-VerifyOnly`, and Git Bash
  syntax checks.
## 2026-08-02: Phase 34 fusion audit and GitHub handoff

- Began a repository-wide audit of the promoted Phase33 3D handoff before
  documenting the 1D/2D/3D fusion contract and publishing it.
- Frozen review boundaries: strict nested/cross-fitted fusion for any reported
  OOF gain, immutable Test1/Test2 labels, exact Phase25 fallback outside the
  600-row BioEmu boundary cohort, and no post-hoc threshold/seed/subset search.
- Planned checks cover row/fold/label alignment, second-level OOF provenance,
  score/threshold semantics, missing-structure fallback, reproducibility,
  focused tests, and Git push scope.
- Initial README search confirmed that Phase33 is documented in `C.21`, while
  the repository also contains older generic stacking guidance that needs an
  explicit provenance warning for the new handoff.
- One combined inspection command returned exit code 1 because `rg` found no
  `AGENTS.md` match; this is a search-result status rather than a repository
  failure. The Git-state inspection is being rerun independently.
- Git audit: branch `main` tracks `origin/main` and starts six commits ahead;
  the worktree contains a large set of earlier Phase26-33 source, script, test,
  data, and result changes. No blanket staging will be used before size and
  relevance checks.
- The promoted handoff contract has the correct fixed threshold, row counts,
  fallback semantics, Test1/Test2 prohibition, and an explicit warning against
  same-table meta-model fitting/scoring.
- Hardened `src/ghxtox/oof_fusion.py`: ordinary second-level OOF screening is
  now labeled exploratory rather than strict nested; source-index uniqueness,
  fold agreement, binary labels, finite probabilities, and `[0,1]` bounds are
  validated; frozen application now reproduces either probability- or
  logit-space screening exactly.
- Extended `tests/test_oof_fusion.py` with logit-application equivalence and
  invalid fold/probability/duplicate-index regression cases.
- Hardened Phase33 resume artifacts with schema/phase/fold/seed, frozen
  protocol SHA256, exact fine-tuning config, held source-index set, row counts,
  finite evidence, and probability/logit consistency checks. New results write
  this metadata; incompatible legacy/stale results now fail closed.
- Added `tests/test_boundary_specialist_3d.py` for valid resume and stale
  metadata/cohort/evidence rejection.
- Added `src/ghxtox/phase33_team_handoff.py`, a deterministic promoted-handoff
  builder that verifies the authorization gate, all 6,387 identities/folds,
  600 boundary rows, 5,787 exact fallbacks, BioEmu availability, bounded-logit
  arithmetic, fixed-threshold predictions, and input/output hashes. It marks
  the artifact as train-OOF-only and not production/Test inference ready.
- Added focused Phase33 handoff tests for exact fallback and unauthorized or
  altered input rejection.
- Changed the Phase33 launcher scripts so resume is opt-in and the output root
  is selectable; this prevents pre-hardening/stale local fold artifacts from
  being trusted automatically.
- First focused test run: 17 passed and one new handoff test failed because its
  synthetic non-boundary row unnecessarily recomputed sigmoid and was not
  bit-exact. The production check correctly rejected it; the fixture was fixed
  to model the protocol's exact fallback.
- Second focused run passed all 18 tests and Ruff formatting/lint checks.
- First real-data deterministic handoff rebuild stopped at source index 12:
  recombining the Phase33 residual from the shorter-precision Phase25 handoff
  CSV differed from the higher-precision Phase33 candidate beyond `1e-10`.
  This is being quantified and will be handled with a serialization-only
  tolerance while exact non-boundary fallback remains bitwise enforced.
- Root cause was exact 0/1 fallback probabilities: bounded-logit recomputation
  intentionally clips them, so residual arithmetic is now verified only on the
  600 boundary rows while non-boundary rows retain the stronger exact-equality
  check.
- The corrected real-data dry rebuild completed with 6,387 rows, 600 BioEmu
  boundary rows, 5,787 exact fallbacks, and maximum absolute residual
  `0.114973 < 0.15`. Against the existing handoff, all audited identity,
  probability, threshold, and Phase33 fields matched within `5e-13`; MCC and
  AUPRC matched exactly.
- Regenerated the formal `runs/three_d_team_handoff_phase33` directory with the
  audited builder. The contract now includes output CSV hashes, OOF-only scope,
  production readiness=false, and strict nested fusion requirements while the
  retained MCC/AUPRC remain unchanged.
- Added README C.22 with the complete 1D/2D/3D alignment contract, strict
  outer-nested reporting path, predeclared low-degree convex-logit fusion and
  monotone calibration, Phase25-versus-Phase33 ablation gate, current OOF-only
  deployment boundary, and the algorithm fixes made in this audit.
- A combined README/diff inspection command hit a PowerShell parser error from
  piping directly after a `foreach` block. No repository action occurred; the
  size audit will collect objects into an array before sorting.
- Git publication audit found that global CSV/data ignore rules would omit the
  Phase33 OOF handoff and import manifest. Added narrow exceptions for compact
  Phase33 predictions/quality, conformer manifest/import checksums, and the
  server feasibility report; raw conformers and model checkpoints remain
  ignored. Added CLI entrypoints for boundary-specialist evaluation and
  deterministic Phase33 handoff generation.
- Full-suite verification stopped during collection because the pre-existing
  `tests/test_multisource_conformer_analysis.py` imports `_bucket`, which is
  absent from the current module. This is unrelated to Phase33 but blocks a
  clean repository test run; compatibility is being audited before push.
- Confirmed the multisource module already uses the canonical, directly tested
  `_length_bucket`; updated the stale test import/call rather than restoring a
  second bucket implementation.
- The unrestricted full `pytest -q` invocation exceeded the long command
  timeout; no child Python/pytest process remained afterward. The suite includes
  project-scale training/integration paths, so verification will be split into
  collected unit modules and explicit Phase26-33/regression subsets rather than
  claiming an unfinished all-suite pass.
- Running the 38 non-training test files completed: 159 passed and five old
  atom-graph smoke tests failed solely because `pytorch_env` lacks optional
  RDKit. A Windows Qt backend shutdown warning also appeared in the report
  figure test; it will be rechecked with `MPLBACKEND=Agg`. All fusion/Phase33
  tests in this run passed.
- Phase33 input/hash verification passed both direct CLI and PowerShell wrapper;
  both relevant Bash scripts passed `bash -n`.
- Full-tree Ruff reports 82 historical formatting drifts and 10 static findings.
  The audit will fix the real static issues (including duplicate dictionary
  keys) but will not mass-format unrelated legacy files; all newly changed
  fusion/Phase33 files remain individually format/lint clean.
- Removed the two duplicate model-config keys from `cross_modal_adapter.py` and
  eight unused/duplicate imports reported by Ruff. Duplicate values were
  identical, so this cleanup changes no model configuration or result.
- Final environment-appropriate suite passed: 161 tests passed and the five
  RDKit-dependent atom-graph tests were explicitly deselected because RDKit is
  absent from every local Conda environment. Both previously separated training
  tests passed individually. `MPLBACKEND=Agg` removed the Qt shutdown fault.
- Full-tree `ruff check src tests` and `compileall -q src` passed. Historical
  Ruff formatting drift remains intentionally untouched outside the files
  edited in this task.
- Publication review found no credential patterns, no files above 1.86 MiB,
  and no divergence from `origin/main`; raw conformers and model weights remain
  ignored. Commit `02c53d5` and the six preceding local engineering commits
  were pushed successfully to GitHub `main`.
