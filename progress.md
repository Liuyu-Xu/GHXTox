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
