# Findings & Decisions

## Requirements
- Follow the agreed priority order for algorithm improvements.
- First use a stronger conformer/multi-conformer 3D prediction stream.
- Add structure-quality-aware dynamic gating.
- Add anti-regression constraints and nested cross-validation.
- Preserve reproducibility and avoid using external test sets for tuning.
- The latest README Appendix C freezes the next engineering task as: current-OOF
  length/source diagnostics, a deterministic 200-sample train-only BioEmu pilot
  manifest, reproducible server scripts, smoke and feasibility assessment only,
  and no full-dataset generation yet.
- The new work must not read Test1/Test2 labels, retune ProtT5/structure weights,
  restart the rejected global bidirectional gate, or select conformers by toxicity.
- Appendix C defines three diagnostic strata exactly as 11-15, 16-30, and
  31-50 aa, and requires 1D/3D/joint OOF metrics, source/length predictability,
  label balance, pLDDT, conformer disagreement, and error-type comparisons.
- A BioEmu pilot continues only if at least 90% of sequences have at least eight
  valid conformers, all three length strata generate stably, there is no systematic
  chain break/residue mismatch/extreme collision, diversity is nonzero and
  physically bounded, and projected time/storage is feasible for 6,387 sequences.
- Existing reusable code includes cohort metrics, multi-source conformer analysis,
  and deterministic fold/label/length-stratified pilot selection, but no BioEmu
  implementation is present yet.
- The aligned train-only OOF sources are available for all 6,387 rows:
  `runs/plm_textcnn_prott5_bsmote_multiseed/oof_predictions.csv` (1D),
  `runs/3d_v2_default/oof_predictions.csv` (3D-v2), and
  `runs/prott5_structure_chemical_cross_attention_reproduced_oof/oof_predictions.csv`
  (retained joint model). They share sequence/label order and the latter two expose
  the group-aware fold.
- The full train fold manifest is
  `data/folds/train_cdhit080_fallback_folds.csv`; the 31-50 pilot manifest and
  PepFlow/AF2 manifests preserve `original_source_index`, allowing strict train-only
  joins without touching external-test files.
- The selected pilot can reuse `select_conformer_pilot.py`, but a 200-row allocation
  cannot be exactly uniform over 3 length bins x 5 folds x 2 labels; the new selector
  needs deterministic near-balanced quota allocation rather than silently changing
  the requested pilot size.
- Official BioEmu sources (checked 2026-07-31) show the Linux-only CUDA package is
  currently `bioemu==1.4.1` on PyPI and requires Python >=3.11. Core sampling is
  `python -m bioemu.sample`; the supported deterministic argument is `base_seed`,
  existing output directories resume to the requested total sample count, and the
  package bundles ColabFold/AF2 embedding retrieval.
- The pilot will pin package `1.4.1` but explicitly pin model checkpoint
  `bioemu-v1.2` as required by the README rationale. It will keep
  `filter_samples=True`; BioEmu documents that this removes steric clashes and
  chain discontinuities and may yield fewer outputs than requested.
- BioEmu outputs a topology PDB plus trajectory/sample files rather than this
  project's conformer-NPZ schema, so smoke assessment must first validate official
  output counts, topology sequence/residue count, diversity, elapsed time, and
  storage before any downstream conversion is authorized.
- Local conformer coverage resolves to four complete cache roots: PepFlow pilot
  250 plus remaining 511 (761 total), AF2 16-30 full 2,034, and AF2 31-50 pilot
  400. The first two generator sources have zero length overlap by construction,
  making generator identity and the 11-15/16-30 boundary perfectly confounded.
- A prior multi-source report already shows very different covered-cohort label
  ratios (PepFlow 111/761 positive; AF2 670/2,034 positive) and much weaker
  PepFlow structural MCC, supporting an explicit source/length diagnostic rather
  than another fusion experiment.
- The local workstation has an RTX 4060 Laptop GPU with 8 GB VRAM and CUDA 12.7;
  the existing PyTorch environment sees it correctly. BioEmu itself is Linux-only,
  and no AutoDL SSH host is configured locally, so execution requires either a
  CUDA-enabled WSL distribution or manual upload to the documented AutoDL path.
- The final frozen pilot contains exactly 200 train rows: labels 100/100, each of
  five group-aware folds 40, and lengths 11-15/16-30/31-50 at 67/67/66. Every
  length x fold x label cell contains six or seven sequences.
- The six-row smoke contains one positive and one negative from each length stratum.
  It is the only currently authorized GPU execution; the 200-row mode requires an
  explicit environment gate after a passing smoke decision.
- The server archive
  `bioemu_pilot_200_server_inputs_20260731.tar.gz` is 24,350 bytes with SHA256
  `857970762a4d58702c73faf3cae57b5d6a6f2708e0cdac188e06128b2224b957`.
- No WSL distribution, Docker, Podman, or configured AutoDL SSH host exists on this
  workstation, so real BioEmu outputs cannot be produced locally despite the
  available RTX 4060.

## Research Findings
- Existing quality-aware dynamic coupling improved OOF MCC from 0.859625 (1D) to 0.865344.
- The current complete default model remains stronger at approximately 0.8712 OOF MCC.
- The repository contains conformer residual OOF artifacts and calibrated fold checkpoints.
- The working tree contains unrelated generated outputs that must not be reverted.
- ProtT5, 3D-v2, and conformer residual OOF files all contain 6,387 aligned rows.
- The conformer residual OOF CSV has per-sample `mean_plddt`, `min_plddt`, `low_plddt_fraction`, `global_gate`, margin, entropy, and sequence-length columns.
- The previous coupler searched only one fixed hyperparameter configuration; it did not use an inner validation role for model/hyperparameter selection.
- The existing `nonnegative_stacking.py` already provides project-consistent nested role loading and retention gates that can be reused conceptually.
- The calibrated conformer stream is a small but statistically positive improvement over 3D-v2: MCC +0.002816 and AUPRC +0.000170 in the existing paired analysis.
- `data/folds/train_nested_groupcv.csv` already defines disjoint train, validation, calibration, and outer-test roles for all five folds.
- The current paper default is the ProtT5/structure/chemical cross-attention adapter with OOF MCC 0.871184, AUPRC 0.953631, Brier 0.044492, and ECE-10 0.025994.
- The default adapter summary records metrics but does not expose an aggregate prediction CSV suitable for row-level retention comparisons.
- Conformer quality signals are non-constant and well populated: mean pLDDT spans 0.282-0.974, low-pLDDT fraction spans 0-1, and global gate spans 0.231-0.429.
- The latest repository already contains `selective_3d_gate.py`, which implements train/validation/calibration/outer-test roles, monotonic quality features, and retention gates.
- The README records that conformer-residual quality gating, non-negative stacking, and signed structure residuals were already screened with strict nested CV and rejected: MCC 0.8614, 0.8633, and 0.8627 versus the frozen default 0.8693.
- Repeatedly tuning another sample-level fusion formula on the same OOF data would add selection freedom without new evidence.
- `cross_modal_adapter.py` already performs residue-level ProtT5-query to 3D key/value attention and supports per-residue chemical summaries, but it does not consume the available multi-conformer residue/global descriptors.
- `conformer_features.py` provides eight residue-level dynamics descriptors and twelve global ensemble descriptors with a hard `conformer_available` mask, which are genuinely new inputs rather than another probability-fusion formula.
- The prior OOF cross-modal feature cache `data/processed/train_prott5_structure_oof_adapter.pt` is currently absent locally, so production screening will require regeneration after the code path is implemented.
- All 2,795 multi-conformer ensemble summaries are present locally: 761 PepFlow peptides and 2,034 AF2/ColabFold peptides, covering 43.76% of training records.
- The frozen interaction baseline is exactly reproducible as `0.62 * ProtT5 + 0.38 * conformer-residual 3D` at threshold 0.60; recomputation matches MCC 0.869346 and AUPRC 0.953598.
- The combined processed artifact covers 2,795 of 6,387 samples (43.7608%) and contains all 2,795 available multi-conformer summaries.
- The strict OOF adapter cache contains 6,387 samples, 201,994 residues, 1,024-dimensional ProtT5 tokens, and 128-dimensional spatial states; every spatial state comes from a checkpoint that excluded its fold.
- The attached conformer tensors have dimensions 8 residue, 12 global, and 3 source features. All tensors are finite, and uncovered samples retain exact-zero global/source descriptors plus a hard availability mask.
- Jointly retraining the chemical/conformer adapter improves the frozen 0.62/0.38 baseline to MCC 0.870828 but remains just below the current chemical-adapter default MCC 0.871184 and has slightly worse Brier than that default.
- Reproducing the chemical-only adapter from the combined cache exactly matches the published default MCC, ACC, SN, and SP, so the new cache does not alter the comparison protocol.
- The first parameter-frozen residual run left frozen-base Dropout active during training. It was retained for audit but superseded by a function-level freeze that keeps all base modules in evaluation mode.
- The final function-frozen seed-42 residual changes uncovered-sample probabilities by exactly zero and produces MCC 0.871583, AUPRC 0.953651, Brier 0.044434, and ECE-10 0.025902, with one fold win and four ties.
- Fixed seeds 42/123/2025 averaged without post-hoc weighting produce MCC 0.871583, AUPRC 0.953629, Brier 0.044437, and ECE-10 0.025854. This has one fold MCC win, four ties, zero losses, and a slight AUPRC regression, so it does not replace the current default.
- The final model decision is to retain the chemical cross-attention default and report residue-level multi-conformer coupling as a positive aggregate but retention-gate-negative ablation.
- The multi-conformer set covers only lengths 11-30; all 3,592 length-31-50 samples are uncovered.
- The covered cohort has a 7.62% default decision error rate versus 3.29% in the uncovered cohort, so conformer availability is confounded with a harder length/domain subset.
- PepFlow coverage contains only 111 toxic versus 650 non-toxic samples, while AF2 contains 670 toxic versus 1,364 non-toxic samples. Shared residual fitting can therefore suffer source and class imbalance.
- The final fixed-seed conformer ensemble changes only one threshold decision among 6,387 samples. The current residual magnitude/objective is too weak to materially affect MCC.
- The default makes 221 false-negative and 110 false-positive decisions, making sensitivity-oriented correction with a specificity constraint the most defensible threshold-level target.
- Recent primary toxicity models support sequence/structure cross-attention and hybrid experts, but this repository has already screened generic descriptor and atom-graph routes; the next code work should target the diagnosed error/coverage bottleneck rather than duplicate those branches.
- The fixed required-control matrix completed without Test1/Test2: Control MCC 0.871583 (1 fold win), Margin MCC 0.871982 (2 wins), Source-length MCC 0.871583 (1 win), and Combined MCC 0.871658 (3 wins, 1 loss).
- Margin-only provides the highest fixed-threshold MCC, but it misses the predeclared three-fold stability gate.
- Combined improves sensitivity to 0.881738 and AUPRC to 0.953672, but Brier worsens to 0.044625 and one fold loses MCC.
- No required-control candidate passes all unchanged retention gates, so no post-hoc seed expansion, full-data refit, or Test1/Test2 evaluation is justified.
- The 31-50 aa pilot result archive has a matching SHA256 and contains 400 balanced samples (200 toxic/200 non-toxic).
- All 400 pilot samples were collected successfully from 20 discovered AF2 models, diversity-selected to exactly eight conformers each.
- The cache contains 3,200 conformer NPZ files plus 400 ensemble summaries, with zero sequence mismatches and no failed samples.
- This is a pilot only; the remaining long-peptide cohort has 3,192 samples after excluding the 400 pilot records.
- The 31-50 covered-cohort residual improves AUROC/AUPRC by 0.00570/0.00560 and lowers Brier/ECE-10 by 0.00738/0.00829.
- Fixed-threshold BACC/MCC decline by 0.00500/0.00960, with four corrected and six newly incorrect decisions.
- The AUROC paired bootstrap interval is wholly positive and the Brier interval wholly negative, but the MCC interval crosses zero and the prespecified MCC non-decrease gate fails.
- Full-training OOF retains small AUROC/AUPRC/Brier improvements but loses 0.000816 MCC; generating the remaining 3,192 long-peptide structures is therefore stopped.
- The old cohort analyser mislabeled all lengths above 30 as `26-30`; it now supports observed 1-15 through 51+ buckets and the 31-50 pilot reports four correct 5-aa strata.
- The latest GitHub README reports a newer geometry self-supervised pretraining and
  three-level 1D/3D fusion experiment that is absent from the current local README.
- That upstream experiment reaches OOF MCC 0.87241 but regresses on historical
  Test1/Test2 transfer, so it is documented as a negative ablation rather than the default.
- Local HEAD `8cb7da2` is exactly one commit behind upstream main `907acc9`.
- The upstream commit changes 206 files, dominated by generated logs, PID files,
  histories, and result artifacts; merging the commit wholesale would pollute the
  current dirty worktree and overwrite local README/code work.
- The algorithmic upstream delta is concentrated in `geometry_pretrain.py`,
  `cross_modal_adapter.py`, `models/conformer_graph.py`,
  `multisource_conformer_analysis.py`, `select_conformer_pilot.py`, and tests.
- Upstream geometry SSL is fold-pure and label-free: each OOF encoder trains on
  the other four folds using masked residue reconstruction, pair distances, and
  6/8/10 A contacts, then attaches residue embeddings to the aligned adapter cache.
- The optional cross-view refinement trains only on multi-conformer samples with
  source-balanced sampling; upstream results show it and later GroupDRO variants
  do not justify promotion, so they should remain optional research paths.
- Upstream's combined 1-50 experiment agrees with the local independent pilot:
  calibration improves, but global MCC/AUROC/AUPRC regress versus the retained
  full2795 graph candidate, so the 400 long peptides must not replace the default.
- High-value merge scope is therefore algorithmic/reproducibility code only:
  geometry SSL, hierarchical fusion controls, dense-graph memory chunking, and
  confidence/length-balanced pilot selection, plus focused tests and compact summaries.
- Selective integration is complete. The local adapter now composes geometry SSL and
  hierarchical fusion with the existing conformer residue/global/source residual,
  decision-margin loss, and fold-local balanced sampler.
- Old non-hierarchical checkpoints remain loadable. If the hierarchical input width
  changes the delta head, that head is deliberately reset to its exact zero-residual
  fallback while compatible projections are retained.
- Dense chemical-graph edge bias is evaluated in exact chunks above 262,144 edge rows;
  a zero-tolerance equality test confirms the chunked and unchunked outputs match.
- The selected result payload is limited to three compact JSON summaries. No upstream
  checkpoint, cache, prediction CSV, PID, history, or training log was imported.
- Final verification: Ruff passed; 28 focused tests passed; the full non-RDKit suite
  passed 114 tests with five existing RDKit-dependent atom-graph tests deselected.
- The retained residue adapter was asymmetric: ProtT5 residues queried the 3D state,
  while no reverse 3D-query/1D-key-value message path existed.
- The new bidirectional branch preserves the old prediction exactly at initialization
  because its independent residual head is zero initialized; structure dropout or an
  explicit unavailable mask also zeros the entire new residual.
- Agreement is measured per aligned residue after both attention directions and is
  multiplied by pLDDT and a learned mutual gate. This yields auditable sample-level
  summaries without treating low-confidence geometric evidence as equally reliable.
- Frozen-base training exposes only `reverse_`, `mutual_`, `bidirectional_`, and
  optional `conformer_` extensions. The baseline adapter remains in evaluation mode,
  so Dropout cannot silently perturb the reference prediction.
- The fixed bidirectional OOF screen intentionally excludes geometry SSL,
  hierarchical motifs, and multi-conformer features. This isolates the causal
  contribution of bidirectionality before any combined candidate is considered.
- No new performance claim is available yet. Test1/Test2 must remain untouched unless
  the prespecified five-condition OOF gate passes.
- The completed bidirectional OOF screen failed the gate: MCC changed from 0.871184
  to 0.870001 and no fold improved MCC. AUROC/AUPRC also decreased slightly.
- Brier improved by 0.0000466 and ECE-10 by 0.001128, confirming that stronger
  interaction can smooth probability calibration without improving discrimination.
- The learned bidirectional residual had mean -0.01120 and mean absolute magnitude
  0.02102. It was active but systematically shifted probabilities downward, losing
  two true positives and adding one false positive.
- Mean mutual agreement was only 0.4954. Forward/reverse attention agreement is not
  a sufficiently directional measure of when 3D evidence corrects the 1D expert.
- Per the frozen protocol, this branch is closed without refit or Test1/Test2 use.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Audit prediction schemas before modifying code | Conformer artifacts may cover different cohorts or use different identifier columns |
| Prefer existing project metric and fold utilities | Keeps results consistent with current paper-validation outputs |
| Add retention criteria at the outer-fold level | Prevents a candidate from being accepted based on one aggregate metric alone |
| Use `full_calibrated_predictions.csv` as the first stronger 3D expert | It has full OOF coverage and useful structure-quality columns |
| Use train for gradient fitting, validation for hyperparameter choice, calibration for Platt/threshold, and outer test only for evaluation | Matches the existing group-aware nested protocol and avoids selection leakage |
| Treat existing sample-level quality-gating experiments as completed negative controls | Avoids duplicating an already rejected path and overfitting the same OOF predictions |
| Extend the retained residue-level adapter with multi-conformer descriptors | Adds conformational dynamics directly before attention and preserves exact fallback when unavailable |
| Keep the frozen 0.62/0.38 baseline unchanged | Isolates the effect of the new residue-level dynamics interaction and preserves paper comparability |
| Keep geometry SSL/hierarchical fusion as an optional ablation | OOF improves slightly, but historical Test1/Test2 transfer does not pass the replacement gate |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| The documented Python environment path is unavailable | Use `D:\anaconda\envs\pytorch_env\python.exe`, which has PyTorch and CUDA |
| The initial cache verification used the wrong field name | Inspect the serialized schema and validate `token_embeddings`/`spatial_embeddings` directly |
| PowerShell default text decoding rendered the latest UTF-8 README appendix incorrectly | Re-read targeted sections with explicit UTF-8 decoding |

## Resources
- `D:\dachuang2026\GHXTox\src\ghxtox\coupled_fusion.py`
- `D:\dachuang2026\GHXTox\src\ghxtox\nonnegative_stacking.py`
- `D:\dachuang2026\GHXTox\runs\conformer_ensemble_residual_oof`
- `D:\dachuang2026\GHXTox\runs\plm_textcnn_prott5_bsmote_multiseed\oof_predictions.csv`

## Visual/Browser Findings
- AutoDL was visible in the user's previously signed-in Chrome tab.
- The instance list currently contains five healthy, powered-off, pay-as-you-go
  RTX 4090 x1 instances, all in Beijing B.
- Instance `074e4a9aca-106dc2e1` (machine 253) is the leading execution target:
  its data disk is only 28.03% used and its release window is the longest
  (14 days 22 hours), giving the pilot enough storage and restart margin.
- Starting any listed instance activates pay-as-you-go billing, so the exact
  hourly rate must be inspected before power-on and reported to the user.
- Browser recovery checks now report that Chrome is not running and the ChatGPT
  Chrome Extension is not detected in the selected profile. The native-messaging
  host itself is correctly installed. No AutoDL action was taken.
- The user manually prepared AutoDL instance `074e4a9aca-106dc2e1`. Its runtime
  check confirms BioEmu 1.4.1, PyTorch 2.13.0+cu130, CUDA 13.0,
  `cuda_available=true`, and an NVIDIA GeForce RTX 4090.
- The selected AutoDL image does not include `tmux`; this is not a BioEmu or CUDA
  failure. Use `nohup` plus a PID/log file for the smoke run.
- The first smoke failure occurred before inference while
  `maybe_download_checkpoint()` called Hugging Face Hub for the official
  `microsoft/bioemu` checkpoint. The final exception was HTTPX
  `RuntimeError: Cannot send a request, as the client has been closed`.
- This is an infrastructure/dependency failure, not a structure-quality result.
  Pin `huggingface-hub==0.36.0`, prefetch the checkpoint and config, and rerun the
  exact frozen six samples before interpreting the gate.
- After pinning Hugging Face Hub, the explicit prefetch reached the older
  requests-based path but timed out connecting to `huggingface.co:443`.
  This confirms the remaining blocker is AutoDL outbound connectivity.
- AutoDL's official remediation is `source /etc/network_turbo` for terminal
  access to Hugging Face. Its official Hugging Face guidance also recommends
  placing `HF_HOME` under `/root/autodl-tmp` to avoid filling the system disk.
- With the proxy enabled, BioEmu advanced to the bundled AlphaFold2 weight
  download but AutoDL's proxy returned HTTP 500 for
  `storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar`.
- BioEmu hard-codes the AF2 cache under `~/.cache/colabfold` and considers it
  complete only when `params/download_finished.txt` exists. The safe next route
  is to disable the HF-only proxy, keep Hugging Face in offline cached mode, and
  download AF2 weights directly into a data-disk-backed symlink.
- The direct aria2 AF2 download completed successfully, but BioEmu 1.4.1 still
  uses `HfFileSystem.glob()` to list named checkpoints before sampling. A local
  Hugging Face cache alone therefore does not eliminate the API dependency.
- BioEmu 1.4.1 accepts paired `ckpt_path` and `model_config_path` arguments.
  Passing both cached files bypasses `maybe_download_checkpoint()` and its
  remote model-listing request.
- A smoke report containing any nonzero or missing return code is an incomplete
  infrastructure run. It must not trigger ESMFlow selection or extrapolate cost
  from failed-row elapsed time; only the identical six-row smoke retry is
  authorized.
- The corrected six-row smoke passed every frozen gate, and the authorized
  200-row BioEmu pilot completed with 200/200 zero return codes.
- The server pilot report records 99.0% strict usability: 198/200 samples pass,
  with 65/67 usable at 11-15 aa and 67/67 plus 66/66 usable in the longer strata.
  It contains 5,683 valid conformers (minimum 12, median 29, mean 28.41) and
  consumed 3.062 GPU hours.
- The result archive downloaded to the workstation is 17,469,312 bytes. Its
  computed SHA256 `f1c10dbba383c9df3bd62ac0614bb1cd87520d18834175883ba14acd63339cfc`
  exactly matches the supplied checksum file.
- Pre-extraction inspection found 1,430 archive entries, zero unsafe paths, zero
  duplicate paths, and exactly 200 each of trajectories, topologies, run-status
  records, and raw batch NPZ files.
- The archived 200-row manifest and selection summary are byte-identical to the
  frozen local inputs (SHA256 `ffc477c...409ff` and `c1408e...8b3d7`), so the
  returned structures correspond to the authorized cohort.
- The deferred Phase 19 diagnostic implementation already exists and is covered
  by a focused alignment/confounding test. It fixes the three training-only
  length strata, uses immutable model thresholds, reports confusion/metrics by
  stratum and generator, and explicitly records that external test labels are
  not read.
- The fixed Phase 19 run covers 6,387 training OOF rows and 3,195 conformer
  rows. Covered generator source and length stratum are perfectly associated
  (Cramér's V 1.0): PepFlow is exclusively 11-15 aa, full AF2 exclusively
  16-30 aa, and the long AF2 pilot exclusively 31-50 aa.
- Fixed-threshold joint OOF MCC by length is 0.649187 (11-15), 0.827701
  (16-30), and 0.919805 (31-50). The corresponding 1D MCCs are 0.594277,
  0.813335, and 0.915097; however, on the balanced 400-row long-AF2 pilot the
  joint MCC 0.740333 remains below the 1D MCC 0.756600.
- BioEmu stores XTC coordinates in nanometers while the existing conformer
  descriptor/cache implementation uses Angstrom cutoffs (6/8/10 A). The new
  converter multiplies filtered C-alpha XTC coordinates by 10 before reusing the
  established descriptor implementation.
- All 200 BioEmu trajectories convert successfully with exact agreement to the
  feasibility report's per-sample valid-frame counts, totaling 5,683 filtered
  conformers.
- Under one BioEmu source, conformer features predict length stratum with
  cross-fitted balanced accuracy 0.729685, so length remains physically encoded
  even after generator unification.
- On 143 same-sample pairs, BioEmu versus the legacy PepFlow/AF2 source is
  distinguishable with balanced accuracy 0.870629. The unified generator
  removes the old length/source identity but does not make its descriptors
  interchangeable with legacy-source features.
- Fixed BioEmu descriptors predict training-only toxicity at balanced accuracy
  0.620. A 1,000-permutation test that preserves fold, length, and label counts
  gives p=0.000999 overall; by length the balanced accuracies are 0.507130,
  0.655971, and 0.818182.
- The same 200-row fixed 1D OOF anchor is correct on 172 rows versus 124 for the
  BioEmu descriptor classifier. Across 66 disagreements, BioEmu corrects nine
  1D errors but introduces 57 new errors (net -48), so direct replacement or
  unbounded expansion is not justified.
- The single preregistered Phase 23 bounded residual leaves 199/200 decisions
  unchanged. It raises AUPRC from 0.922592 to 0.922792 and lowers Brier from
  0.140379 to 0.140219, but MCC falls from 0.727161 to 0.718125.
- The only decision change is `peptide_800` (fold 3, positive, 11-15 aa): the
  frozen 1D probability 0.453088 is moved to 0.449308, crossing the unchanged
  0.45 threshold in the wrong direction. Thus corrected/newly-incorrect is 0/1.
- Excluding the two prespecified strict-quality exceptions does not change the
  conclusion: MCC falls from 0.742514 to 0.733250 even though AUPRC, Brier, and
  ECE improve slightly.
- The bounded candidate fails the preregistered MCC and net-correction gates.
  It must be stopped without trying another residual weight on the same pilot.
