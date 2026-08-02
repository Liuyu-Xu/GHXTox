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
- Phase 24 can reuse a common ensemble-summary schema for AF2-16-30,
  AF2-31-50, and BioEmu: aligned C-alpha coordinates, residue RMSF,
  conformer RMSD/Rg, pair-distance mean/std, and 6/8/10 A contact occupancy.
- The PepFlow full manifest contains 6,088 conformer rows for 761 unique
  peptides rather than one row per peptide. Phase 24 must deduplicate by
  `original_id` and resolve the per-sample ensemble summary instead of treating
  conformers as independent labeled examples.
- BioEmu contributes 200 unique samples and 143 same-sample overlaps with the
  legacy sources. Paired versions must stay together in the canonical fold so
  no generator view of a held-out peptide appears in training.
- All four sources resolve to the same ensemble-summary keys. The unique
  training union is 3,252 peptides: 3,195 legacy-covered plus 57 BioEmu-only;
  the three legacy sources are mutually disjoint by construction.
- The reusable `ensemble_feature_tensors()` already defines eight bounded
  residue features and twelve global features from the ensemble statistics.
  Phase 24 should extend this with explicit pairwise distance/contact channels
  rather than collapse the contact matrices to row means again.
- Existing all-training baselines are aligned by `source_index`: 6,387-row
  3D-v2 and calibrated conformer-residual OOF files plus the canonical five-fold
  manifest. The new standalone output can therefore be compared on exactly its
  3,252-sample union and exported with a missing-structure mask for all rows.
- The workstation PyTorch environment has CUDA-enabled PyTorch 2.3.1 on an RTX
  4060 Laptop GPU, so a small fixed five-fold contact graph can run locally;
  AutoDL is not required for Phase 24.
- The frozen existing 3D-v2 decision threshold is 0.677819. Its all-training OOF
  MCC is 0.836202 and AUPRC 0.928001; matched-cohort comparisons must use this
  existing threshold rather than retune it on the new 3D-covered subset.
- Fixed-epoch outer-fold training is preferable to early stopping on each held
  OOF fold, because using the evaluation fold for checkpoint selection would
  make the claimed OOF performance optimistic.
- PepFlow and BioEmu manifests preserve canonical `source_index`, but the two
  AF2 manifests use a local 0-based subset index. Canonical alignment for every
  source must be derived from `original_id=peptide_N` as `N-1`, then validated
  against sequence and label in the 6,387-row fold manifest.
- After canonical `peptide_N -> N-1` alignment, the exact union is 3,252 rows
  with 1,008 positives and 2,244 negatives. Matched 3D-v2 at its frozen 0.677819
  threshold has MCC 0.764328 and AUPRC 0.887914 on this union.
- The fixed Phase 24 graph encoder finishes five folds locally in about 78
  seconds. Its OOF MCC/AUPRC are 0.594800/0.818271, below matched 3D-v2
  0.764328/0.887914, with zero fold MCC wins and weaker results in all three
  length strata.
- Pair consistency partially works: held same-sample legacy/BioEmu embeddings
  have mean cosine 0.941583 and mean absolute probability delta 0.077696.
  However, a fold-local source probe still achieves BACC 0.814685, so the
  representation retains substantial generator identity and is not suitable
  for fusion handoff.
- Phase 24 must stop without increasing the consistency weight or searching a
  new architecture on the same OOF result. The useful team deliverable should
  instead package the already retained 3D-v2 plus capability-aware conformer
  residual stream and its quality/missingness metadata.
- The retained team handoff covers all 6,387 canonical training rows with exact
  fold/label/sequence alignment. At frozen threshold 0.677819 it has MCC
  0.839018, AUPRC 0.928171, Brier 0.059399, and ECE-10 0.051109.
- Exactly 3,195 rows expose conformer features (PepFlow 761, AF2-16-30 2,034,
  AF2-long-pilot 400); the other 3,192 rows preserve the exact base-3D fallback
  and an explicit missing-conformer mask.
- OOF handoff tables are valid base inputs but cannot be fitted and scored by a
  fusion meta-model on the same rows. Formal fusion claims require the existing
  nested protocol because upstream predictions for other folds may have been
  trained using the outer evaluation fold.
- Phase 26 audit finds that the retained capability-aware conformer checkpoint
  is currently a seed-42 residual trained fold-wise from fixed chemical-site
  3D-v2 checkpoints. Its frozen config uses up to 50 epochs, batch size 32,
  learning rate 0.001, held-fold-internal validation fraction 0.15, patience 8,
  BCE-loss monitoring, and only the conformer encoders/branch/logit trainable.
- A valid multi-seed experiment must distinguish residual-head randomness from
  independently trained 3D-v2 randomness. Merely rerunning the small conformer
  residual under new seeds would not support the claim that 3D-v2 itself is a
  multi-seed ensemble; the upstream base-fold training provenance must be
  audited before the frozen seed set is finalized.
- Crosswalking the earlier nine-item 3D roadmap against completed work shows
  that BioEmu-200, length/source diagnostics, bounded quality residuals,
  contact-occupancy graph aggregation, cross-modal coupling, and masked geometry
  pretraining have already been executed or integrated. Their outcomes do not
  support repeating the same experiments under new names.
- Full BioEmu generation is no longer justified by the 200-row evidence: the
  pilot is operationally feasible, but same-sample generator identity remains
  strong and the bounded downstream candidate lowered MCC. ESMFlow and
  AFsample2 are therefore fallback-generator research, not current priorities.
- The still-useful part of the old contact-occupancy proposal is the bounded
  ensemble descriptors plus explicit missingness/quality contract already
  included in the retained 3D handoff. A new standalone contact graph was tested
  and failed substantially, so only the descriptors—not that graph model—should
  be retained.
- The exact Mac-Diff-inspired idea from the old roadmap—a 3D residue reading
  only 1D tokens in a high-contact-occupancy spatial neighborhood—has not been
  evaluated verbatim. Existing work tested residue interaction, conformer
  residuals, and a global bidirectional agreement gate; the latter improved
  calibration but lost MCC in all five folds. A strictly local contact-masked
  coupler remains technically distinct, but it belongs to the fusion workstream
  and should wait for a new frozen validation setting rather than become another
  search on the exhausted current OOF.
- Phase 26 local preflight confirms all five retained chemical-site 3D-v2 fold
  checkpoints and all five seed-42 conformer-residual fold checkpoints exist.
  Required processed tensors/fold manifests are present; the two largest full
  processed caches are about 1.16 GB and 1.17 GB, while the pilot250 training
  tensor is about 20 MB.
- `ghxtox.train` reads the seed only from the JSON config; its CLI has no seed
  override. Multi-seed automation must therefore materialize immutable
  seed-specific config files instead of editing the retained seed-42 config.
- The current `set_seed()` covers Python and PyTorch CPU/CUDA RNGs but does not
  request deterministic CUDA algorithms. Exact byte-for-byte retraining is not
  guaranteed across devices; the reference gate needs exact row/fold alignment
  plus preregistered metric/probability tolerances rather than checkpoint hashes.
- The original 3D-v2 seed-42 protocol trains only the zero-initialized chemical
  site branch from fixed `chemical_site_final_control_fold{fold}` checkpoints;
  it uses 30 epochs, batch 32, LR `5e-4`, weight decay `1e-4`, coordinate noise
  `0.12`, and Pareto AUPRC selection subject to matched-control MCC. Varying the
  seed at this exact branch-training stage is therefore the faithful independent
  3D-v2 seed experiment; the frozen upstream control remains common by design.
- Seed-specific conformer residuals must then initialize from the matching
  seed-specific 3D-v2 fold checkpoints and train on the same pilot250 folds.
  Averaging only base branches or attaching every residual to seed 42 would not
  represent the preregistered complete-path ensemble.
- Phase 26 protocol preparation initially recomputed AUPRC as `0.927682`
  because the shared training metric helper casts serialized probabilities to
  float32 before ranking. This creates artificial ties among very close scores.
  Phase 26 now uses the exact float64 metric implementation already used by the
  retained handoff, whose `0.928171` AUPRC remains the frozen baseline.
- Phase 26's first aggregation implementation covered seed/fold/length metrics
  and calibration but did not yet emit the preregistered generator-source
  cohorts, decision-disagreement accounting, or measured artifact time/storage.
  These report fields must be added before the three-seed gate is consumed.
- Phase 26 completed with all three 6,387-row OOF streams. Seed 42 passed the
  reference reproduction gate (probability Pearson `0.999564`; MCC/AUPRC/Brier
  differences `0.002000/0.000269/0.000275`), authorizing seeds 123 and 2025.
- The three individual retained-path seeds are tightly clustered but all below
  the Phase 25 reference: MCC `0.837018/0.836632/0.836202`, AUPRC
  `0.927901/0.926662/0.927475`, versus reference `0.839018/0.928171`.
- The preregistered equal-weight ensemble fails promotion with MCC `0.835649`
  and AUPRC `0.927630`. Brier and ECE-10 improve to `0.058648/0.048849`, but
  aggregate MCC/AUPRC and the 11-15 aa length stratum regress beyond the frozen
  tolerance.
- Decision disagreement analysis shows only 16 ensemble-vs-reference threshold
  changes: four corrections and twelve regressions (net `-8`). All three seeds
  agree on 6,333/6,387 decisions (`0.991545`), so the negative result is stable
  rather than a failed run.
- Phase 27 P0 audit confirms the retained handoff already exports mean pLDDT,
  low-pLDDT fraction, mean C-alpha RMSF, conformer RMSD, radius of gyration, and
  8 A contact uncertainty. Repackaging these fields alone would repeat a tested
  quality-residual direction rather than add new 3D evidence.
- The reusable ensemble summaries currently expose aligned C-alpha coordinates,
  RMSF, radius of gyration, pair-distance mean/std, and 6/8/10 A contact
  occupancy. Atom-level SASA and DSSP cannot be recovered from these summaries;
  Phase 27 must first determine whether full-atom PDB/topology artifacts are
  available, then freeze a descriptor contract consistent with actual inputs.
- The Phase 25 handoff remains the immutable Phase 27 reference: 6,387 rows,
  3,195 conformer-covered rows, threshold `0.677819`, MCC `0.839018`, and AUPRC
  `0.928171`. Phase 27 must preserve exact fallback on the 3,192 rows without
  conformer summaries.
- The workstation contains exactly 3,195 train-structure PDB files, matching
  the retained conformer coverage. An inspected AF2 file contains multiple
  models, full heavy-atom side chains/backbone atoms, and per-residue pLDDT in
  the B-factor field, so new atom/residue spatial descriptors can be extracted
  without generating more structures.
- Existing cache code already parses every valid PDB model into C-alpha,
  backbone, functional-group, and chemical-site coordinates. Phase 27 can add
  self-contained residue-exposure, hydrophobic/charge patch, shape, and contact
  order descriptors while reusing validated sequence/source alignment.
- DSSP/SASA outputs are not currently cached. To keep the experiment portable,
  secondary structure should be labeled explicitly as a backbone-geometry
  proxy, and exposure should use a frozen atom/residue occlusion definition
  unless an already-installed deterministic SASA implementation is confirmed.
- The 3,195 raw train PDBs resolve exactly by source: 2,034 under the AF2
  16-30 full set, 400 under the AF2 31-50 pilot, and 761 under the PepFlow
  short-peptide set. This matches the handoff source counts with no extra
  generated rows entering the Phase 27 cohort.
- The declared project dependencies contain NumPy/PyTorch/NetworkX and optional
  scikit-learn/Biopython, but no deterministic SASA or DSSP package. Phase 27
  should therefore implement a small pure-NumPy contract instead of adding a
  platform-dependent executable solely for this experiment.
- Phase 27 completed the frozen 20-descriptor candidate over all 3,195 covered
  rows in 87 seconds on CPU. Fold-local source standardization reduced the
  linear generator-source probe BACC from `0.917099` to `0.333333`, confirming
  that the preprocessing removed the dominant linear source mean/scale signal.
- The physical-descriptor residual slightly improves AUPRC from `0.928171` to
  `0.928202` and Brier from `0.059399` to `0.059270`, but MCC falls from
  `0.839018` to `0.838166`. It changes eight fixed-threshold decisions, with
  three corrections and five regressions (net `-2`).
- Three of five fold MCC values are non-decreasing, but fold 3 loses `0.007737`.
  The 11-15 aa/PepFlow cohort is the material failure: MCC drops from
  `0.650427` to `0.643650`, beyond the frozen `0.005` tolerance. The fixed
  global candidate therefore retains the Phase 25 handoff and cannot be
  rescued by post-hoc feature or residual-weight selection.
- Phase 28 feasibility confirms Biopython's deterministic Shrake-Rupley SASA is
  installed and works directly on PepFlow's eight-model PDBs. Representative
  13-15 aa models contain 97-138 heavy atoms after removing 117-129 explicit
  hydrogens, and 100-point residue/atom SASA takes `0.0085-0.0132` seconds per
  model after parsing.
- The 761-row short-peptide surface extraction should therefore take roughly
  one to three CPU minutes and requires no new structure generation. Phase 28
  can freeze heavy-atom-only handling and a bounded 12-feature surface contract
  without adding an external binary dependency.
- Phase 28 completed all 761 PepFlow rows with exactly eight PDB models each in
  101.5 CPU seconds. All 5,626 non-target rows preserve the Phase 25 probability
  exactly, confirming the short-only fallback contract.
- The all-atom surface specialist improves short-peptide AUPRC from `0.692577`
  to `0.693980`, overall AUPRC from `0.928171` to `0.928200`, Brier from
  `0.059399` to `0.059343`, and ECE-10 from `0.051109` to `0.050989`.
  Nevertheless, short MCC falls from `0.650427` to `0.645344` and overall MCC
  falls to `0.838649`.
- Only one fixed-threshold decision changes, in fold 3, and it changes from
  correct to incorrect (`0` corrections, `1` regression). Four folds are
  unchanged and fold 3 loses `0.001830` MCC. The frozen global, short-MCC, and
  positive-net-correction gates fail, so the Phase 25 handoff remains retained.
- Phase 29 begins from the combined Phase 27/28 result: hand-crafted scalar 3D
  descriptors can improve ranking/calibration but have not learned the
  threshold-local residue interactions needed to improve MCC. The next
  candidate must therefore train on residue/chemical-site geometry directly,
  not fit another logistic residual over aggregate descriptors.
- Any Phase 29 claim must be outer-fold safe: each held-fold short prediction
  must come from a fold-specific checkpoint trained without that fold. A model
  trained on all OOF rows or selected by held-fold epoch performance would be
  leakage and is explicitly outside the authorized protocol.
- The prior threshold-neighborhood asymmetric margin loss exists only in
  `cross_modal_adapter.py`, where it regularizes the 1D/3D coupling adapter.
  The core residue/chemical-site 3D-v2 trainer in `train.py` currently has no
  decision-boundary loss, so adding a fixed-threshold term there is not a
  duplicate of the Phase 6/7 adapter control.
- The core 3D trainer already supports coordinate noise, fold-specific
  checkpoint initialization, prefix-based parameter freezing, conformer delta
  regularization, and direct residue/functional-group/chemical-site tensors.
  This is sufficient to build a bounded short-specific fine-tuning experiment
  without introducing another sample-level descriptor model.
- Five fold-specific retained conformer checkpoints are present locally at
  `runs/conformer_ensemble_residual_calibrated_fold{0..4}/best_model.pt`, along
  with the five chemical-site base checkpoints. Each conformer checkpoint is
  about 6.18 MB, so fold-correct initialization and a compact AutoDL handoff
  are feasible without regenerating the base models.
- The retained conformer refinement already freezes the base and trains only
  `conformer_residue_encoder`, `conformer_global_encoder`,
  `conformer_ensemble_branch`, and `conformer_ensemble_logit`. A Phase 29
  short specialist should keep this bounded trainable set (rather than unfreeze
  the whole sequence/EGNN model) and change the training cohort/loss only.
- The 3D-v2 chemical-site stage uses coordinate noise `0.12` and Pareto
  validation, whereas the retained conformer residual uses no coordinate
  noise, no class reweighting, and BCE-selected early stopping. Because the
  candidate is only 761 short rows, a fixed epoch count derived without held
  fold metrics is preferable to another held-fold early-stopping choice.
- Direct tensor audit corrects an important assumption: the retained
  `train_chemical_sites_final_esm2_pepflow.pt` has conformer features for only
  the stratified pilot250 (139 negatives, 111 positives), not all 761 PepFlow
  PDB rows. The other 511 generated short structures are negative-only and were
  deliberately excluded after the earlier full761 screen failed.
- The five retained conformer checkpoints selected epochs `23, 18, 9, 12, 1`
  (median `12`). Phase 29 can prospectively fix 12 epochs so it does not choose
  a candidate checkpoint from the held outer fold. It should train on the same
  pilot250 cohort and evaluate its 250 held-fold OOF replacements while leaving
  every other Phase 25 probability exact.
- The existing asymmetric penalty correctly converts the probability threshold
  `0.677819` to logit space before applying a hinge width of `0.15`. The same
  mathematical contract can be moved into the core trainer with fixed weight
  `0.10` and positive multiplier `1.5`, eliminating any new hyperparameter
  search.
- The retained Phase 25 handoff's 3,195 conformer rows are assembled from
  multiple generator-specific OOF streams; the PepFlow fold checkpoints only
  alter the 250 pilot rows. Therefore Phase 29 must replace exactly those 250
  held-fold probabilities in the retained handoff and preserve all other 6,137
  rows byte-for-byte at the probability level.
- The pilot250 folds are label-balanced by construction (per-fold positives
  `17-31`, negatives `26-30`) and retain canonical sample IDs/source indices.
  They provide the correct fold manifest for specialist training; final
  aggregation must align to the canonical 6,387-row handoff by source index and
  verify sample ID, label, sequence, and fold.
- The 12-epoch freeze is conservative but materially different from the old
  held-fold BCE selector: the retained checkpoints ranged from epoch 1 to 23.
  The new final checkpoint must be the epoch-12 state regardless of validation
  metrics, while validation outputs remain diagnostic only.
- The exact Phase 25 conformer initialization is
  `chemical_site_ablation_include_same_residue_fold{fold}`, not the similarly
  named final-control checkpoint. All five exact initial checkpoint hashes are
  now frozen in the Phase 29 protocol so the AutoDL run cannot silently change
  its base.
- The canonical Phase 29 protocol and all compact outputs use only
  `runs/short_margin_3d/`. Its verifier now covers the training config, both
  processed tensors, both fold manifests, the Phase 25 reference OOF, and all
  five base checkpoints before an AutoDL run may start.
- Phase 29 completed in 173.1 seconds on the local RTX 4060. The fixed 12-epoch
  margin candidate lowers global MCC by `0.000408` and pilot250 MCC by
  `0.007636`; global/pilot AUPRC also fall by `0.000071/0.001656`.
- The failure is localized but decisive: fold 1 changes one positive from a
  correct to an incorrect fixed-threshold prediction, lowering its MCC by
  `0.038184`; the other four folds change no decisions. Net corrections are
  `-1`, while all 6,137 fallback probabilities remain exact.
- Three consecutive short-3D directions (source-robust scalar descriptors,
  all-atom surface descriptors, and threshold-aware residual fine-tuning) now
  fail MCC while sometimes marginally improving calibration. The limiting
  evidence is the small 111-positive pilot and fold variance, not lack of one
  more scalar feature or margin constant. The next defensible existing-data
  direction is label-free residue-level multi-conformer pretraining with source
  invariance, followed by one separately frozen fine-tune; higher-value progress
  requires new positive short-peptide conformers or external validation.
- With new data explicitly unavailable, Phase 30 is restricted to existing
  artifacts: 3,195 legacy multi-conformer rows, 200 BioEmu rows, and 143
  same-sequence legacy/BioEmu pairs. The latter are the only clean evidence for
  generator invariance because generator source and length are otherwise
  confounded.
- Existing `geometry_pretrain.py` already trains fold-pure masked-residue,
  pair-distance, 6/8/10 A contact, and ESMFold-anchor-to-random-conformer
  objectives. A defensible new candidate must instead learn from two ensemble
  conformers directly and use matched legacy/BioEmu pairs; merely raising the
  old consistency weight or rerunning its OOF screen is prohibited.
- Phase 24 already trained a supervised two-layer contact graph with cosine
  consistency on the same 143 pairs. It achieved high paired cosine but weak
  toxicity prediction and high source-probe BACC. Phase 30 must pretrain without
  toxicity labels first, initialize a retained strong 3D path, and measure source
  leakage before one fixed fine-tune rather than train another standalone graph
  from scratch.
- `multiconformer_graph_3d.py` already provides the canonical, fold-safe 3,252
  sample union builder and resolves all source indices from `peptide_N`. It can
  be reused for Phase 30 data alignment, but its supervised `EdgeGatedContactEncoder`
  and 0.5 classifier must not be reused as the candidate objective.
- The existing ensemble summaries contain the aligned per-conformer C-alpha
  coordinates needed for two independently sampled conformer views; the current
  Phase 24 loader instead collapses them to distance mean/std and contact
  occupancy before training. Phase 30 can therefore add new residue-level
  information without generating or inventing data.
- Direct adversarial classification over all PepFlow/AF2/BioEmu samples is not
  valid because source and length are confounded. Generator invariance will be
  restricted to same-sequence legacy/BioEmu pairs; unpaired rows contribute only
  conformer reconstruction/consistency losses.
- Canonical existing manifests are present for PepFlow full761, AF2 16-30
  full2034, AF2 31-50 pilot400, and BioEmu200. No extra dataset download or
  generation is needed for the Phase 30 audit and implementation.
- All four resolved `ensemble_summary.npz` schemas provide `aligned_coords`,
  `pair_distance_mean/std`, and 6/8/10 A contact occupancy. PepFlow/AF2 store
  eight aligned conformers; BioEmu stores its filtered variable count. Phase 30
  can therefore train two stochastic conformer views and ensemble-distribution
  targets directly from existing bytes.
- Ensemble summaries do not expose per-conformer pLDDT. To avoid creating a
  generator-specific confidence shortcut, Phase 30 should use a fixed confidence
  of one for its label-free ensemble views and reserve the retained model's
  original pLDDT path for the immutable base prediction.
- Phase 30 is prospectively fixed to reuse the established four-epoch geometry
  SSL budget and optimizer, but changes the views from ESMFold-versus-ensemble
  to ensemble-conformer-versus-ensemble-conformer. Fold encoders remain label-free
  and see only the other four folds.
- The downstream candidate is not a new standalone classifier: it freezes the
  Phase 30 encoder and adds a zero-initialized residue-attention residual of at
  most `0.35` logit to the immutable Phase 25 probability. Samples without an
  existing ensemble-domain embedding must remain numerically exact fallbacks.
- The complete Phase 30 module is now present and syntactically valid. It keeps
  the Phase 25 probabilities immutable, trains a separate fold-pure encoder per
  outer fold, and records both exact-fallback equality and source-robustness
  diagnostics before any promotion can be authorized.
- Label-free ensemble-conformer pretraining did learn its reconstruction task
  and improved ranking/calibration, but that signal did not improve the fixed
  classification boundary: global MCC changed by `-0.001032`, with six
  corrections versus eight regressions.
- Very high legacy/BioEmu paired cosine (`0.993872`) did not imply domain
  invariance. A fold-local linear source probe still reached BACC `0.779720`,
  so generator identity remains encoded in directions not captured by mean
  cosine alone.
- The dominant failure is again the 11-15 aa layer (`MCC -0.018224`), whereas
  16-30 aa gained `+0.002023` MCC and 31-50 aa lost only `-0.001413`.
  With no new data, the current existing-data 3D branch has now exhausted the
  preregistered high-priority representation, descriptor, surface, margin, and
  seed-ensemble routes; Phase 25 is the defensible retained output.
- The proposed Phase 31 model has only four learned parameters per outer fold.
  Sharing one positive slope preserves ranking within every length stratum;
  only two strongly regularized length offsets can alter cross-stratum
  calibration and fixed-threshold decisions. This is materially less flexible
  than fitting three independent Platt models.
- A fixed calibrated threshold of `0.5` is essential to the new claim. The fair
  reference remains the retained Phase 25 probability at its frozen threshold
  `0.677819`; searching either threshold after seeing Phase 31 OOF would turn a
  small calibration experiment into post-hoc metric optimization.
- The official Phase 25 handoff explicitly states that its probabilities were
  not recalibrated and its threshold was not retuned. Phase 31 therefore needs
  a new optional handoff with both `phase25_probability` and
  `length_calibrated_probability`; overwriting `toxicity_probability` before a
  promotion gate passes would break the existing team contract.
- The repository's generic `fit_platt` utility enforces a positive monotonic
  scale, but it cannot express the two strongly shrunk length intercepts. The
  Phase 31 implementation should keep the same stable logit clipping and metric
  conventions while fitting the frozen four-parameter objective directly.
- The user's suggested four-fold-fit/one-fold-apply rule prevents direct label
  use from the held fold, but it does not repair the provenance of the four
  Phase 25 OOF covariates: each was produced by an upstream model that excluded
  its own fold and may have trained on the current held fold. The repository
  handoff contract explicitly prohibits treating this as a fully nested fusion
  claim.
- `data/folds/train_nested_groupcv.csv` supplies disjoint train, validation,
  calibration, and test roles, but no corresponding strict nested Phase 25
  probability stream exists. A valid official promotion would require
  retraining the complete Phase 25 upstream path inside each outer fold, using
  the calibration role only after base training and checkpoint selection.
- The frozen length offsets were heavily shrunk as intended: across folds the
  short offsets were only `-0.0030` to `-0.0050` and long offsets `+0.0035` to
  `+0.0075`. The learned change was therefore almost entirely a global slope
  compression (`0.3413-0.4227`) plus negative global intercept, not meaningful
  length-specific correction.
- Global Platt-like compression greatly improves calibration but is the wrong
  objective for the retained MCC boundary: Brier improves by `0.003249` and
  ECE-10 by `0.033081`, while 19 correct decisions are broken for only eight
  corrections. The failure is not optimizer instability; every fold converged
  and probability Pearson/Spearman remain `0.986097/0.996198`.
- Because the inexpensive screen misses several gates by material margins,
  retraining the full Phase 25 path under strict nested roles would incur large
  GPU cost without positive screening evidence. The actual feasible decision
  is to stop this calibration route and retain the raw Phase 25 score for the
  fusion team.
- Phase 32 can avoid the Phase 31 second-level OOF provenance problem by
  training its supervised head without any Phase 25 probability or logit. Each
  fold-specific 3D head sees structures and toxicity labels from only the other
  four outer folds; the immutable Phase 25 held score is combined only after
  training when the held row is predicted.
- All five Phase 30 checkpoints share the expected two-layer state schema.
  `encoder.layers.1` is a concrete bounded unfreeze target; the SSL node/pair
  decoders are not part of inference and will stay frozen and unused.
- Full eight-conformer backpropagation would multiply the effective graph batch
  by eight on the local RTX 4060. The frozen feasible contract instead samples
  two conformers per training row and aggregates up to eight conformers only at
  inference, retaining ensemble information without an unnecessary memory or
  runtime expansion.
- Real inputs are finite, but some residue channels have exactly zero variance
  across two conformers. Direct `sqrt(variance)` therefore creates an infinite
  derivative; multiplying that derivative by the zero-initialized head gradient
  yields NaNs in `encoder.layers.1`. Using `sqrt(variance + 1e-6)` preserves the
  intended standard-deviation feature while making the partial-unfreeze path
  differentiable at zero variance.
- Phase 32 demonstrates that limited supervised adaptation of the label-free
  encoder is technically stable but still does not add enough independent
  boundary signal. The standalone covered-row head is weak (MCC `0.601901`,
  AUPRC `0.768539`); bounding it against Phase 25 protects calibration but cannot
  guarantee useful threshold transitions.
- The final candidate repeats the pattern seen in several earlier 3D additions:
  AUPRC improves by `0.000061`, Brier by `0.000110`, and ECE-10 by `0.000418`,
  while MCC loses `0.000889`. Its ten changed decisions contain four corrections
  and six regressions, making the frozen promotion failure substantive rather
  than a calibration or numerical artifact.
- Partial unfreezing does not materially worsen measured generator leakage:
  paired embedding cosine is `0.991944` and source-probe BACC `0.776224`, close
  to Phase 30. Yet every length layer's MCC is lower (`-0.003092`, `-0.000116`,
  `-0.001360`), so source stability alone is insufficient for promotion.

## Phase 33 initial design findings
- The retained handoff contains 415 fixed-threshold errors. A read-only audit
  shows that the 10% of rows closest to the frozen decision threshold contain
  175 of them (`42.17%`), making a label-blind boundary cohort the highest-value
  target for expensive unified-generator evidence.
- Every one of the 6,387 rows has a base structure, but only 3,195 have a
  multi-conformer ensemble. Raw performance for ensemble-covered and uncovered
  rows is not causally comparable because generator and length are strongly
  confounded; full indiscriminate structure generation is therefore not
  authorized by that subgroup contrast.
- The existing 200-row BioEmu pilot is technically usable (198/200 strict
  usable, 5,683 valid conformers, 3.062 GPU hours). Phase 33 may reuse only the
  frozen boundary-cohort overlap and must generate every missing selected row
  with the same BioEmu protocol.
- The user-supplied screenshot is treated as design input only. The actionable
  local evidence is the Phase 25 handoff plus the completed Phase 20-32 audits;
  no external claim in the screenshot changes the frozen evaluation boundary.
- The existing `bioemu_pilot.run_manifest` intentionally refuses manifests
  larger than 200 rows and requires an explicit smoke/pilot authorization flag.
  Phase 33 must add a separate boundary-generation entry point rather than
  weakening the completed Phase 20 safety contract.
- The Phase 33 AutoDL manifest will be label-free. Labels remain only in the
  local canonical handoff/fold files and may be joined after generation for
  fold-pure training; the server needs only identity, sequence, fold, frozen
  boundary distance, deterministic seed, and reuse status.
- The prospective cohort rule is the originally proposed global absolute
  probability distance to `0.677819`, with exactly 600 rows and `source_index`
  as the deterministic tie-breaker. Diagnostic alternatives using logit
  distance or forced fold quotas were inspected but are explicitly rejected;
  they will not be selected based on their observed labels or errors.
- The frozen rule naturally retains all five outer folds (counts
  `53/235/108/145/59`) and all three length strata (`69/246/285` for
  11-15/16-30/31-50 aa), so no label-aware balancing or fold quota is needed.
- The formal selector produced exactly 600 rows and its server manifest has no
  label column. Existing verified BioEmu outputs can be reused for 26 selected
  rows; the remaining 574 rows require new generation.
- The frozen cohort covers 69/246/285 rows in the 11-15/16-30/31-50 aa strata
  and 53/235/108/145/59 rows in outer folds 0-4. Post-freeze diagnostics find
  394 positives, 206 negatives, and 157 existing Phase 25 errors; these values
  did not participate in membership selection.
- Extrapolation from the completed 200-row pilot gives 50.545 median seconds
  and 358,714 mean bytes per sequence, projecting 8.059 RTX 4090 GPU hours and
  0.206 GB for the 574 missing rows.
- The portable AutoDL bundle needs only six `ghxtox` source files, the existing
  BioEmu installer, the new resumable launcher, and the compact Phase 33 data
  directory. A clean extraction test reproduced all 574 dry-run commands with
  no labels, confirming that no large local cache or prior run directory is
  accidentally required inside the upload zip.
- The completed Phase 33 server generation is usable as a specialist-training
  cache: all 574 newly generated rows and all 26 reused pilot rows align to the
  frozen 600-row cohort, yielding 16,510 filtered conformers with no Test1/Test2
  access and no label exposure on the AutoDL side.
- Boundary-only residuals solve the main failure mode observed in earlier 3D
  fusion attempts. The standalone BioEmu boundary head remains weak
  (boundary-only MCC `0.170354`), but as a small fixed residual near the Phase
  25 threshold it produces a positive net correction without perturbing the
  5,787 non-boundary rows.
- Phase 33 is the first 3D-only upgrade in this sequence that passes the frozen
  promotion gate: MCC `0.839758` versus `0.839018`, AUPRC `0.928204` versus
  `0.928171`, Brier `0.059378` versus `0.059399`, ECE-10 `0.051094` versus
  `0.051109`, four fold MCC non-decreases, all length layers within tolerance,
  and decision transitions `3` corrected / `1` regressed.
## Phase 34 fusion audit findings

- The latest README already ends with a Phase33 section (`C.21`) and contains
  older generic ensemble/stacking guidance. The new documentation must clearly
  distinguish the promoted standalone 3D handoff from the historical full
  1D/2D/3D default and must not imply that the Phase33 gain is itself a fusion
  gain.
- The repository already documents that a second-level model fitted to OOF
  predictions can have invalid provenance unless its outer evaluation rows were
  excluded from every upstream model that produced the meta-features. The new
  fusion contract therefore needs two reporting tiers: a strict nested OOF
  headline protocol, and a clearly labeled exploratory cross-fitted screen when
  strict nested upstream streams are unavailable.
- Phase33 keeps exact Phase25 probability fallback on 5,787 non-boundary rows
  and changes only the frozen 600-row BioEmu boundary cohort. Fusion consumers
  must join by stable row identity and verify label/fold consistency rather than
  relying on CSV row order.
- The local branch is `main`, tracks `origin/main`, and is already six commits
  ahead before this task. The worktree also contains many earlier Phase26-33
  changes and untracked artifacts, so publishing requires an explicit file/size
  audit instead of a blanket `git add .`.
- The Phase33 handoff contract correctly states: no fusion weights, no
  recalibration, no threshold retuning, fixed 3D threshold `0.677819`, 6,387
  rows, 600 boundary rows, and 5,787 exact fallbacks. It already warns that
  fitting and scoring a meta-model on this same OOF table is biased.
- No repository-level `AGENTS.md` is present, so there are no additional local
  agent instructions beyond the project documentation and frozen protocols.
- The primary Phase33 execution path is fold-pure at the visible split point:
  for held fold `f`, it trains on `sample.fold != f`, predicts only
  `sample.fold == f`, then combines the held standalone 3D logit with the held
  Phase25 probability after supervised training. All 600 evidence rows are
  required before evaluation, and fallback differences are measured on the
  other 5,787 rows.
- A concrete reproducibility risk exists in `--resume`: an existing
  `fold_N/result.pt` is currently trusted without checking its fold, source
  indices, protocol/input hashes, or fixed fine-tuning configuration. A stale
  result from a prior protocol/output directory could therefore be silently
  merged. This must be closed before publishing.
- The fixed Phase33 training path does not pass Phase25 probabilities/logits
  into `_train_fold`; training uses only BioEmu tensors and labels, fixed eight
  epochs, fold-local class weighting, and no checkpoint/threshold search.
  Phase25 is introduced only in `_combine_held_evidence` after held prediction.
- Each Phase30 initialization checkpoint is validated for the requested fold
  and encoder configuration before use. The final evidence residual is bounded
  by `0.15*tanh(logit)`, so its absolute logit effect cannot exceed 0.15.
- There is no focused `tests/test_boundary_specialist_3d.py` yet. The published
  Phase33 path therefore lacks direct regression tests for exact fallback,
  held-only evidence, promotion-gate behavior, and safe resume validation.
- Boundary membership is genuinely label-blind in code: candidates are sorted
  only by `abs(toxicity_probability - 0.677819)` and `source_index`; labels are
  kept in a separate lookup and joined only for post-freeze diagnostics. The
  existing test flips every label and verifies that the selected 600 IDs remain
  identical, while the server manifest is verified to have no label column.
- Fold coverage is a post-selection validity gate rather than a label-aware
  quota. This preserves the predeclared global boundary ranking and aborts if
  the frozen cohort would be unusable instead of silently changing membership.
- `oof_fusion.screen_oof_fusion` performs a leave-one-base-fold-out
  second-level screen on existing upstream OOF probabilities. This prevents the
  held row's label from directly fitting its fusion weight/threshold, but it is
  not a strict nested estimate because meta-training OOF features may come from
  upstream models whose training sets included the current meta-held fold. Its
  current module wording overstates the provenance and must be corrected.
- `oof_fusion.apply_frozen_fusion` has a real reproducibility bug: screening can
  choose `fusion_space="logit"`, but the application path always averages in
  probability space and exposes no fusion-space option. Frozen logit-space
  parameters therefore cannot be faithfully applied.
- The fusion alignment path validates sequence/label and source index rowwise,
  but does not currently validate expert-fold agreement, duplicate source
  indices, finite probabilities, or probability range. These are inexpensive
  contract checks worth adding before team handoff.
- Phase33 is currently an OOF 3D handoff, not a complete production/Test
  inference artifact. A final full-train specialist and a frozen label-blind
  target boundary rule are still required before it can be applied to unseen
  rows; README must not imply otherwise.
- The frozen Phase33 selection artifact records the top-600 ranking rule but
  not the numerical 600th-row probability-margin cutoff. That is sufficient to
  reproduce the training cohort from the full Phase25 OOF table, but a future
  unseen-set inference policy must separately freeze either a training-derived
  absolute-margin cutoff or a target-set fraction before any test outcome is
  viewed.
- Both Phase33 launcher scripts currently force `--resume`. After adding strict
  resume metadata, the pre-hardening local `result.pt` files will intentionally
  fail closed. The launchers need an explicit opt-in resume switch (and ideally
  a selectable clean output root) so a normal reproduction is not coupled to
  stale local artifacts.
- The Phase33 team handoff directory is referenced only by README/artifacts;
  there is no source/script entrypoint that deterministically regenerates it
  from the Phase25 handoff plus Phase33 OOF output. This is a reproducibility
  gap to close before GitHub publication.
- The Phase25 handoff builder already contains the correct warning that even a
  plain outer split of ordinary upstream OOF tables is insufficient for a
  strict nested claim. The Phase33 fusion section should reuse this exact
  distinction rather than claim the generic `oof_fusion` screen is definitive.
- The current manually produced Phase33 CSV correctly recomputes logit,
  fixed-threshold prediction, `2*abs(p-0.5)` margin, and normalized binary
  entropy after replacing the 3D score. It also keeps Phase33 BioEmu
  availability in separate audit columns, so legacy conformer availability is
  not silently redefined.
- The current Phase33 handoff contract omits hashes of its own output CSVs and
  lacks a machine-readable `OOF-only / not production inference` status. A
  deterministic builder should add both.
- Real-data rebuild exposed a boundary condition in verification, not in the
  model: exact fallback rows may legitimately contain probabilities exactly 0
  or 1, whereas recomputing through the bounded-logit helper clips them to
  `1e-7`/`1-1e-7`. Residual arithmetic must be checked only for the 600 boundary
  rows; non-boundary rows must instead be checked by exact probability equality
  and zero delta. This preserves both numerical safety and the stronger fallback
  invariant.
- The audited generator reproduced the existing Phase33 handoff on all checked
  identity/score/mask/threshold fields (maximum numeric difference below
  `5e-13`) and exactly reproduced MCC/AUPRC. The formal directory was then
  regenerated with output hashes and explicit OOF-only scope.
- The frozen fusion design is now: nonnegative convex combination of clipped
  1D/2D/3D logits fitted on outer-train, positive-slope Platt calibration fitted
  only on outer-calibration, and fixed post-calibration threshold 0.5. The
  Phase25-vs-Phase33 ablation must use identical outer-nested streams and all
  preregistered gates; current ordinary OOF tables remain exploratory only.
- The working tree contains many compact prior Phase26-33 engineering artifacts
  in addition to this audit. No untracked file observed in the size audit exceeds
  about 1.2 MB, but ignored raw caches/checkpoints still need a final staged-file
  review before pushing.
- The full-suite collection error is a stale test, not a missing production
  algorithm: `multisource_conformer_analysis` intentionally imports and calls
  the canonical `_length_bucket` from `conformer_cohort_analysis`, while its old
  test still imports the removed local alias `_bucket`. The same canonical
  bucket behavior already has direct tests; the stale import should be updated
  rather than reintroducing duplicate logic.
