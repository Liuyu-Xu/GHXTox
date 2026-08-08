# Task Plan: Run the current GHXTox default model locally

## Goal
Reproduce one local run of the canonical `GHXTox-Multidimensional-Fusion` model without changing model code or trained artifacts, and verify outputs against the frozen reference metrics.

## Current Review Goal
Resolve the multi-conformer residual's source-domain and one-sided probability-shift problem before treating the current Test1/Test2 extension as a retained model. Only after a matched-source, training-frozen comparison is available should paired bootstrap and manuscript integration resume.

## Phases
- [complete] 1. Audit local Python/GPU/dependency state and required model/data artifacts.
- [complete] 2. Locate and validate missing assets; selected canonical end-to-end Test2 inference with the audited multi-conformer extension.
- [complete] 3. Stage validated assets, prepare an isolated runtime, and execute into a new output directory.
- [complete] 4. Compare predictions and metrics with canonical frozen outputs.
- [complete] 5. Record exact runtime, outputs, results, tolerances, and environment boundary for handoff.
- [complete] 6. Map `v1.tex` sections, model names, architecture claims, operating points, and metric tables.
- [complete] 7. Cross-check every current-default claim against canonical config, source contract, README, and machine summaries.
- [complete] 8. Classify findings by severity and prepare exact line-level replacement guidance without editing `v1.tex`.
- [complete] 9. Deliver a concise audit separating model changes from metric changes.
- [complete] 10. Define the manuscript-wide replacement policy and the two-dataset conformer-ablation presentation.
- [complete] 11. Apply all model-description, headline-metric, calibration, expert-comparison, bootstrap, and conclusion revisions to `v1.tex`.
- [complete] 12. Replace the stale conformer/length Figure 6 material with separate Test1 and Test2 multi-conformer ablation placeholders and rigorous English explanations.
- [complete] 13. Run consistency searches, compile the manuscript, inspect warnings/errors, and correct regressions.
- [complete] 14. Report the completed manuscript changes without modifying the user's PyTorch environment.
- [complete] 15. Inventory all explicit TODOs, placeholder figures/tables, and structurally empty manuscript sections.
- [complete] 16. Identify analyses that require missing sample-level artifacts or new experimental evidence.
- [complete] 17. Classify remaining work by submission priority and recommend the next concrete action sequence.
- [complete] 18. Deliver the manuscript-completion audit without editing the paper.
- [complete] 19. Inspect repository conventions and establish a reproducible paper-figure code directory following the scientific-figure data contract.
- [complete] 20. Remove Section 3.5.4 from `v1.tex` and check surrounding section flow.
- [complete] 21. Inspect the complete relevant StrucToxNet text/pages for residue attribution, Integrated Gradients, seed consistency, structural hotspots, chemical-site deletion, pLDDT gating, spatial adjacency, and probability-shift analyses.
- [complete] 22. Classify proposed interpretability additions by scientific value, implementation effort, and current GHXTox artifact availability.
- [complete] 23. Compile/verify the revised manuscript and report whether the interpretability overview should be biological or model-based.
- [complete] 24. Validate and install the five restored Test1 assets without overwriting canonical outputs or changing PyTorch.
- [complete] 25. Audit current 1D/3D inference contracts and define a matched, frozen perturbation protocol and representative-case sampling rule.
- [complete] 26. Implement and test current-model residue masking, three-seed stability, chemical-site deletion, and pLDDT--gate extraction under `paper_figures/scripts/`.
- [complete] 27. Execute the prespecified Test1/Test2 interpretation cohort in the isolated runtime and freeze CSV/JSON outputs with checksums.
- [complete] 28. Freeze the verified figure-data package; defer actual plotting at the user's request and retain manuscript figure placeholders with complete English explanatory captions.
- [complete] 29. Revise the manuscript Methods/Results to match only the completed analyses, add rigorous figure placeholders/captions, compile twice, and visually verify the affected pages.
- [complete] 30. Audit the newly added Methods, Results, captions, limitations, and conclusion against the surrounding manuscript for terminology, tense, register, logical transitions, claim strength, and formulaic AI-writing patterns.
- [complete] 31. Apply targeted English and contextual revisions, reducing repetitive disclaimers, stock transitions, overpacked parallel lists, and project-report phrasing without changing verified values, experimental scope, labels, or placeholder intent.
- [complete] 32. Run manuscript-wide consistency checks, compile twice, inspect the affected PDF pages, and record the final language-QA outcome.
- [complete] 33. Document the completed interpretability experiments, manuscript-language policy, frozen outputs, verification commands, and deferred-figure status in the project README.
- [complete] 34. Re-audit all manuscript TODOs, placeholders, metadata, unsupported claims, and missing submission artifacts, then report the remaining work by priority.
- [complete] 35. Diagnose the small multi-conformer gain: quantify coverage, fusion dilution, threshold geometry, residual direction, source/domain matching, and the prior OOF promotion decisions.
- [pending] 35A. Correct the experimental protocol: recover or reproduce the exact source-aware full2795 checkpoint, evaluate it once on the frozen Test1/Test2 caches without test-set tuning, and decide from the pre-existing OOF rule whether multi-conformer inference remains a default or is reported as an exploratory near-null ablation.
- [pending] 36. Implement a versioned, deterministic class-stratified paired-bootstrap script with 5,000 resamples and seed 2026, including confidence intervals, directional probabilities, and checksum/provenance output.
- [pending] 37. Execute and independently verify the paired-bootstrap results for BACC, MCC, AUPRC, and Brier on both benchmarks without loading or modifying PyTorch.
- [pending] 38. After the matched-source conformer diagnostic determines whether a retained comparison remains scientifically meaningful, either integrate verified intervals or demote the present result to an exploratory near-null ablation; reconcile the README, Table 7, limitations, and conclusion accordingly.
- [pending] 39. Compile twice, audit LaTeX diagnostics, visually inspect the affected PDF pages, and record the completed statistical update.
- [complete] 40. Audit `v1.tex`, existing `paper_figures` assets, and frozen data manifests to define the teammate-facing figure/table production scope.
- [complete] 41. Write a publication-facing figure/table handoff inside `paper_figures`, separating draw-now deliverables from multi-conformer and other evidence-uncertain items that must remain paused.
- [complete] 42. Cross-check every requested panel against frozen source files, manuscript placeholders, scientific-figure naming/data-firewall rules, and the no-drawing-yet boundary; then finalize the handoff.
- [complete] 43. Audit the divergent local/remote history, staged and untracked files, ignore rules, large artifacts, and sensitive-content risk before upload.
- [complete] 44. Remove generated QA/LaTeX intermediates from the upload set, include reproducible source/data manifests, stage the intended latest project state, and create a local commit.
- [in_progress] 45. Rebase the local commit(s) onto the latest `origin/main`, resolving conflicts without discarding teammate or user changes, then run repository checks.
- [pending] 46. Push the integrated `main` branch to GitHub and verify the remote commit and clean intended worktree state.

## Constraints
- Treat `configs/default_multidimensional_fusion.json` as canonical.
- Do not retrain, recalibrate, or tune on Test1/Test2.
- Do not overwrite canonical artifacts under `runs/default_multidimensional_fusion/`.
- Preserve all pre-existing working-tree changes.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| PowerShell parser rejected a pipeline immediately after `foreach` in the asset audit command. | 1 | No model command ran; switch to an explicit result array before formatting. |
| A second directory-summary command accidentally reused the same invalid `foreach ... |` pattern. | 2 | Stop using formatted pipelines for loop output; emit one tab-separated line per path. |
| Parallel post-copy verification returned exit code 1 from one child and suppressed the other child outputs. | 1 | Cache counts were received; rerun hash comparison and each strict cache audit separately with explicit exit reporting. |
| Initial ProtT5 snapshot download used an overly short shell timeout and was terminated after ~5 seconds. | 1 | Hugging Face left resumable partial cache; rerun the same snapshot request with a long timeout so it resumes safely. |
| First end-to-end Test2 run stopped before model loading because `data/folds/train_cdhit080_fallback_folds.csv` is absent. | 1 | Inspect aligned OOF JSON and local training sequence sources; reconstruct the exact manifest if all fields can be verified, otherwise request it from teammate. |
| Recomputed Platt coefficients differed from stored server coefficients by ~3.5e-6 / 7.3e-6, failing an overly strict 1e-12 assertion. | 1 | Manifest loaded correctly; quantify probability and decision impact under current scikit-learn, accept only if operationally equivalent, and report numeric drift. |
| Second Test2 run reached ProtT5 loading but Transformers 4.57.6 rejected PyTorch 2.3.1 because `.bin` loading now requires torch >=2.6 for CVE-2025-32434. | 1 | Do not bypass by downgrading the safety check; use a separate isolated runtime with a compatible torch, leaving the user's PyTorch unchanged. |
| User explicitly prohibited upgrading their PyTorch; termination arrived after pip had already replaced 2.3.1 with 2.6.0 and left an invalid-distribution warning. | 1 | Stop model work and immediately restore the original PyTorch 2.3.1 CUDA 11.8 from the official source, then verify CUDA. Any future runtime must be isolated from the user's environment. |
| Post-run verification required all metrics except Brier to match at 1e-12; AUPRC differed by 5.43e-5 and triggered the assertion. | 1 | The model run succeeded; classification metrics and AUROC are exact, AUPRC/Brier round to the canonical four decimals. Verify with explicit tolerances and report environment-level numeric drift rather than claiming bitwise identity. |
| Bundled `pdftoppm.cmd` wrapper resolved first but referenced an unavailable internal path, so page rendering produced no PNG. | 1 | Located the concrete Poppler executable under the bundled native runtime and will invoke that `.exe` directly. |
| Cleanup of the QA-only `tmp/pdfs` directory was rejected by the local command policy before execution. | 1 | No deletion occurred; leave the small untracked QA intermediates in place rather than attempting to bypass the policy. |
| Completeness audit tried the shorthand local filename `predictions.csv`, but the retained local run uses `test2_predictions.csv`. | 1 | Directory listing identified the exact filename; use it for the subsequent read and correct the earlier findings entry. |
| The bundled native Poppler directory contains `pdfinfo`/`pdftoppm` but no `pdftotext.exe`, so StrucToxNet text extraction did not run. | 1 | PDF metadata was read successfully; switch text extraction to the verified TeX Live `pdftotext.exe` or bundled Python `pypdf` rather than retrying the missing binary. |
| Passing the RAR temp path directly inside a double-quoted inline Python command let PowerShell expand `$DRa4400`, producing a false missing-file path. | 1 | Do not escape the path ad hoc; pass it through a task-specific environment variable and read it with `os.environ` in Python. |
| Initial isolated-runtime probe targeted `ghxtox_runtime_py310/python.exe`, but this is a Windows virtual environment with the executable under `Scripts/python.exe`. | 1 | Use `D:/dachuang2026/ziliao/ghxtox_runtime_py310/Scripts/python.exe`; no environment modification occurred. |
| A parallel source/config inspection requested a nonexistent `runs/3d_v2_conformer_ensemble_candidate/summary.json`, causing the wrapper to suppress the successful search output. | 1 | No artifact was changed; inspect the checkpoint payload and source construction path directly, and avoid assuming that checkpoint directories contain a summary file. |
| A broad `rg` command included two nonexistent source paths (`training.py` and `cli.py`), so `rg` returned exit code 2 after still printing the useful matches. | 1 | Restrict subsequent searches to verified files such as `structure_inference.py`, `predict.py`, `utils.py`, and the model module. No experiment artifact was changed. |
| The chemical-diagnostic search included singular `models/chemical_site.py`; the actual module is plural `models/chemical_sites.py`, so `rg` returned exit code 2 after printing the needed keys. | 1 | Use the verified plural module path. The recovered keys are `chemical_site_count`, `chemical_edge_count`, and `chemical_residual_norm`; no experiment artifact was changed. |
| A pre-run directory listing returned exit code 1 because the derived interpretability-input directory does not exist yet. | 1 | This is the expected first-run state. Run the new preparation script, which creates only `paper_figures/data/current_interpretability/inputs` and does not modify canonical inputs. |
| First smoke run stopped at the deliberately strict 1D baseline check: two selected cases differed from the canonical calibrated probabilities by up to 0.0016229 (tolerance 0.0005). | 1 | The runner correctly refused to emit manuscript evidence. Compare the hidden-state head path against the model's direct forward pass and the stored per-seed predictions; identify whether the discrepancy is implementation or environment-level before changing any tolerance. No 3D experiment ran and no canonical artifact was modified. |
| First full-cohort run stopped at the same strict 1D check with a 0.0012257 maximum error across 16 cases, despite matching original padding lengths. | 1 | Preserve the entire original 16-record batch—not only its padded length—during the fp16 baseline forward, because batched GEMM shape can affect low-order results. Compute occlusion effects against an unoccluded reference in the same variant batch, then anchor those effects to the exact original-batch baseline logits. Do not relax the tolerance. |
| Second full-cohort run passed 1D validation and reached the full 3D baseline, but the maximum Test1/Test2 3D probability difference was 0.0005491 versus the initial 5e-05 tolerance; the first-16 smoke subset had differed by only 8.64e-06. | 1 | Keep formal outputs blocked. Quantify the discrepancy separately by split, sample, quantile, and threshold impact using the same isolated runtime. Change the tolerance only if the evidence shows bounded environment/source drift rather than an input/checkpoint mismatch. |
| The isolated model runtime does not include Matplotlib, so the first plotting-dependency probe raised `ModuleNotFoundError: matplotlib`. | 1 | Do not install packages into either PyTorch environment. Use the read-only bundled workspace document/plotting runtime for figure generation while keeping the model outputs frozen and unchanged. |
| A combined limitations/conclusion patch failed because the conclusion context wrapped `The controlled conformer ablation` differently from the expected patch text. | 1 | The patch was rejected atomically and changed nothing. Apply the limitations/future block and the exact conclusion block separately using verified local context. |
| A final planning-file patch included an empty `findings.md` update hunk and was rejected by `apply_patch`. | 1 | No file changed in that attempt. Reapplied the valid task/progress updates separately, then appended the findings with a nonempty verified hunk. |
| A second narrowly scoped attempt to delete only the QA-render directory was rejected by local command policy before execution. | 2 | No deletion occurred. Stop retrying; retain the PNG inspection intermediates under `tmp/pdfs/` and keep them outside the manuscript deliverables. |
| Two combined language-revision patches were rejected because the Introduction places `Ablation experiments` on the preceding sentence line, so the assumed hunk boundary was still inexact. | 2 | Both patches were rejected atomically and changed nothing. Read the exact short contexts and apply each paragraph as an independent hunk. |
| The first combined Results-language patch assumed that `Thus` began a new source line, but it followed the preceding numeric sentence on the same line. | 1 | The patch was rejected atomically and changed nothing. Apply the Results overview, residue interpretation, and captions as separate exact-context patches. |
| A broad structural/expert/case-study revision patch was rejected because the category-level sentence starts mid-line after the preceding threshold-crossing sentence. | 1 | The patch was rejected atomically and changed nothing. Read the exact affected block after earlier line shifts and patch one subsection at a time. |
| The first two-pass compile after language editing reported a new 1.84207 pt overfull box at the compound phrase `probability-level linear fusion`. | 1 | Replace the noun stack with the more idiomatic `combines their probabilities by linear fusion`, then recompile and recheck the log. |
| Two `rg --files` prediction inventories returned no matches, including `-uu`; the local Windows path/filter behavior did not expose the ignored run CSVs. | 2 | No files were changed. Diagnose with an unfiltered short `rg` listing and use native PowerShell recursion if needed rather than repeating the same filtered command. |
| A combined handoff/final-progress patch used `frozen result directory` wording that did not exactly match the existing `frozen project data` line. | 1 | The patch was rejected atomically. Read the exact local lines, apply the handoff/plan update with verified context, and append progress separately. |
| Parallel Git audit contained a PowerShell `foreach ... | Format-List` parser error, suppressing all sibling outputs. | 1 | No Git mutation ran. Collect loop rows into a variable before formatting and rerun the divergence, ignore and size audits separately. |
| Frozen interpretability verifier was called with `--results-dir`; its actual option is singular `--result-dir`, causing the parallel wrapper to suppress the credential-scan output. | 1 | No file or Git state changed. Rerun the verifier with the documented singular option and run the candidate-file scan separately. |
| First candidate credential scan built a nested two-element PowerShell array instead of a flat changed-file list; `git diff --check` also found trailing blank lines in two new README files. | 1 | Flatten the two Git path lists explicitly before scanning; remove only the redundant EOF blank lines and rerun all checks. |
| First staged-diff check found one intentional Markdown hard-break as trailing whitespace and three redundant EOF blank lines in new handoff files. | 1 | Replace the hard-break with a normal paragraph line and remove only the redundant blank lines; rerun staged syntax/diff checks before committing. |
