# Publication validation (2026-10-26)

- Public implementation synchronized from upstream commit `a03f846f502e07c3e3a179e1720a0a1518d6596d`.
- Public `src/`, `configs/`, `tests/`, `dataset/`, dependency files and `research/multiconformer/` were compared against upstream and were unchanged. Root additions are research navigation and Git asset handling.
- All 1811 selected research files verified against their published SHA256, including staged Git blob bytes. Larger text assets are gzip-compressed; `restore_assets.py --verify-only` independently verified the complete asset manifest.
- Python syntax checks passed for the public source/tests and the research source, scripts, analysis and tests. Main research reading-entry links were checked.
- High-confidence token/private-key scan covered the selected research and legacy archive and found no credentials. Weight/feature binaries, downloaded manuscripts, third-party download caches and private backups are excluded.
- Initial public tests in the existing environment produced 28 passes and 6 failures, all at the PyTorch–NumPy interface (`Numpy is not available` with PyTorch 2.3.1 and NumPy 2.2.6). No source workaround or existing-environment package change was applied.
- Re-running the same full public test suite with a separate, validation-only NumPy 1.26.4 directory passed **34/34 tests**. That temporary dependency directory is not published. This is a source/interface check, not a new neural-model performance evaluation.
- Neither full GHXTox weights nor labels, thresholds or old test predictions were changed by this publication. Research-side training/evaluation scripts that require excluded assets were not claimed to have been re-run.

The original local working directory and its modified files were preserved. Local backup contains Git history, the binary patch and the selected research sources. Earlier fork work is archived separately under `../legacy_hypeptox/` and its Git ancestry is retained.
