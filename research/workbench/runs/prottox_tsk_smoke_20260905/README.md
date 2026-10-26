# ProtTox-TSK local smoke — 2026-09-05

## Purpose

This directory is a local RTX 4060 feasibility check only. It confirms that frozen ProtT5 residue inference, TriStat extraction, physicochemical descriptors, MLP/KAN head optimization, checkpoint writing, OOF CSV writing, and one-sequence prediction all execute together.

It is not a reportable evaluation and must not be compared with historical GHXTox results:

- only 128 records were selected;
- the source's first 128 records were single-class and are retained only as a diagnostic attempt;
- `features_128_balanced.pt` instead uses deterministic 64/64 label-balanced sampling, not an audited group split;
- training used one epoch per fold and the fallback folds are not the project group manifest;
- neither head is promoted by this smoke run.

## Valid smoke artifacts

- `features_128_balanced.pt` plus its `.manifest.json`: true frozen ProtT5 cache from the local offline snapshot.
- `mlp_1epoch_balanced/`: five fold checkpoints, final full-data checkpoint, OOF CSV and manifest.
- `kan_1epoch_balanced/`: equivalent KAN candidate artifacts.

The next permitted comparison is a full-cache, audited group-fold MLP/KAN experiment under the same frozen protocol.
