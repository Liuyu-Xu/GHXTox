# GHXTox paper figures

This directory contains the reproducible code and project-owned inputs used to
produce manuscript figures for GHXTox.

## Directory layout

- `scripts/`: plotting and figure-assembly code.
- `data/`: frozen GHXTox tables or manifests that identify their canonical
  source files. Data from reference papers must never be copied here.
- `output/`: generated vector and raster figure exports.

## Reproducibility contract

Every final figure should record:

1. the scientific question answered by the figure;
2. the exact input CSV/JSON or manifest and its checksum;
3. the dataset, sample count, class count, threshold, and missing-value policy;
4. the retained model/configuration identifier;
5. the plotting script and command;
6. the normalization and uncertainty definitions;
7. editable/vector output plus a high-resolution raster preview.

Reference papers may inform layout, typography, and visual hierarchy only.
All numerical marks must be derived from frozen GHXTox project artifacts.

## Current teammate handoff

- [`TEAM_FIGURE_HANDOFF.md`](TEAM_FIGURE_HANDOFF.md) is the authoritative
  draw-now/draft-only/paused production brief.
- [`FIGURE_STATUS.csv`](FIGURE_STATUS.csv) is the machine-readable tracker.
- [`internal_to_publication_names.csv`](internal_to_publication_names.csv)
  maps internal provenance labels to manuscript-facing names.

As of 2026-08-08, only the two dataset-specific residue-occlusion figures and
the restricted structural-confidence/chemical-site sensitivity figure are
approved for final drawing. Figures or tables involving multi-conformer
evaluation, final expert probabilities, paired uncertainty, or unresolved
model selection remain paused. Stable single-structure architecture cores may
be prepared only as editable drafts and must not be inserted into the
manuscript as final artwork.
