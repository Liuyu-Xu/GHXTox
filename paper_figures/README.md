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

As of 2026-08-10, the two dataset-specific residue-occlusion figures, the
restricted structural-confidence/chemical-site sensitivity figure, and the
formal current-default training-OOF selective-versus-random prediction figure are approved for
final drawing. The current-default frozen-threshold Test1/Test2 transfer figure is also complete
and approved as established-benchmark transfer evidence. The frozen 1D--3D
expert-disagreement package is complete as a restricted complementarity figure:
it reports four correctness groups, reciprocal rescue, and five prespecified
attribute contrasts, but it does not support a causal conformer claim. The OOF package reports
five fixed coverage points, equal-coverage random rejection, and paired uncertainty;
the transfer package applies the unchanged OOF cutoff and reports natural Test1/Test2
coverage. Both packages use the direct normalized decision-boundary proximity
selected on the canonical OOF vector; neither is a newly collected independent-dataset validation.
Figures or tables involving multi-conformer evaluation or unresolved paired
uncertainty remain paused. Stable single-structure architecture cores may be
prepared only as editable drafts until their layout and captions receive final
manuscript sign-off.
