# Figure scripts

Place one reproducible plotting or figure-assembly script here for each
manuscript figure. Use publication-facing names such as
`figure_1_overview.py`, not internal run or checkpoint identifiers.

Scripts should accept explicit input and output paths, avoid row-order-only
joins, and write vector output whenever the figure type permits it.

Current frozen-data script:

- `plot_selective_prediction_risk_coverage.py` renders the selected
  training-OOF policy only. It is retained as the historical policy-selection
  visualization.
- `plot_selective_prediction_formal_comparison.py` is the formal reporting
  script. It renders the five fixed coverage points against equal-coverage
  random rejection and packages paired uncertainty, exact source tables,
  provenance, checksums, SVG/PDF, and a 600-dpi PNG. The script accepts both
  historical logistic-score policies and the current direct margin policy;
  publication text is derived from the supplied frozen policy.
- `plot_selective_prediction_test_transfer.py` applies no selection rule; it
  renders the already frozen Test1/Test2 transfer artifact. Panels report
  natural fixed-cutoff coverage, accepted error, and MCC, while the companion
  table retains BACC, sensitivity, and specificity. The package includes paired
  uncertainty, exact source tables, provenance, checksums, SVG/PDF, and a
  600-dpi PNG. Its cutoff annotation is read from the supplied policy rather
  than embedded in the artwork code.
- `plot_expert_disagreement_analysis.py` renders the frozen Test1/Test2
  1D--3D complementarity analysis. It packages four-group class composition,
  reciprocal rescue rates with Wilson intervals, and five prespecified
  3D-only versus 1D-only effects with bootstrap intervals, together with exact
  source tables, manifests, checksums, SVG/PDF, and a 600-dpi PNG.
