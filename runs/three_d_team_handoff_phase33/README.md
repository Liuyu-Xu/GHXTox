# Phase33 Promoted 3D OOF Team Handoff

This directory is a train-only OOF input contract for downstream 1D/2D/3D fusion.
It is not a production/Test inference artifact.

Use `oof_3d_predictions.csv::toxicity_probability` as the standalone 3D OOF score and join only by `source_index` after verifying sample/fold/label alignment.
The fixed 3D threshold remains `0.677819`.

Phase33 changes only the frozen 600 BioEmu decision-boundary rows; all other rows are exact Phase25 fallbacks.
For a formal fusion claim, regenerate outer-nested upstream 1D/2D/3D streams. A meta-model fitted to ordinary OOF tables is exploratory only.

See `handoff_contract.json` for hashes, schema, provenance, and usage rules.
