# GHXTox Default Model

The single public prediction default is GHXTox-3D-Chemical-Adapter-Residual.

- Canonical configuration: configs/default_3d_hybrid.json
- Canonical entry point: ghxtox-predict / ghxtox-hybrid-predict
- Sequence-only reference: ghxtox-prott5-predict

## Architecture

1. A 15-member ProtT5-BSMOTE TextCNN sequence expert (3 seeds x 5 group folds).
2. A pLDDT-aware 3D structure expert with ESM2 states, C-alpha EGNN, and an all-atom chemical-site branch.
3. Frozen training-OOF probability fusion: ProtT5 0.62, 3D expert 0.38.
4. A residue-aligned ProtT5-query cross-modal adapter using 3D states and chemical-site summaries. Its output is a bounded, zero-initialized logit residual.
5. A frozen disagreement-only BACC decision residual. The 1 x 16 MLP uses 26 deployable features and may replace the joint decision with the ProtT5 teacher decision only when its gate probability reaches `0.9298851979999941`.
6. Frozen joint-model decision threshold 0.60 and ProtT5 teacher threshold 0.45.

The BACC residual is decision-only: it changes `prediction` for selected disagreements but preserves `toxicity_probability`. Consequently, AUROC, AUPRC, Brier, and ECE-10 remain properties of the chemical-adapter probability output.

The complete chemical-site cache and residue-aligned adapter cache are required for the public route. Missing 3D inputs must fail explicitly; sequence-only fallback is allowed only with an explicit fallback flag.

## Retained Results

The retained local validation artifacts are:

- Test1: ACC 0.9440, BACC 0.9251, SN 0.8813, SP 0.9690, MCC 0.8611, AUROC 0.9816, AUPRC 0.9650.
- Test2: ACC 0.9450, BACC 0.8608, SN 0.7609, SP 0.9608, MCC 0.6603, AUROC 0.9578, AUPRC 0.7849.

Relative to the pre-residual joint decision, the frozen residual changes no Test1 decision metric and corrects one Test2 false negative. These are retained historical artifacts, not permission to tune weights, gates, or thresholds on Test1/Test2.
