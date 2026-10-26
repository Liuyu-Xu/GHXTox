# Topology feasibility audit

This directory contains a deterministic, training-OOF-only audit. It did not train or run a model and did not use Test1 or Test2 labels.

Recommendation: `conditional_method_development`.

The frozen primary Cys>=4 endpoint did not show excess current-3D error: the label-standardized error gap was -0.01227 (95% bootstrap CI [-0.04139, 0.01611]). A prespecified Cys>=6 sensitivity did identify a narrower failure mode, with a 0.07013 gap (95% CI [0.03251, 0.11107]) driven by false positives among rare high-Cys negatives. This supports a targeted pilot, not an expected whole-benchmark gain.

The reported Cys--Cys distances come from predicted coordinates and are candidate-contact statistics only. They do not establish native disulfide connectivity, cyclization, or experimentally verified topology.

The current chemical-site branch already encodes sulfur compatibility, continuous distance, orientation, and pLDDT. Any follow-up topology model should therefore add global Cys-pair competition, matching constraints, or weighted candidate-topology aggregation rather than another unconstrained local edge scorer.

Before server training, obtain per-peptide Cys partner pairs, evidence provenance/confidence, unknown-connection masks, cyclization status, and whether geometry was optimized with disulfide constraints. If only partial annotations exist, unknown pairs must not be treated as negatives. Evaluate a frozen-base lightweight head by five-fold training OOF, with Cys>=6 negative-class false-positive rate as the targeted endpoint and full-cohort MCC/BACC/AUPRC as non-degradation constraints. Stop if the high-Cys false-positive reduction is not directionally consistent in at least four folds, if full-cohort metrics regress, or if the apparent gain requires threshold retuning.

Reproduce from the repository root with the isolated project runtime, without changing packages:

```powershell
$env:PYTHONPATH='D:\dachuang2026\GHXTox\src'
& 'D:\dachuang2026\ziliao\ghxtox_runtime_py310\Scripts\python.exe' scripts/audit_topology_feasibility.py --protocol configs/topology_feasibility_audit_v1.json --output-dir runs/topology_feasibility_audit_20260809_reproduction
```
