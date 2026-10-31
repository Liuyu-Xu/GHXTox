# Soft-Cys topology strict OOF: final result

## Frozen run

- Result package: `topology_strict_oof_v1_final.tar.gz`
- Package SHA-256: `606f042abce6243dae61873034fcf3dd65fd413130497207b6ada80097aefce3`
- Five strict outer folds; each retained backbone was scratch-trained for the frozen 12 epochs.
- The topology experiment completed all 25 fold-by-variant runs and 5,000 paired group-bootstrap resamples.
- The audit records `external_test_labels_used: false`.

## OOF result

| Variant | MCC | Balanced accuracy | AUROC | AUPRC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.7834 | 0.8873 | 0.9527 | 0.9109 | 0.0785 |
| Local | 0.7916 | 0.8942 | 0.9514 | 0.9026 | 0.0748 |
| Matching | 0.7961 | 0.8962 | 0.9536 | 0.9102 | 0.0743 |
| Topology | 0.7957 | 0.8950 | 0.9525 | 0.9091 | 0.0747 |

Topology versus baseline had MCC delta 0.0123 (95% paired group-bootstrap CI [0.0070, 0.0182]) and Brier delta with CI [-0.0054, -0.0023]. Its Cys>=4 label-standardized error was 0.1591 versus 0.1640 for baseline.

## Decision

The frozen promotion rule is not met: topology does not exceed the matching control on any primary discrimination metric. Record this as a negative feasibility result. Do not change the frozen training settings or proceed to partial-unfreeze/full training from this result.

## Engineering boundary

Keep the soft-Cys candidate cache, matching, topology generator, and optional residual branch as an auditable experimental module. The default product configuration must leave `soft_cys_topology` disabled. Any later promotion requires a newly frozen protocol and a new independent evaluation run.
