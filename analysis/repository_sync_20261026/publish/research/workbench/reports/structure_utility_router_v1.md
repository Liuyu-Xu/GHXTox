# Structure-utility router v1

## Contract

`runs/unified_inference_oof_v1/contract.csv` contains 6,387 identity-aligned training-OOF records. It includes the sequence, label, group-held-out fold, sequence probability (`p_seq`), retained single-structure probability (`p_struct`), retained fusion probability, sequence-only Cys/length features, and three-seed sequence dispersion.

The router feature set is frozen in `configs/structure_utility_router_v1.json`. It contains only sequence-derived fields and sequence-seed dispersion. It excludes `p_struct`, `p_fusion`, pLDDT, coordinates, candidate topology information, and every other feature unavailable before structure prediction.

## Cross-fitted protocol

The target is the prespecified OOF Brier gain:

`(p_seq - label)^2 - (p_struct - label)^2`.

For each outer fold, a ridge model is fitted only on the other four folds and predicts expected gain for the held fold. Structure is selected only when the predicted gain exceeds the frozen structure cost of 0.0025. The route does not tune a threshold, feature set, model, or cost after OOF evaluation.

## Result

The cross-fitted router selected the structure path for 4.59% of records. Its Brier score was 0.04594, compared with 0.04537 for the sequence-only route; its cost-adjusted utility was also worse (0.04605 versus 0.04537). The retained single-structure route was worse than sequence-only on this contract, with Brier 0.05940.

The result is therefore a negative routing screen: do not deploy the current structure-utility router to trigger structure computation and do not tune this frozen protocol after reviewing its OOF result. Preserve the contract, protocol, cross-fitted predictions, and router states as an auditable training-only record. A future router requires a newly frozen protocol and a distinct candidate structure-enabled path with demonstrated OOF utility.
