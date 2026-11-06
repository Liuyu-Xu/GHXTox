import csv
import json
import tempfile
import unittest
from pathlib import Path

from ghxtox.structure_utility_router import _read_config, run_cross_fitted_router
from ghxtox.unified_inference import CONTRACT_FIELDS, sequence_features, sha256


class StructureUtilityRouterTests(unittest.TestCase):
    def test_sequence_features_are_pre_structure_only(self):
        features = sequence_features("ACCCG")
        self.assertEqual(features["sequence_length"], 5)
        self.assertEqual(features["cys_count"], 3)
        self.assertEqual(features["cys_adjacent_pairs"], 2)
        self.assertEqual(features["cys_terminal_count"], 0)
        self.assertEqual(sequence_features("AAAA")["cys_span_fraction"], 0.0)

    def test_cross_fitted_router_rejects_post_structure_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.csv"
            contract.write_text("source_index\n0\n", encoding="utf-8")
            protocol = {
                "artifact_type": "ghxtox_structure_utility_router_protocol", "status": "frozen_before_oof",
                "input_contract": str(contract), "input_contract_sha256": sha256(contract),
                "pre_esmfold_feature_columns": ["p_struct"],
                "target": {"loss": "brier", "definition": "(p_seq-label)^2 - (p_struct-label)^2"},
                "model": {"type": "ridge", "alpha": 1.0},
                "route_policy": {"structure_cost": 0.0, "trigger": "predicted_brier_gain_gt_structure_cost"},
                "classification_threshold": 0.5,
            }
            path = root / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_config(path)

    def test_cross_fitted_router_writes_only_oof_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.csv"
            rows = []
            for fold in range(5):
                for label in (0, 1):
                    sequence = "CCAA" if label else "AAAA"
                    rows.append({
                        "source_index": len(rows), "sample_id": str(len(rows)), "sequence": sequence, "label": label,
                        "fold": fold, "group_id": f"g{fold}_{label}", "p_seq": .8 if label == 0 else .2,
                        "p_struct": .1 if label == 0 else .9, "p_fusion": .5, **sequence_features(sequence),
                        "sequence_seed_dispersion": .02, "sequence_seed_dispersion_available": 1,
                    })
            with contract.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CONTRACT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            protocol = {
                "artifact_type": "ghxtox_structure_utility_router_protocol", "status": "frozen_before_oof",
                "input_contract": str(contract), "input_contract_sha256": sha256(contract),
                "pre_esmfold_feature_columns": ["sequence_length", "cys_count", "sequence_seed_dispersion"],
                "target": {"loss": "brier", "definition": "(p_seq-label)^2 - (p_struct-label)^2"},
                "model": {"type": "ridge", "alpha": 1.0},
                "route_policy": {"structure_cost": .01, "trigger": "predicted_brier_gain_gt_structure_cost"},
                "classification_threshold": .5,
            }
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            output = root / "router"
            summary = run_cross_fitted_router(protocol_path, output)
            self.assertTrue(summary["cross_fitted"])
            self.assertFalse(summary["post_structure_features_used_by_router"])
            with (output / "cross_fitted_predictions.csv").open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), 10)
            self.assertEqual({row["route"] for row in predictions}, {"structure"})


if __name__ == "__main__":
    unittest.main()
