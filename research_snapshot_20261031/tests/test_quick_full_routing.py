import csv
import json
import tempfile
import unittest
from pathlib import Path

from ghxtox.quick_full_routing import build_quick_full_baseline, quick_sequence_features, run_quick_full_router
from ghxtox.unified_inference import sha256


class QuickFullRoutingTests(unittest.TestCase):
    def test_quick_features_are_sequence_only(self):
        values = quick_sequence_features("ACCKR")
        self.assertEqual(values["sequence_length"], 5)
        self.assertEqual(values["cys_count"], 2)
        self.assertGreater(values["net_charge_ph7"], 0)
        self.assertIn("hydropathy_kd", values)

    def test_baseline_and_router_are_training_oof_and_reject_post_full_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = {
                "artifact_type": "ghxtox_web_quick_sequence_policy", "training_source": "frozen training OOF only",
                "model": {"calibration": {"slope": 1.0, "intercept": 0.0}},
                "decision": {"threshold": 0.5}, "reject": {"reject_band": 0.2},
            }
            policy_path = root / "policy.json"; policy_path.write_text(json.dumps(policy), encoding="utf-8")
            raw = [0.1, 0.9, 0.45, 0.55, 0.15, 0.85, 0.4, 0.6, 0.2, 0.8]
            seed_paths = []
            for seed in range(3):
                path = root / f"seed{seed}.json"; path.write_text(json.dumps({"oof": raw}), encoding="utf-8"); seed_paths.append(path)
            rows = []
            for i, probability in enumerate(raw):
                label, sequence = int(probability > 0.5), "ACCK" if i % 2 else "AAAA"
                full = 0.9 if label else 0.1
                rows.append({"source_index": i, "sample_id": str(i), "sequence": sequence, "label": label, "fold": i % 5, "group_id": f"g{i}", "p_seq": probability, "p_struct": full, "p_fusion": full, "sequence_seed_dispersion": 0.0})
            contract = root / "contract.csv"
            with contract.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            baseline_dir = root / "baseline"
            baseline = build_quick_full_baseline(contract, policy_path, seed_paths, baseline_dir, full_threshold=0.5)
            self.assertTrue(baseline["training_oof_only"])
            ledger = baseline_dir / "quick_full_oof_ledger.csv"
            protocol = {
                "artifact_type": "ghxtox_quick_full_adaptive_router_protocol", "status": "frozen_before_oof",
                "baseline_ledger": str(ledger), "baseline_ledger_sha256": sha256(ledger),
                "quick_threshold": 0.5, "full_threshold": 0.5,
                "pre_structure_feature_columns": ["p_quick", "quick_margin", "sequence_length", "net_charge_ph7"],
                "model": {"type": "ridge", "alpha": 1.0},
                "route_policy": {"trigger": "predicted_brier_gain_gt_full_analysis_cost", "full_analysis_cost": 0.0},
                "promotion_gate": {},
            }
            protocol_path = root / "protocol.json"; protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            result = run_quick_full_router(protocol_path, root / "router")
            self.assertTrue(result["cross_fitted"])
            self.assertFalse(result["post_full_features_used"])
            protocol["pre_structure_feature_columns"] = ["p_full"]
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_quick_full_router(protocol_path, root / "router_bad")


if __name__ == "__main__":
    unittest.main()
