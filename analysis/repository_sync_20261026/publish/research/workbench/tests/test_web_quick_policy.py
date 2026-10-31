from __future__ import annotations

import json
import unittest
from pathlib import Path

from ghxtox.web_quick_policy import apply_policy, load_policy


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "web_quick_sequence_policy_v1.json"


class WebQuickPolicyTests(unittest.TestCase):
    def test_policy_is_independent_from_full_fusion_operating_point(self):
        policy = load_policy(POLICY)
        self.assertFalse(policy["external_test_labels_used"])
        self.assertNotEqual(policy["decision"]["threshold"], 0.47022256531284556)
        self.assertEqual(policy["reject"]["target_coverage"], 0.9)
        self.assertEqual(policy["model"]["seeds"], [42, 123, 2025])

    def test_reject_band_returns_uncertain_but_keeps_the_binary_call(self):
        policy = load_policy(POLICY)
        result = apply_policy(float(policy["decision"]["threshold"]), policy)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["label"], "Uncertain")
        self.assertEqual(result["binary_label"], "Toxic")
        self.assertTrue(result["recommend_full_analysis"])

    def test_policy_input_hashes_are_present(self):
        policy = json.loads(POLICY.read_text(encoding="utf-8"))
        self.assertEqual(len(policy["inputs"]), 4)
        self.assertTrue(all(len(entry["sha256"]) == 64 for entry in policy["inputs"]))


if __name__ == "__main__":
    unittest.main()
