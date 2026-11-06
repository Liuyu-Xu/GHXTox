"""Regression tests for the frozen ProtTox-3DKD deployment runner."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import torch

from ghxtox.prottox_3dkd_predict import MODEL_NAME, predict_sequences


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "prottox_3dkd_v1_golden.json"


def _configured() -> bool:
    return bool(os.environ.get("GHXTOX_PROTTOX_3DKD_MANIFEST", "").strip())


@unittest.skipUnless(_configured(), "Set GHXTOX_PROTTOX_3DKD_MANIFEST to run the golden deployment test")
class ProtTox3DKDGoldenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = os.environ["GHXTOX_PROTTOX_3DKD_MANIFEST"]
        self.model_path = os.environ.get("GHXTOX_PROTTOX_3DKD_MODEL_PATH") or None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    def test_golden_single_sequence_probabilities(self) -> None:
        for case in self.golden["cases"]:
            with self.subTest(name=case["name"]):
                result = predict_sequences(
                    [case["sequence"]],
                    self.manifest,
                    model_path=self.model_path,
                    device=self.device,
                )[0]
                self.assertAlmostEqual(
                    float(result["probability"]),
                    float(case["probability"]),
                    delta=float(self.golden["tolerance_absolute"]),
                )
                self.assertEqual(result["label"], case["prediction"])

    def test_ensemble_contract(self) -> None:
        result = predict_sequences(
            [self.golden["cases"][0]["sequence"]],
            self.manifest,
            model_path=self.model_path,
            device=self.device,
        )[0]
        self.assertEqual(result["provenance"]["model_name"], MODEL_NAME)
        self.assertEqual(result["provenance"]["ensemble_seeds"], [20260905, 20260906])
        self.assertEqual(sorted(result["member_probabilities"]), ["20260905", "20260906"])


if __name__ == "__main__":
    unittest.main()
