import os
import tempfile
import unittest
from pathlib import Path

from ghxtox.local_web_api import _validate_sequence, runtime_status


class LocalWebApiTests(unittest.TestCase):
    def test_sequence_validation_is_strict(self):
        self.assertEqual(_validate_sequence(" acd\nef "), "ACDEF")
        with self.assertRaises(ValueError):
            _validate_sequence("ACX")
        with self.assertRaises(ValueError):
            _validate_sequence("AC")

    def test_assets_without_runner_remain_non_predictive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("full_seed42.pt", "full_seed123.pt", "full_seed2025.pt", "best_model.pt"):
                (root / name).write_bytes(b"checkpoint")
            previous = os.environ.pop("GHXTOX_WEB_RUNNER", None)
            try:
                status = runtime_status(asset_root=root)
            finally:
                if previous is not None:
                    os.environ["GHXTOX_WEB_RUNNER"] = previous
            self.assertIn(status["state"], {"unavailable", "configuration_required"})
            self.assertFalse(status["model"]["soft_cys_default_enabled"])
            self.assertFalse(status["model"]["structure_utility_router_enabled"])
            self.assertEqual(runtime_status(asset_root=root, runner="canonical-runner")["state"], "ready")
