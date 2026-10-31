import copy
import importlib.util
import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

import torch

from ghxtox.constants import AA_TO_IDX
from ghxtox.features import RESIDUE_FEATURE_DIM
from ghxtox.models import GHXToxModel
from ghxtox.models.soft_cys_topology import SoftCysTopologyBranch, soft_matching
from ghxtox.soft_cys_topology import build_candidate_features, cached_candidate_features, candidate_matchings
from ghxtox.topology_oof import collate_topology, freeze_backbone, validate_manifest, validate_checkpoint_provenance, sha256, main as run_oof


def record(sequence="CCCC", sample_id="p", label=1):
    n = len(sequence)
    coords = torch.stack([torch.arange(n).float() * 2, torch.zeros(n), torch.zeros(n)], -1)
    types = torch.zeros(n, 2, 8)
    for i, aa in enumerate(sequence):
        if aa in "CM":
            types[i, 0, 6] = 1
    return {"sample_id": sample_id, "sequence": sequence, "label": label,
            "aa_ids": torch.tensor([AA_TO_IDX[aa] for aa in sequence]),
            "group_ids": torch.zeros(n, dtype=torch.long),
            "residue_features": torch.zeros(n, RESIDUE_FEATURE_DIM),
            "coords": coords, "plddt": torch.full((n,), .8),
            "chemical_site_coords": coords[:, None, :].repeat(1, 2, 1),
            "chemical_site_types": types,
            "chemical_site_orientations": torch.tensor([1., 0, 0]).expand(n, 2, 3).clone(),
            "chemical_site_mask": types.any(-1),
            "chemical_site_orientation_mask": types.any(-1)}


def config():
    return {"model": {"hidden_dim": 16, "num_attention_heads": 4, "num_sequence_layers": 1,
                      "num_egnn_layers": 1, "dropout": .2, "chemical_site_branch": True}}


class SoftCysTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_geometry_symmetry_rigid_motion_missing_and_methionine(self):
        item = record("CCMCC")
        original = build_candidate_features(item)
        self.assertEqual(original["indices"].tolist(), [0, 1, 3, 4])
        self.assertTrue(torch.equal(original["pair_features"], original["pair_features"].transpose(0, 1)))
        moved = copy.deepcopy(item)
        rotation = torch.linalg.qr(torch.randn(3, 3))[0]
        for key in ("coords", "chemical_site_coords"):
            moved[key] = moved[key] @ rotation + torch.tensor([5., -3., 8.])
        moved["chemical_site_orientations"] = moved["chemical_site_orientations"] @ rotation
        rotated = build_candidate_features(moved)
        torch.testing.assert_close(original["pair_features"], rotated["pair_features"], atol=2e-6, rtol=2e-6)
        item["chemical_site_coords"][0, 0, 0] = float("nan")
        invalid = build_candidate_features(item)
        self.assertFalse(invalid["pair_mask"][0].any())
        self.assertTrue(torch.isfinite(invalid["pair_features"]).all())
        item["plddt"][0] = 80
        with self.assertRaises(ValueError):
            build_candidate_features(item)

    def test_cache_invalidation(self):
        item = record()
        item["soft_cys_cache"] = cached_candidate_features(item)
        self.assertIs(cached_candidate_features(item), item["soft_cys_cache"])
        item["coords"][0, 0] += .1
        self.assertNotEqual(cached_candidate_features(item)["signature"], item["soft_cys_cache"]["signature"])

    def test_soft_matching_capacity_and_gradients(self):
        scores = torch.full((5, 5), 50., requires_grad=True)
        allowed = ~torch.eye(5, dtype=torch.bool)
        free_scores = torch.zeros(5, requires_grad=True)
        p, free = soft_matching(scores, allowed, free_scores)
        torch.testing.assert_close(p, p.T)
        torch.testing.assert_close(p.sum(1) + free, torch.ones(5))
        self.assertTrue((p.sum(1) <= 1 + 1e-6).all())
        self.assertTrue((p.diag() == 0).all())
        (p.sum() + free.sum()).backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        isolated, unpaired = soft_matching(scores, torch.zeros_like(allowed), free_scores)
        self.assertEqual(isolated.sum(), 0)
        torch.testing.assert_close(unpaired, torch.ones(5))

    def test_candidates_are_disjoint_bounded_and_have_free_option(self):
        candidates = candidate_matchings(torch.randn(7, 7), ~torch.eye(7, dtype=torch.bool), torch.zeros(7), 5)
        self.assertIn((), candidates)
        self.assertLessEqual(len(candidates), 5)
        self.assertEqual(len(candidates), len(set(candidates)))
        for pairs in candidates:
            indices = [i for pair in pairs for i in pair]
            self.assertEqual(len(indices), len(set(indices)))

    def test_null_missing_sites_padding_and_gradient(self):
        batch = collate_topology([record("AMAAA"), record("C"), record("CC")])
        h = torch.randn(3, 5, 16)
        branch = SoftCysTopologyBranch(16)
        torch.nn.init.normal_(branch.projection.weight)
        residual, diagnostics = branch(h, batch)
        self.assertEqual(residual[:2].abs().sum(), 0)
        self.assertEqual(residual[2, 2:].abs().sum(), 0)
        self.assertTrue(torch.isfinite(residual).all())
        residual.square().sum().backward()
        self.assertGreater(branch.pair_score[0].weight.grad.abs().sum(), 0)
        for item in diagnostics["topology_evidence"]:
            torch.testing.assert_close(item["pairing"].sum(-1) + item["free"], torch.ones_like(item["free"]))
        batch["chemical_site_mask"].zero_()
        residual, _ = branch(h, batch)
        self.assertEqual(residual.abs().sum(), 0)

    def test_checkpoint_compatibility_frozen_training_and_no_cys_batch(self):
        cfg = config()
        base = GHXToxModel(cfg).eval()
        cfg["model"]["soft_cys_topology"] = True
        model = GHXToxModel(cfg).eval()
        missing = model.load_state_dict(base.state_dict(), strict=False)
        self.assertFalse(missing.unexpected_keys)
        self.assertTrue(all(k.startswith("soft_cys_topology.") for k in missing.missing_keys))
        batch = collate_topology([record(), record("AAAA")])
        torch.testing.assert_close(base(batch)["logits"], model(batch)["logits"], atol=0, rtol=0)
        snapshot = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("soft_cys_topology.")}
        optimizer = torch.optim.Adam(freeze_backbone(model), lr=.01)
        self.assertFalse(model.fusion.training)
        for current in (batch, collate_topology([record("AAAA")])):
            optimizer.zero_grad()
            model(current)["logits"].sum().backward()
            optimizer.step()
        for key, value in snapshot.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        null_batch = collate_topology([record("AAA"), record("C")])
        with torch.no_grad():
            torch.testing.assert_close(base(null_batch)["logits"], model(null_batch)["logits"], atol=1e-7, rtol=1e-6)

    def test_cached_dynamic_and_spatial_only(self):
        cfg = config()
        cfg["model"].update(soft_cys_topology=True, modality="spatial_only")
        model = GHXToxModel(cfg).eval()
        items = [record(), record("CAC")]
        dynamic = model(collate_topology(items))
        for item in items:
            item["soft_cys_cache"] = cached_candidate_features(item)
        cached = model(collate_topology(items))
        torch.testing.assert_close(cached["topology_descriptors"], dynamic["topology_descriptors"])
        self.assertIn("topology_evidence", cached)

    def test_manifest_rejects_group_leakage(self):
        records = [record("A" * (i + 1), str(i), i % 2) for i in range(5)]
        rows = [{"source_index": str(i), "fold": str(i), "group": str(i),
                 "sample_id": r["sample_id"], "sequence": r["sequence"], "label": str(r["label"])} for i, r in enumerate(records)]
        self.assertEqual(len(validate_manifest(records, rows)), 5)
        rows[1]["group"] = "0"
        with self.assertRaises(ValueError):
            validate_manifest(records, rows)

    def test_checkpoint_provenance_rejects_held_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "base.pt"
            path.write_bytes(b"checkpoint")
            provenance = {"checkpoint_sha256": sha256(path), "training_data_sha256": "a",
                          "fold_manifest_sha256": "b", "external_test_labels_used": False,
                          "all_supervised_ancestors_included": True, "fitting_indices": [0, 1], "selection_indices": []}
            validate_checkpoint_provenance(provenance, path, "a", "b", [2], 3)
            provenance["selection_indices"] = [2]
            with self.assertRaises(ValueError):
                validate_checkpoint_provenance(provenance, path, "a", "b", [2], 3)

    def test_oof_evaluation_alignment_and_controls(self):
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_topology_aware_oof.py"
        spec = importlib.util.spec_from_file_location("evaluate_topology", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = [{"source_index": str(i), "sample_id": str(i), "sequence": "A" * (i+1),
                     "label": str(i % 2), "fold": str(i // 2), "group_id": str(i)} for i in range(10)]
        rows = [{**row, "variant": variant, "probability": .9 if row["label"] == "1" else .1}
                for variant in ("baseline", "local", "matching", "topology", "null") for row in manifest]
        result = module.evaluate(rows, manifest, resamples=10)
        self.assertEqual(result["metrics"]["baseline"]["mcc"], 1)
        with self.assertRaises(ValueError):
            module.evaluate(rows[:-1], manifest, resamples=10)

    def test_end_to_end_synthetic_five_fold_prototype(self):
        import csv
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = [record("CC" + "A" * (i + 1), str(i), i % 2) for i in range(10)]
            train_path, manifest_path = root / "train.pt", root / "folds.csv"
            torch.save({"records": items}, train_path)
            rows = [{"source_index": i, "sample_id": str(i), "sequence": item["sequence"],
                     "label": i % 2, "fold": i // 2, "group": str(i)} for i, item in enumerate(items)]
            with manifest_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            template = Path(__file__).resolve().parents[1] / "configs/topology_aware_structural_v1.json"
            protocol = json.loads(template.read_text())
            protocol.update(status="frozen_before_candidate_oof", processed_train=str(train_path),
                            training_data_sha256=sha256(train_path), fold_manifest=str(manifest_path),
                            fold_manifest_sha256=sha256(manifest_path), epochs=1, batch_size=5, stability_repeats=1)
            base_config = config()
            for fold in range(5):
                checkpoint = root / f"base{fold}.pt"
                torch.save({"config": base_config, "model_state": GHXToxModel(base_config).state_dict()}, checkpoint)
                provenance_path = root / f"provenance{fold}.json"
                provenance = {"checkpoint_sha256": sha256(checkpoint), "training_data_sha256": sha256(train_path),
                              "fold_manifest_sha256": sha256(manifest_path), "external_test_labels_used": False,
                              "all_supervised_ancestors_included": True,
                              "fitting_indices": [i for i in range(10) if i // 2 != fold], "selection_indices": []}
                provenance_path.write_text(json.dumps(provenance))
                protocol["checkpoints"].append({"fold": fold, "path": str(checkpoint), "provenance": str(provenance_path)})
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            output = root / "output"
            with patch("sys.argv", ["topology_oof", "--protocol", str(protocol_path), "--output-dir", str(output)]):
                run_oof()
            self.assertEqual(json.loads((output / "run_audit.json").read_text())["status"], "complete")
            with (output / "oof_predictions.csv").open() as handle:
                predicted = list(csv.DictReader(handle))
            self.assertEqual(len(predicted), 50)
            script = Path(__file__).resolve().parents[1] / "scripts/evaluate_topology_aware_oof.py"
            spec = importlib.util.spec_from_file_location("evaluate_topology_e2e", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with manifest_path.open() as handle:
                manifest = list(csv.DictReader(handle))
            result = module.evaluate(predicted, manifest, resamples=5)
            self.assertIsNotNone(result["stability"]["pairing_perturbation_mae"])


if __name__ == "__main__":
    unittest.main()
