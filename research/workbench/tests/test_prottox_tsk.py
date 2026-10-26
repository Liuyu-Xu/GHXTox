"""Unit and CPU smoke tests for the isolated ProtTox-TSK route."""

from __future__ import annotations

import tempfile
import unittest
import csv
import json
from pathlib import Path
from unittest.mock import patch

import torch

from ghxtox.local_web_api import prottox_tsk_runtime_status
from ghxtox.prottox_tsk_cache import _select_records
from ghxtox.prottox_tsk_manifest import build_manifest
from ghxtox.models import PHYS_CHEM_FEATURE_NAMES, TriStatKANHead, TriStatMLPHead, physchem_feature_vector, tristat_pool
from ghxtox.models import SPECTRAL_FEATURE_NAMES, SpectralRefineConfig, SpectralRefineModel, spectral_physchem_feature_vector
from ghxtox.models import CysConditionalConfig, CysConditionalResidualModel, cys_regime
from ghxtox.models import LocalMotifConfig, LocalMotifResidualModel
from ghxtox.prottox_tsk import CACHE_FORMAT_VERSION, MODEL_NAME, MODEL_VERSION, file_sha256, load_checkpoint, sequence_sha256
from ghxtox.prottox_spectral_refine import load_checkpoint as load_spectral_checkpoint
from ghxtox.prottox_spectral_compare import compare as compare_spectral
from ghxtox.prottox_tsk_policy import apply_policy, build_policy, load_policy
from ghxtox.prottox_tsk_error_audit import audit as audit_oof_errors
from ghxtox.prottox_local_motif_audit import LOCAL_MOTIF_FEATURE_NAMES, audit as audit_local_motifs, local_motif_feature_vector
from ghxtox.prottox_local_motif_cache import CACHE_VERSION as LOCAL_MOTIF_CACHE_VERSION, cache_residue_embeddings
from ghxtox.train_prottox_local_motif import train as train_local_motif
from ghxtox.prottox_local_motif_compare import compare as compare_local_motif
from ghxtox.train_prottox_spectral_refine import train as train_spectral
from ghxtox.train_prottox_cys_conditional import train as train_cys_conditional
from ghxtox.train_prottox_tsk import train
from ghxtox.models.structure_quality_proxy import StructureQualityProxyConfig, StructureQualityProxyStudent
from ghxtox.prottox_struct_guided import AUXILIARY_TARGET_NAMES, build_cache as build_structure_guided_cache, load_checkpoint as load_structure_guided_checkpoint
from ghxtox.prottox_struct_guided_compare import _seed as structure_guided_seed, compare as compare_structure_guided
from ghxtox.train_prottox_struct_guided import train as train_structure_guided


class ProtToxTSKTest(unittest.TestCase):
    @staticmethod
    def _structure_guided_fixture(root: Path, records_count: int = 20) -> tuple[Path, Path]:
        base_cache, folds, structure_oof, output = root / "base.pt", root / "folds.csv", root / "structure.csv", root / "structure.pt"
        records, fold_rows, structure_rows = [], [], []
        for index in range(records_count):
            sequence = "ACD" + "K" * (index + 1)
            sample_id, historical_id, fold, label = f"s{index}", f"h{index}", (index // 2) % 5, index % 2
            records.append({"sample_id": sample_id, "sequence": sequence, "sequence_sha256": sequence_sha256(sequence), "label": label, "tristat": torch.randn(12), "physchem": torch.tensor(physchem_feature_vector(sequence), dtype=torch.float32)})
            fold_rows.append({"sample_id": sample_id, "historical_sample_id": historical_id, "sequence": sequence, "label": label, "group_id": f"g{index}", "fold": fold})
            structure_rows.append({"sample_id": historical_id, "sequence": sequence, "label": label, "fold": fold, "mean_plddt": 0.50 + 0.01 * (index % 5), "min_plddt": 0.30 + 0.01 * (index % 5), "low_plddt_fraction": 0.10 * (index % 5)})
        torch.save({"format_version": CACHE_FORMAT_VERSION, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "encoder": {"identifier": "synthetic", "frozen": True, "embedding_dim": 4}, "records": records}, base_cache)
        for path, rows, fields in ((folds, fold_rows, ["sample_id", "historical_sample_id", "sequence", "label", "group_id", "fold"]), (structure_oof, structure_rows, ["sample_id", "sequence", "label", "fold", *AUXILIARY_TARGET_NAMES])):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        build_structure_guided_cache(base_cache, folds, structure_oof, output)
        return base_cache, output

    def test_structure_guided_cache_binds_historical_targets_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_cache, cache = self._structure_guided_fixture(root)
            payload = torch.load(cache, map_location="cpu", weights_only=False)
            self.assertEqual(payload["base_cache"]["sha256"], file_sha256(base_cache))
            self.assertEqual(payload["records"][0]["historical_sample_id"], "h0")
            self.assertEqual(tuple(payload["records"][0]["structure_quality_targets"].shape), (3,))
            self.assertEqual(payload["structure_quality_source"]["scope"], "training-only offline structure-quality targets; not inference inputs")

    def test_structure_guided_cpu_training_and_checkpoint_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, cache = self._structure_guided_fixture(root)
            summary = train_structure_guided(cache, root / "candidate", variant="structure_quality_guided", epochs=1, learning_rate=1e-3, batch_size=4, shared_dim=12, proxy_hidden_dim=6, decision_hidden_dim=8, dropout=0.0, auxiliary_weight=0.1, seed=7, device_name="cpu")
            model, mean, std, aux_mean, aux_std, metadata = load_structure_guided_checkpoint(root / "candidate" / "final_model.pt")
            self.assertEqual(summary["cache"]["records"], 20)
            self.assertEqual(model.config.input_dim, 21)
            self.assertEqual(mean.numel(), 21)
            self.assertEqual(std.numel(), 21)
            self.assertEqual(aux_mean.numel(), 3)
            self.assertEqual(aux_std.numel(), 3)
            self.assertEqual(metadata["variant"], "structure_quality_guided")

    def test_structure_guided_promotion_is_paired_oof_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_paths, control_paths, candidate_paths = [], [], []
            for seed, gain in ((20260905, 0.006), (20260906, 0.005)):
                baseline = {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "seed": seed, "cache": {"sha256": "base"}, "fold_manifest": "folds.csv", "oof_metrics": {"mcc": 0.80, "auprc": 0.90, "brier": 0.06}}
                common = {"model_name": "ProtTox-StructGuide", "model_version": "prottox-struct-guide-v1", "seed": seed, "cache": {"sha256": "struct"}, "base_cache": {"sha256": "base"}, "fold_manifest": {"path": "folds.csv"}}
                control = {**common, "variant": "toxicity_only_control", "oof_metrics": {"mcc": 0.801, "auprc": 0.900, "brier": 0.059}}
                candidate = {**common, "variant": "structure_quality_guided", "oof_metrics": {"mcc": 0.80 + gain, "auprc": 0.899, "brier": 0.061}}
                paths = root / f"base_{seed}.json", root / f"control_{seed}.json", root / f"candidate_{seed}.json"
                for path, payload in zip(paths, (baseline, control, candidate), strict=True):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                baseline_paths.append(paths[0]); control_paths.append(paths[1]); candidate_paths.append(paths[2])
            report = compare_structure_guided(baseline_paths, control_paths, candidate_paths, root / "report.json")
            self.assertEqual(report["decision"], "promote_for_one_time_test1_audit")

    def test_structure_guided_comparator_recovers_legacy_baseline_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "legacy.pt"
            torch.save({"metadata": {"seed": 20261905}}, checkpoint)
            self.assertEqual(structure_guided_seed({"final_checkpoint": str(checkpoint)}), 20260905)

    def test_structure_guided_student_exposes_single_toxicity_logit_and_three_proxies(self) -> None:
        model = StructureQualityProxyStudent(StructureQualityProxyConfig(input_dim=21, shared_dim=12, proxy_hidden_dim=6, decision_hidden_dim=8, dropout=0.0))
        logit, proxies = model(torch.randn(4, 21))
        self.assertEqual(tuple(logit.shape), (4,))
        self.assertEqual(tuple(proxies.shape), (4, 3))

    def test_tristat_pool_ignores_padding(self) -> None:
        embeddings = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [999.0, 999.0]]])
        pooled = tristat_pool(embeddings, torch.tensor([[True, True, False]]))
        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 4.0, 3.0, 6.0, 1.0, 2.0]])))

    def test_physchem_contract(self) -> None:
        values = physchem_feature_vector("ACKC")
        self.assertEqual(len(values), len(PHYS_CHEM_FEATURE_NAMES))
        self.assertEqual(values[0], 4.0)
        self.assertEqual(values[6], 2.0)
        self.assertEqual(values[8], 2.0)

    def test_spectral_dct_contract_is_deterministic_and_finite(self) -> None:
        first = spectral_physchem_feature_vector("ACDEFGHIK")
        second = spectral_physchem_feature_vector("ACDEFGHIK")
        self.assertEqual(first.numel(), len(SPECTRAL_FEATURE_NAMES))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(spectral_physchem_feature_vector("A"), torch.zeros(len(SPECTRAL_FEATURE_NAMES))))

    def test_zero_initialized_gated_residual_equals_stage_one_at_start(self) -> None:
        model = SpectralRefineModel(SpectralRefineConfig(base_input_dim=21, spectral_dim=len(SPECTRAL_FEATURE_NAMES), variant="gated_residual", hidden_dim=12, spectral_hidden_dim=8, dropout=0.0)).eval()
        base, spectral = torch.randn(4, 21), torch.randn(4, len(SPECTRAL_FEATURE_NAMES))
        self.assertTrue(torch.equal(model(base, spectral), model.base_logit(base)))

    def test_cys_conditional_boundaries_and_zero_residual_invariance(self) -> None:
        self.assertTrue(torch.equal(cys_regime(torch.tensor([0, 1, 2, 3, 4, 7])), torch.tensor([0, 0, 1, 1, 2, 2])))
        model = CysConditionalResidualModel(CysConditionalConfig(base_input_dim=21, variant="cys_conditional_residual", hidden_dim=12, residual_hidden_dim=8, dropout=0.0)).eval()
        base, regimes = torch.randn(4, 21), torch.tensor([0, 1, 2, 0])
        self.assertTrue(torch.equal(model(base, regimes), model.base_logit(base)))

    def test_local_motif_zero_residual_invariance_and_matched_control(self) -> None:
        base, residues = torch.randn(3, 21), torch.randn(3, 5, 4)
        mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False], [True, True, True, True, False]])
        candidate = LocalMotifResidualModel(LocalMotifConfig(base_input_dim=21, residue_embedding_dim=4, variant="multiscale_cnn_attention", hidden_dim=12, local_dim=12, convolution_dim=4, residual_hidden_dim=8, dropout=0.0)).eval()
        control = LocalMotifResidualModel(LocalMotifConfig(base_input_dim=21, residue_embedding_dim=4, variant="nonlocal_projection_control", hidden_dim=12, local_dim=12, convolution_dim=4, residual_hidden_dim=8, dropout=0.0)).eval()
        self.assertTrue(torch.equal(candidate(base, residues, mask), candidate.base_logit(base)))
        self.assertTrue(torch.equal(control(base, residues, mask), control.base_logit(base)))
        self.assertLess(abs(candidate.local_branch_parameter_count() - control.local_branch_parameter_count()), 60)

    def test_heads_emit_one_logit_each(self) -> None:
        inputs = torch.randn(3, 21)
        self.assertEqual(tuple(TriStatMLPHead(21, hidden_dim=12)(inputs).shape), (3,))
        self.assertEqual(tuple(TriStatKANHead(21, hidden_dim=12, grid_size=5)(inputs).shape), (3,))

    def test_cpu_fold_training_and_checkpoint_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "cache.pt"
            records = []
            for index in range(10):
                sequence = "ACKC" if index % 2 else "VVVV"
                records.append({"sample_id": f"s{index}", "sequence": sequence, "sequence_sha256": sequence_sha256(sequence), "label": index % 2, "tristat": torch.randn(12), "physchem": torch.tensor(physchem_feature_vector(sequence), dtype=torch.float32)})
            torch.save({"format_version": CACHE_FORMAT_VERSION, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "encoder": {"identifier": "synthetic", "frozen": True, "embedding_dim": 4}, "records": records}, cache_path)
            manifest = root / "folds.csv"
            manifest.write_text("sample_id,fold\n" + "\n".join(f"s{index},{(index // 2) % 2}" for index in range(10)) + "\n", encoding="utf-8")
            summary = train(cache_path, root / "run", head_kind="mlp", fold_manifest=str(manifest), folds=2, epochs=1, learning_rate=1e-3, hidden_dim=12, grid_size=5, dropout=0.0, seed=7, device_name="cpu")
            self.assertEqual(summary["cache"]["records"], 10)
            model, mean, std, metadata = load_checkpoint(root / "run" / "final_model.pt")
            self.assertEqual(model.config.head_kind, "mlp")
            self.assertEqual(mean.numel(), 21)
            self.assertEqual(std.numel(), 21)
            self.assertEqual(metadata["trained_records"], 10)

    def test_cpu_spectral_training_uses_audited_five_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "cache.pt"
            records = []
            for index in range(20):
                sequence = "ACKC" if index % 2 else "VVVV"
                records.append({"sample_id": f"s{index}", "sequence": sequence, "sequence_sha256": sequence_sha256(sequence), "label": index % 2, "tristat": torch.randn(12), "physchem": torch.tensor(physchem_feature_vector(sequence), dtype=torch.float32)})
            torch.save({"format_version": CACHE_FORMAT_VERSION, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "encoder": {"identifier": "synthetic", "frozen": True, "embedding_dim": 4}, "records": records}, cache_path)
            manifest = root / "folds.csv"
            manifest.write_text("sample_id,fold\n" + "\n".join(f"s{index},{(index // 2) % 5}" for index in range(20)) + "\n", encoding="utf-8")
            summary = train_spectral(cache_path, root / "spectral", variant="gated_residual", fold_manifest=str(manifest), folds=5, epochs=1, learning_rate=1e-3, hidden_dim=12, spectral_hidden_dim=8, dropout=0.0, seed=7, device_name="cpu")
            self.assertEqual(summary["variant"], "gated_residual")
            model, base_mean, base_std, spectral_mean, spectral_std, metadata = load_spectral_checkpoint(root / "spectral" / "final_model.pt")
            self.assertEqual(model.config.spectral_dim, len(SPECTRAL_FEATURE_NAMES))
            self.assertEqual(base_mean.numel(), 21)
            self.assertEqual(spectral_mean.numel(), len(SPECTRAL_FEATURE_NAMES))
            self.assertEqual(metadata["variant"], "gated_residual")

    def test_cpu_cys_conditional_training_uses_audited_five_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "cache.pt"
            records = []
            for index in range(20):
                sequence = "ACKC" if index % 2 else "VVVV"
                records.append({"sample_id": f"s{index}", "sequence": sequence, "sequence_sha256": sequence_sha256(sequence), "label": index % 2, "tristat": torch.randn(12), "physchem": torch.tensor(physchem_feature_vector(sequence), dtype=torch.float32)})
            torch.save({"format_version": CACHE_FORMAT_VERSION, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "encoder": {"identifier": "synthetic", "frozen": True, "embedding_dim": 4}, "records": records}, cache_path)
            manifest = root / "folds.csv"
            manifest.write_text("sample_id,fold\n" + "\n".join(f"s{index},{(index // 2) % 5}" for index in range(20)) + "\n", encoding="utf-8")
            summary = train_cys_conditional(cache_path, root / "cys", variant="cys_conditional_residual", fold_manifest=str(manifest), folds=5, epochs=1, learning_rate=1e-3, hidden_dim=12, residual_hidden_dim=8, dropout=0.0, seed=7, device_name="cpu")
            self.assertEqual(summary["variant"], "cys_conditional_residual")
            self.assertEqual(summary["regime_counts"]["cys_0_1"] + summary["regime_counts"]["cys_2_3"] + summary["regime_counts"]["cys_4_plus"], 20)

    def test_web_status_does_not_accept_historical_assets(self) -> None:
        previous_checkpoint = __import__("os").environ.pop("GHXTOX_PROTTOX_TSK_CHECKPOINT", None)
        previous_runner = __import__("os").environ.pop("GHXTOX_PROTTOX_TSK_RUNNER", None)
        try:
            status = prottox_tsk_runtime_status()
        finally:
            if previous_checkpoint is not None:
                __import__("os").environ["GHXTOX_PROTTOX_TSK_CHECKPOINT"] = previous_checkpoint
            if previous_runner is not None:
                __import__("os").environ["GHXTOX_PROTTOX_TSK_RUNNER"] = previous_runner
        self.assertEqual(status["state"], "configuration_required")
        self.assertFalse(status["model"]["legacy_ghxtox_components_enabled"])
        self.assertFalse(status["model"]["quick_full_workflow_enabled"])

    def test_smoke_selection_is_class_balanced(self) -> None:
        records = [{"sample_id": f"s{index}", "sequence": "ACD", "label": index % 2} for index in range(12)]
        selected = _select_records(records, 6, stratify_labels=True, seed=5)
        self.assertEqual(len(selected), 6)
        self.assertEqual(sum(record["label"] == 0 for record in selected), 3)
        self.assertEqual(sum(record["label"] == 1 for record in selected), 3)

    def test_manifest_maps_by_sequence_not_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            torch.save({"sample_ids": ["new_b", "new_a", "new_c", "new_d", "new_e"], "sequences": ["BBBB", "AAAA", "CCCC", "DDDD", "EEEE"], "labels": [1, 0, 0, 1, 0]}, source)
            historical = root / "historical.csv"
            historical.write_text("sample_id,sequence,label,group_id,fold\nold_a,AAAA,0,g1,0\nold_b,BBBB,1,g2,1\nold_c,CCCC,0,g3,2\nold_d,DDDD,1,g4,3\nold_e,EEEE,0,g5,4\n", encoding="utf-8")
            audit = build_manifest(source, historical, root / "mapped.csv")
            rows = (root / "mapped.csv").read_text(encoding="utf-8").splitlines()
            self.assertIn("new_b,BBBB,1,g2,1,old_b", rows[1])
            self.assertEqual(audit["checks"]["sequence_mapping"], "exact_one_to_one")

    def test_training_oof_policy_is_checkpoint_bound_and_has_three_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "cache.pt"
            records = []
            for index in range(20):
                sequence = "ACKC" if index % 2 else "VVVV"
                records.append({"sample_id": f"s{index}", "sequence": sequence, "sequence_sha256": sequence_sha256(sequence), "label": index % 2, "tristat": torch.randn(12), "physchem": torch.tensor(physchem_feature_vector(sequence), dtype=torch.float32)})
            torch.save({"format_version": CACHE_FORMAT_VERSION, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "encoder": {"identifier": "synthetic", "frozen": True, "embedding_dim": 4}, "records": records}, cache_path)
            manifest = root / "folds.csv"
            manifest.write_text("sample_id,fold\n" + "\n".join(f"s{index},{(index // 2) % 5}" for index in range(20)) + "\n", encoding="utf-8")
            train(cache_path, root / "run", head_kind="mlp", fold_manifest=str(manifest), folds=5, epochs=1, learning_rate=1e-3, hidden_dim=12, grid_size=5, dropout=0.0, seed=7, device_name="cpu")
            policy_path = root / "policy.json"
            artifact = build_policy(cache_path, root / "run" / "oof_predictions.csv", root / "run" / "final_model.pt", policy_path)
            self.assertEqual(set(artifact["policies"]), {"screening", "balanced", "strict"})
            policy = load_policy(policy_path, root / "run" / "final_model.pt", mode="screening")
            calibrated, threshold = apply_policy([0.2, 0.8], policy, mode="screening")
            self.assertEqual(tuple(calibrated.shape), (2,))
            self.assertGreaterEqual(threshold, 0.0)
            self.assertLessEqual(threshold, 1.0)

    def test_policy_rejects_oof_misaligned_to_training_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "cache.pt"
            records = [{"sample_id": f"s{index}", "sequence": "ACKC", "sequence_sha256": sequence_sha256("ACKC"), "label": index % 2, "tristat": torch.randn(12), "physchem": torch.tensor(physchem_feature_vector("ACKC"), dtype=torch.float32)} for index in range(10)]
            torch.save({"format_version": CACHE_FORMAT_VERSION, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "encoder": {"identifier": "synthetic", "frozen": True, "embedding_dim": 4}, "records": records}, cache_path)
            oof_path = root / "oof.csv"
            with oof_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence_sha256", "label", "fold", "probability"])
                writer.writeheader()
                for index, record in enumerate(records):
                    writer.writerow({"sample_id": record["sample_id"], "sequence_sha256": record["sequence_sha256"], "label": 1 - record["label"] if index == 0 else record["label"], "fold": index % 5, "probability": 0.8 if record["label"] else 0.2})
            with self.assertRaisesRegex(ValueError, "identity/label mismatch"):
                build_policy(cache_path, oof_path, root / "missing_checkpoint.pt", root / "policy.json")

    def test_spectral_promotion_rule_is_paired_and_oof_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_paths, candidate_paths = [], []
            for seed, mcc_delta in ((20260905, 0.006), (20260906, 0.005)):
                baseline = {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "seed": seed, "cache": {"sha256": "cache"}, "fold_manifest": "folds.csv", "oof_metrics": {"mcc": 0.80, "auprc": 0.90, "brier": 0.06}}
                candidate = {"model_name": "ProtTox-SpectralRefine", "model_version": "prottox-spectral-refine-v1", "variant": "gated_residual", "seed": seed, "cache": {"sha256": "cache"}, "fold_manifest": "folds.csv", "oof_metrics": {"mcc": 0.80 + mcc_delta, "auprc": 0.899, "brier": 0.058}}
                base_path, candidate_path = root / f"base_{seed}.json", root / f"candidate_{seed}.json"
                base_path.write_text(json.dumps(baseline), encoding="utf-8")
                candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
                baseline_paths.append(base_path)
                candidate_paths.append(candidate_path)
            report = compare_spectral(baseline_paths, candidate_paths, root / "report.json")
            self.assertEqual(report["decision"], "promote_for_one_time_test1_audit")

    def test_oof_error_audit_aligns_records_and_detects_stable_false_negative_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path, oof_path, folds_path = root / "cache.pt", root / "oof.csv", root / "folds.csv"
            records, oof_rows, fold_rows = [], [], []
            for fold in range(5):
                for kind, label, probability, sequences in (("fn", 1, 0.1, ("ACDEFGHIKL", "ACDEFGHIKLM")), ("tp", 1, 0.9, ("ACDE", "ACDEF")), ("fp", 0, 0.9, ("VVVVVVVVVV", "VVVVVVVVVVV")), ("tn", 0, 0.1, ("VVVV", "VVVVV"))):
                    for repeat in range(2):
                        sequence = sequences[repeat]
                        sample_id = f"{kind}_{fold}_{repeat}"
                        records.append({"sample_id": sample_id, "sequence": sequence, "sequence_sha256": sequence_sha256(sequence), "label": label, "physchem": torch.tensor(physchem_feature_vector(sequence), dtype=torch.float32)})
                        oof_rows.append({"sample_id": sample_id, "sequence_sha256": sequence_sha256(sequence), "label": label, "fold": fold, "probability": probability})
                        fold_rows.append({"sample_id": sample_id, "fold": fold})
            torch.save({"format_version": CACHE_FORMAT_VERSION, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "records": records}, cache_path)
            for path, rows, fields in ((oof_path, oof_rows, ["sample_id", "sequence_sha256", "label", "fold", "probability"]), (folds_path, fold_rows, ["sample_id", "fold"])):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            report = audit_oof_errors(cache_path, oof_path, folds_path, root / "audit")
            length = next(item for item in report["comparisons"]["false_negative_vs_true_positive"]["features"] if item["feature"] == "length")
            self.assertGreater(length["standardized_mean_difference"], 0.20)
            self.assertTrue(length["stable"])
            self.assertEqual(sum(row["records"] for row in report["cysteine_regimes"]), len(records))

    def test_local_motif_audit_is_fold_pure_and_detects_order_residual(self) -> None:
        self.assertEqual(local_motif_feature_vector("KAAAVVK").numel(), len(LOCAL_MOTIF_FEATURE_NAMES))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path, oof_path, folds_path = root / "cache.pt", root / "oof.csv", root / "folds.csv"
            records, oof_rows, fold_rows = [], [], []
            for fold in range(5):
                for kind, probability, sequences in (
                    ("fn", 0.1, ("KAAAVVK", "AAAVKAK")),
                    ("tp", 0.9, ("AAKVVKA", "AAAVKKA")),
                ):
                    for repeat, sequence in enumerate(sequences):
                        sample_id = f"{kind}_{fold}_{repeat}"
                        records.append({"sample_id": sample_id, "sequence": sequence, "sequence_sha256": sequence_sha256(sequence), "label": 1, "physchem": torch.tensor(physchem_feature_vector(sequence), dtype=torch.float32)})
                        oof_rows.append({"sample_id": sample_id, "sequence_sha256": sequence_sha256(sequence), "label": 1, "fold": fold, "probability": probability})
                        fold_rows.append({"sample_id": sample_id, "fold": fold})
            torch.save({"format_version": CACHE_FORMAT_VERSION, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "records": records}, cache_path)
            for path, rows, fields in ((oof_path, oof_rows, ["sample_id", "sequence_sha256", "label", "fold", "probability"]), (folds_path, fold_rows, ["sample_id", "fold"])):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            report = audit_local_motifs(cache_path, oof_path, folds_path, root / "audit")
            self.assertTrue(report["architecture_premise_supported"])
            stable = report["low_cys_false_negative_vs_true_positive"]["global_residualized"]["stable_features"]
            self.assertIn("residual_longest_hydrophobic_run_fraction", {item["feature"] for item in stable})

    def test_local_motif_residue_cache_binds_to_base_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_cache = root / "base.pt"
            records = []
            for index, sequence in enumerate(("ACKC", "VVVV")):
                records.append({"sample_id": f"s{index}", "sequence": sequence, "sequence_sha256": sequence_sha256(sequence), "label": index % 2, "tristat": torch.randn(12), "physchem": torch.tensor(physchem_feature_vector(sequence), dtype=torch.float32)})
            torch.save({"format_version": CACHE_FORMAT_VERSION, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "encoder": {"identifier": "prot_t5_xl_half_uniref50-enc", "frozen": True, "embedding_dim": 4}, "source": {"path": "synthetic", "sha256": "source"}, "records": records}, base_cache)
            embeddings = [torch.randn(len(record["sequence"]), 4) for record in records]
            with patch("ghxtox.prottox_local_motif_cache._residue_embeddings", return_value=(embeddings, "prot_t5_xl_half_uniref50-enc")):
                manifest = cache_residue_embeddings(base_cache, root / "residue.pt", model_name="prot_t5_xl_half_uniref50-enc", model_path="synthetic", device="cpu", batch_size=2)
            payload = torch.load(root / "residue.pt", map_location="cpu", weights_only=False)
            self.assertEqual(manifest["format_version"], LOCAL_MOTIF_CACHE_VERSION)
            self.assertEqual(payload["base_cache"]["sha256"], file_sha256(base_cache))
            self.assertEqual(payload["records"][0]["residue_embeddings"].dtype, torch.float16)

    def test_cpu_local_motif_training_uses_audited_five_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "residue.pt"
            records = []
            for index in range(20):
                sequence = "ACKC" if index % 2 else "VVVV"
                records.append({"sample_id": f"s{index}", "sequence": sequence, "sequence_sha256": sequence_sha256(sequence), "label": index % 2, "tristat": torch.randn(12), "physchem": torch.tensor(physchem_feature_vector(sequence), dtype=torch.float32), "residue_embeddings": torch.randn(len(sequence), 4).half()})
            torch.save({"format_version": LOCAL_MOTIF_CACHE_VERSION, "model_name": "ProtTox-LocalMotif", "model_version": LOCAL_MOTIF_CACHE_VERSION, "base_cache": {"path": "base", "sha256": "base-cache"}, "encoder": {"identifier": "prot_t5_xl_half_uniref50-enc", "frozen": True, "embedding_dim": 4, "storage_dtype": "float16"}, "records": records}, cache_path)
            manifest = root / "folds.csv"
            manifest.write_text("sample_id,fold\n" + "\n".join(f"s{index},{(index // 2) % 5}" for index in range(20)) + "\n", encoding="utf-8")
            summary = train_local_motif(cache_path, root / "local", variant="multiscale_cnn_attention", fold_manifest=str(manifest), folds=5, epochs=1, learning_rate=1e-3, hidden_dim=12, local_dim=12, convolution_dim=4, residual_hidden_dim=8, dropout=0.0, batch_size=4, seed=7, device_name="cpu")
            self.assertEqual(summary["variant"], "multiscale_cnn_attention")
            self.assertEqual(summary["cache"]["records"], 20)

    def test_local_motif_promotion_rule_is_paired_and_oof_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_paths, control_paths, candidate_paths = [], [], []
            for seed, gain in ((20260905, 0.006), (20260906, 0.005)):
                baseline = {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "seed": seed, "cache": {"sha256": "base"}, "fold_manifest": "folds.csv", "oof_metrics": {"mcc": 0.80, "auprc": 0.90, "brier": 0.06}}
                common = {"model_name": "ProtTox-LocalMotif", "model_version": "prottox-local-motif-v1", "seed": seed, "cache": {"sha256": "residue"}, "base_cache": {"sha256": "base"}, "fold_manifest": "folds.csv", "parameter_count": 1000, "local_branch_parameter_count": 200}
                control = {**common, "variant": "nonlocal_projection_control", "oof_metrics": {"mcc": 0.801, "auprc": 0.900, "brier": 0.059}}
                candidate = {**common, "variant": "multiscale_cnn_attention", "local_branch_parameter_count": 200, "oof_metrics": {"mcc": 0.80 + gain, "auprc": 0.899, "brier": 0.058}}
                paths = (root / f"base_{seed}.json", root / f"control_{seed}.json", root / f"candidate_{seed}.json")
                for path, payload in zip(paths, (baseline, control, candidate), strict=True):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                baseline_paths.append(paths[0])
                control_paths.append(paths[1])
                candidate_paths.append(paths[2])
            report = compare_local_motif(baseline_paths, control_paths, candidate_paths, root / "report.json")
            self.assertEqual(report["decision"], "promote_for_one_time_test1_audit")


if __name__ == "__main__":
    unittest.main()
