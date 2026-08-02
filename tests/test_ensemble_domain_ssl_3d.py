import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ghxtox.conformer_cache import ensemble_descriptors
from ghxtox.ensemble_domain_ssl_3d import run_phase30


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, length: int, offset: float) -> None:
    path.parent.mkdir(parents=True)
    base = np.zeros((length, 3), dtype=np.float32)
    base[:, 0] = np.arange(length, dtype=np.float32) * 3.8
    conformers = np.stack(
        [base + (offset + conformer * 0.02) for conformer in range(3)], axis=0
    )
    np.savez_compressed(path, **ensemble_descriptors(conformers))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase30_is_fold_pure_and_preserves_exact_fallback(tmp_path: Path) -> None:
    folds = []
    legacy = []
    bioemu = []
    baseline = []
    legacy_root = tmp_path / "legacy"
    bioemu_root = tmp_path / "bioemu"
    source_index = 0
    for fold in (0, 1):
        for length in (12, 20, 40):
            for label in (0, 1):
                sample_id = f"peptide_{source_index + 1}"
                sequence = ("ACDEFGHIKLMNPQRSTVWY" * 2)[:length]
                folds.append(
                    {
                        "source_index": source_index,
                        "sample_id": f"train_{source_index}",
                        "label": label,
                        "sequence": sequence,
                        "group_id": f"group_{source_index}",
                        "fold": fold,
                    }
                )
                legacy_summary = legacy_root / sample_id / "ensemble_summary.npz"
                bioemu_summary = bioemu_root / sample_id / "ensemble_summary.npz"
                manifest_row = {
                    "source_index": source_index,
                    "sample_id": sample_id,
                    "original_id": sample_id,
                    "sequence": sequence,
                    "label": label,
                }
                legacy.append({**manifest_row, "cache_path": str(legacy_summary)})
                bioemu.append({**manifest_row, "cache_path": str(bioemu_summary)})
                baseline.append(
                    {
                        "source_index": source_index,
                        "sample_id": f"train_{source_index}",
                        "fold": fold,
                        "label": label,
                        "sequence": sequence,
                        "toxicity_probability": 0.8 if label else 0.2,
                    }
                )
                _write_summary(legacy_summary, length, label * 0.1)
                _write_summary(bioemu_summary, length, label * 0.1 + 0.01)
                source_index += 1

    for label, length in ((0, 12), (1, 40)):
        sequence = ("ACDEFGHIKLMNPQRSTVWY" * 2)[:length]
        baseline.append(
            {
                "source_index": source_index,
                "sample_id": f"train_{source_index}",
                "fold": source_index % 2,
                "label": label,
                "sequence": sequence,
                "toxicity_probability": 0.8 if label else 0.2,
            }
        )
        source_index += 1

    fold_path = tmp_path / "folds.csv"
    legacy_path = tmp_path / "legacy.csv"
    bioemu_path = tmp_path / "bioemu.csv"
    baseline_path = tmp_path / "baseline.csv"
    _write_csv(fold_path, folds)
    _write_csv(legacy_path, legacy)
    _write_csv(bioemu_path, bioemu)
    _write_csv(baseline_path, baseline)
    hashed = [fold_path, legacy_path, bioemu_path, baseline_path]
    protocol = {
        "schema_version": 1,
        "phase": 30,
        "status": "frozen_before_candidate_oof",
        "external_test_labels_read": False,
        "inputs": {
            "fold_manifest": str(fold_path),
            "legacy_manifests": [str(legacy_path)],
            "legacy_cache_roots": [str(legacy_root)],
            "legacy_source_names": ["Legacy"],
            "bioemu_manifest": str(bioemu_path),
            "bioemu_cache_root": str(bioemu_root),
            "reference_oof": str(baseline_path),
            "sha256": {str(path): _sha256(path) for path in hashed},
        },
        "cohort": {
            "canonical_union_samples": 12,
            "legacy_samples": 12,
            "bioemu_samples": 12,
            "matched_legacy_bioemu_pairs": 12,
            "bioemu_only_samples": 0,
        },
        "pretraining": {
            "structure_feature_dim": 16,
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "seed": 20260801,
            "max_conformers_per_source": 3,
            "matched_source_weight": 0.25,
        },
        "fine_tuning": {
            "hidden_dim": 8,
            "dropout": 0.0,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "delta_l1": 0.01,
            "maximum_absolute_logit_delta": 0.35,
        },
        "evaluation": {"decision_threshold": 0.5},
        "promotion_gate": {
            "global_mcc_strictly_above": 1.0,
            "global_auprc_at_least": 1.0,
            "maximum_brier": 1.0,
            "maximum_ece_10": 1.0,
            "minimum_fold_mcc_non_decreases": 1,
            "maximum_length_mcc_regression": 1.0,
            "maximum_length_auprc_regression": 1.0,
            "minimum_net_decision_corrections": 0,
            "maximum_matched_source_probe_bacc": 1.0,
            "minimum_matched_embedding_cosine": -1.0,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    output = tmp_path / "output"
    report, decision = run_phase30(protocol_path, output, device_name="cpu")

    assert report["cohort"]["samples"] == 12
    assert report["source_robustness"]["paired_rows"] == 12
    assert report["fallback"] == {
        "rows": 2,
        "maximum_absolute_probability_difference": 0.0,
    }
    assert decision["external_test_labels_read"] is False
    assert decision["full_bioemu_generation_authorized"] is False
    with (output / "oof_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 14
    assert sum(int(row["ssl_available"]) for row in rows) == 12
    assert all(
        row["reference_probability"] == row["candidate_probability"]
        for row in rows
        if row["ssl_available"] == "0"
    )
