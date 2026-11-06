import csv
from pathlib import Path

import numpy as np

from ghxtox.conformer_cache import ensemble_descriptors
from ghxtox.multiconformer_graph_3d import (
    build_multiconformer_samples,
    run_multiconformer_graph_oof,
)


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
        [base + (offset + index * 0.03) for index in range(4)], axis=0
    )
    np.savez_compressed(path, **ensemble_descriptors(conformers))


def test_multiconformer_graph_builds_fold_safe_handoff(tmp_path: Path) -> None:
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
                legacy_path = legacy_root / sample_id / "ensemble_summary.npz"
                bioemu_path = bioemu_root / sample_id / "ensemble_summary.npz"
                common = {
                    "source_index": source_index,
                    "sample_id": sample_id,
                    "original_id": sample_id,
                    "sequence": sequence,
                    "label": label,
                }
                legacy.append({**common, "cache_path": str(legacy_path)})
                bioemu.append({**common, "cache_path": str(bioemu_path)})
                baseline.append(
                    {
                        "source_index": source_index,
                        "label": label,
                        "toxicity_probability": 0.8 if label else 0.2,
                    }
                )
                _write_summary(legacy_path, length, label * 0.2)
                _write_summary(bioemu_path, length, label * 0.2 + 0.01)
                source_index += 1

    fold_path = tmp_path / "folds.csv"
    legacy_path = tmp_path / "legacy.csv"
    bioemu_path = tmp_path / "bioemu.csv"
    baseline_path = tmp_path / "baseline.csv"
    _write_csv(fold_path, folds)
    _write_csv(legacy_path, legacy)
    _write_csv(bioemu_path, bioemu)
    _write_csv(baseline_path, baseline)

    samples, metadata = build_multiconformer_samples(
        fold_path,
        [legacy_path],
        [legacy_root],
        ["Legacy"],
        bioemu_path,
        bioemu_root,
    )
    assert len(samples) == 12
    assert metadata["paired_samples"] == 12
    assert all(sample.primary.source == "Legacy" for sample in samples)

    output = tmp_path / "output"
    report, decision = run_multiconformer_graph_oof(
        fold_path,
        [legacy_path],
        [legacy_root],
        ["Legacy"],
        bioemu_path,
        bioemu_root,
        baseline_path,
        output,
        epochs=1,
        batch_size=4,
        hidden_dim=8,
        embedding_dim=8,
        layers=1,
        device_name="cpu",
    )

    assert report["cohort"]["samples"] == 12
    assert report["cohort"]["paired_samples"] == 12
    assert report["protocol"]["external_test_labels_read"] is False
    assert decision["full_bioemu_generation_authorized"] is False
    assert (output / "oof_predictions.csv").exists()
    assert (output / "bioemu_oof_views.csv").exists()
    assert (output / "full_handoff_manifest.csv").exists()
    with np.load(output / "full_oof_embeddings.npz") as payload:
        assert payload["embeddings"].shape == (12, 8)
        assert payload["structure_available"].all()
