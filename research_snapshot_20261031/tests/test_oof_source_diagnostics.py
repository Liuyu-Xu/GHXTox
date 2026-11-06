import csv
from pathlib import Path

import numpy as np

from ghxtox.oof_source_diagnostics import diagnose_oof_sources


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_oof_source_diagnostics_aligns_models_and_detects_confounding(
    tmp_path: Path,
) -> None:
    folds = []
    one_d = []
    three_d = []
    joint = []
    source_manifests = [[], [], []]
    cache_roots = [tmp_path / f"cache_{index}" for index in range(3)]
    lengths = (12, 20, 40)
    source_names = ["PepFlow", "AF2-16-30", "AF2-31-50-pilot"]
    source_index = 0
    for fold in (0, 1):
        for source_class, length in enumerate(lengths):
            for label in (0, 1):
                sample_id = f"peptide_{source_index + 1}"
                sequence = "A" * length
                folds.append(
                    {
                        "source_index": source_index,
                        "sample_id": sample_id,
                        "label": label,
                        "sequence": sequence,
                        "group_id": f"g{source_index}",
                        "fold": fold,
                    }
                )
                probability = 0.9 if label else 0.1
                one_d.append(
                    {
                        "source_index": source_index,
                        "sample_id": f"train_{source_index}",
                        "sequence": sequence,
                        "label": label,
                        "toxicity_probability": probability,
                    }
                )
                common = {
                    "source_index": source_index,
                    "fold": fold,
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "label": label,
                    "toxicity_probability": probability,
                }
                three_d.append(
                    {
                        **common,
                        "mean_plddt": 0.2 + 0.3 * source_class,
                        "min_plddt": 0.1 + 0.2 * source_class,
                        "low_plddt_fraction": 0.8 - 0.3 * source_class,
                        "global_gate": 0.2 + 0.1 * source_class,
                    }
                )
                joint.append(common)
                indexed_id = f"indexed_{source_index}"
                source_manifests[source_class].append(
                    {
                        "sample_id": indexed_id,
                        "original_id": sample_id,
                        "sequence": sequence,
                        "label": label,
                        "cache_path": "",
                    }
                )
                sample_root = cache_roots[source_class] / indexed_id
                sample_root.mkdir(parents=True)
                np.savez_compressed(
                    sample_root / "ensemble_summary.npz",
                    residue_rmsf=np.full(length, source_class + 1, dtype=np.float32),
                    conformer_rmsd=np.asarray([source_class + 0.1], dtype=np.float32),
                    radius_gyration=np.asarray([source_class + 1.0], dtype=np.float32),
                    contact_occupancy_8=np.full(
                        (length, length), 0.1 + 0.2 * source_class, dtype=np.float32
                    ),
                )
                source_index += 1

    fold_path = tmp_path / "folds.csv"
    one_d_path = tmp_path / "one_d.csv"
    three_d_path = tmp_path / "three_d.csv"
    joint_path = tmp_path / "joint.csv"
    manifest_paths = [tmp_path / f"source_{index}.csv" for index in range(3)]
    _write_csv(fold_path, folds)
    _write_csv(one_d_path, one_d)
    _write_csv(three_d_path, three_d)
    _write_csv(joint_path, joint)
    for path, rows in zip(manifest_paths, source_manifests):
        _write_csv(path, rows)

    report, rows = diagnose_oof_sources(
        one_d_path,
        three_d_path,
        joint_path,
        fold_path,
        manifest_paths,
        cache_roots,
        source_names,
    )

    assert len(rows) == 12
    assert set(report["by_length"]) == {"11-15", "16-30", "31-50"}
    assert report["confounding"]["covered_generator_vs_length_cramers_v"] == 1.0
    assert (
        report["overall"]["comparisons"]["joint_vs_one_d"]["decision_disagreement_rate"]
        == 0.0
    )
    assert (
        report["by_length"]["11-15"]["comparisons"]["joint_vs_one_d"]["net_corrected"]
        == 0
    )
    assert (
        report["feature_predictability"][
            "generator_cohort_from_structure_and_conformer_features"
        ]["balanced_accuracy"]
        == 1.0
    )
    assert report["protocol"]["external_test_labels_read"] is False
