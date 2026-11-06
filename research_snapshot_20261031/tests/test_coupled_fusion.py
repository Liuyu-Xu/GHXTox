from __future__ import annotations

import csv

from ghxtox.coupled_fusion import screen_coupled_fusion


def _write_predictions(path, probabilities, labels, folds=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source_index",
            "sample_id",
            "sequence",
            "label",
            "toxicity_probability",
            "global_3d_gate",
        ]
        if folds is not None:
            fieldnames.append("fold")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (probability, label) in enumerate(zip(probabilities, labels)):
            row = {
                "source_index": index,
                "sample_id": f"sample_{index}",
                "sequence": f"ACD{index}K",
                "label": label,
                "toxicity_probability": probability,
                "global_3d_gate": 0.7 if index % 2 == 0 else 0.3,
            }
            if folds is not None:
                row["fold"] = folds[index]
            writer.writerow(row)


def test_coupled_fusion_crossfits_and_writes_outputs(tmp_path) -> None:
    labels = [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    folds = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    one_d = [0.20, 0.35, 0.70, 0.74, 0.45, 0.40, 0.55, 0.65, 0.30, 0.62, 0.28, 0.68]
    three_d = [0.15, 0.42, 0.82, 0.62, 0.30, 0.25, 0.72, 0.58, 0.22, 0.78, 0.35, 0.60]
    one_d_path = tmp_path / "one_d.csv"
    three_d_path = tmp_path / "three_d.csv"
    _write_predictions(one_d_path, one_d, labels, folds=folds)
    _write_predictions(three_d_path, three_d, labels, folds=folds)

    summary = screen_coupled_fusion(
        one_d_predictions=one_d_path,
        three_d_predictions=three_d_path,
        output_dir=tmp_path / "coupled",
        hidden_dim=16,
        epochs=20,
        threshold_step=0.05,
        seed=7,
    )

    assert summary["protocol"]["test_labels_used_for_selection"] is False
    assert summary["crossfit_metrics"]["mcc"] >= -1.0
    assert (tmp_path / "coupled" / "summary.json").exists()
    assert (tmp_path / "coupled" / "crossfit_predictions.csv").exists()
    assert (tmp_path / "coupled" / "frozen_coupler.pt").exists()
