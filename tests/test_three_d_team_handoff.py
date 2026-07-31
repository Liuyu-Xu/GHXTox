import csv
import json
from pathlib import Path

from ghxtox.three_d_team_handoff import build_three_d_team_handoff


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_three_d_handoff_preserves_frozen_probabilities_and_missing_mask(
    tmp_path: Path,
) -> None:
    folds = []
    predictions = []
    diagnostics = []
    for source_index, label in enumerate((0, 1, 0, 1)):
        sequence = "ACDEFG"
        fold = source_index % 2
        probability = 0.8 if label else 0.2
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
        predictions.append(
            {
                "source_index": source_index,
                "fold": fold,
                "sample_id": f"peptide_{source_index + 1}",
                "sequence": sequence,
                "label": label,
                "logit": 1.0 if label else -1.0,
                "toxicity_probability": probability,
                "probability_margin": 0.6,
                "probability_entropy": 0.5,
                "mc_mean_probability": probability,
                "mc_std_probability": 0.0,
                "mean_plddt": 0.8,
                "min_plddt": 0.7,
                "low_plddt_fraction": 0.0,
                "global_gate": 0.9,
                "sequence_length": 6,
            }
        )
        available = source_index < 2
        diagnostics.append(
            {
                "source_index": source_index,
                "sample_id": f"peptide_{source_index + 1}",
                "fold": fold,
                "group_id": f"group_{source_index}",
                "label": label,
                "sequence": sequence,
                "length": 6,
                "length_stratum": "11-15",
                "generator_source": "PepFlow" if available else "not_generated",
                "mean_residue_rmsf": 0.1 if available else "",
                "mean_conformer_rmsd": 0.2 if available else "",
                "mean_radius_gyration": 0.3 if available else "",
                "contact_uncertainty_8": 0.4 if available else "",
            }
        )
    fold_path = tmp_path / "folds.csv"
    prediction_path = tmp_path / "predictions.csv"
    diagnostic_path = tmp_path / "diagnostics.csv"
    decision_path = tmp_path / "decision.json"
    _write_csv(fold_path, folds)
    _write_csv(prediction_path, predictions)
    _write_csv(diagnostic_path, diagnostics)
    decision_path.write_text(json.dumps({"decision": "retain"}), encoding="utf-8")

    output = tmp_path / "handoff"
    report = build_three_d_team_handoff(
        prediction_path,
        fold_path,
        diagnostic_path,
        decision_path,
        output,
    )

    assert report["rows"] == 4
    assert report["conformer_available"] == 2
    assert report["missing_conformer"] == 2
    assert report["threshold_retuned"] is False
    with (output / "oof_3d_predictions.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["toxicity_probability"]) for row in rows] == [0.2, 0.8, 0.2, 0.8]
    assert [int(row["missing_conformer_mask"]) for row in rows] == [0, 0, 1, 1]
