"""Package the retained 3D OOF stream for leakage-safe team fusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.bootstrap_ci import _metrics
from ghxtox.oof_source_diagnostics import _read_csv
from ghxtox.utils import save_json


DEFAULT_THRESHOLD = 0.677819


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_three_d_team_handoff(
    retained_oof: str | Path,
    fold_manifest: str | Path,
    diagnostic_rows: str | Path,
    model_decision: str | Path,
    output_root: str | Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Join frozen 3D predictions and train-only quality metadata exactly once."""

    predictions = {int(row["source_index"]): row for row in _read_csv(retained_oof)}
    folds = {int(row["source_index"]): row for row in _read_csv(fold_manifest)}
    diagnostics = {int(row["source_index"]): row for row in _read_csv(diagnostic_rows)}
    expected = set(range(len(folds)))
    if set(predictions) != expected or set(diagnostics) != expected:
        raise ValueError(
            "Predictions, diagnostics, and folds must cover the same rows."
        )

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    contract_rows = []
    quality_rows = []
    labels = []
    probabilities = []
    for source_index in sorted(expected):
        prediction = predictions[source_index]
        fold = folds[source_index]
        diagnostic = diagnostics[source_index]
        label = int(fold["label"])
        sequence = str(fold["sequence"])
        canonical_fold = int(fold["fold"])
        if int(prediction["label"]) != label or int(diagnostic["label"]) != label:
            raise ValueError(f"Label mismatch at source_index {source_index}.")
        if (
            str(prediction["sequence"]) != sequence
            or str(diagnostic["sequence"]) != sequence
        ):
            raise ValueError(f"Sequence mismatch at source_index {source_index}.")
        if (
            int(prediction["fold"]) != canonical_fold
            or int(diagnostic["fold"]) != canonical_fold
        ):
            raise ValueError(f"Fold mismatch at source_index {source_index}.")
        probability = float(prediction["toxicity_probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid probability at source_index {source_index}.")
        generator = str(diagnostic["generator_source"])
        conformer_available = generator != "not_generated"
        quality = {
            "source_index": source_index,
            "fold": canonical_fold,
            "length": int(diagnostic["length"]),
            "length_stratum": diagnostic["length_stratum"],
            "base_structure_available": 1,
            "conformer_available": int(conformer_available),
            "missing_conformer_mask": int(not conformer_available),
            "generator_source": generator,
            "mean_plddt": prediction["mean_plddt"],
            "min_plddt": prediction["min_plddt"],
            "low_plddt_fraction": prediction["low_plddt_fraction"],
            "global_gate": prediction["global_gate"],
            "mean_residue_rmsf": diagnostic["mean_residue_rmsf"],
            "mean_conformer_rmsd": diagnostic["mean_conformer_rmsd"],
            "mean_radius_gyration": diagnostic["mean_radius_gyration"],
            "contact_uncertainty_8": diagnostic["contact_uncertainty_8"],
        }
        quality_rows.append(quality)
        contract_rows.append(
            {
                "source_index": source_index,
                "sample_id": prediction["sample_id"],
                "fold": canonical_fold,
                "label": label,
                "sequence": sequence,
                "toxicity_probability": probability,
                "logit": prediction["logit"],
                "prediction": int(probability >= threshold),
                "decision_threshold": threshold,
                "probability_margin": prediction["probability_margin"],
                "probability_entropy": prediction["probability_entropy"],
                "mc_mean_probability": prediction["mc_mean_probability"],
                "mc_std_probability": prediction["mc_std_probability"],
                **{
                    key: value
                    for key, value in quality.items()
                    if key not in {"source_index", "fold"}
                },
            }
        )
        labels.append(label)
        probabilities.append(probability)

    prediction_path = output / "oof_3d_predictions.csv"
    quality_path = output / "oof_3d_quality.csv"
    _write_csv(contract_rows, prediction_path)
    _write_csv(quality_rows, quality_path)
    source_counts = {}
    for row in quality_rows:
        source = str(row["generator_source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    report = {
        "status": "complete",
        "artifact": "retained_capability_aware_3d_oof_team_handoff",
        "external_test_labels_read": False,
        "fusion_weights_included": False,
        "probabilities_recalibrated": False,
        "threshold_retuned": False,
        "fixed_threshold": threshold,
        "rows": len(contract_rows),
        "positive": int(sum(labels)),
        "negative": int(len(labels) - sum(labels)),
        "conformer_available": int(
            sum(row["conformer_available"] for row in quality_rows)
        ),
        "missing_conformer": int(
            sum(row["missing_conformer_mask"] for row in quality_rows)
        ),
        "generator_source_counts": source_counts,
        "metrics_at_fixed_threshold": _metrics(
            np.asarray(labels, dtype=np.int64),
            np.asarray(probabilities, dtype=np.float64),
            threshold,
        ),
        "provenance": {
            "retained_oof": str(retained_oof),
            "retained_oof_sha256": _sha256(retained_oof),
            "fold_manifest": str(fold_manifest),
            "fold_manifest_sha256": _sha256(fold_manifest),
            "diagnostic_rows": str(diagnostic_rows),
            "diagnostic_rows_sha256": _sha256(diagnostic_rows),
            "model_decision": str(model_decision),
            "model_decision_sha256": _sha256(model_decision),
            "output_predictions_sha256": _sha256(prediction_path),
            "output_quality_sha256": _sha256(quality_path),
        },
        "handoff_rules": [
            "Join only by source_index and verify label/fold alignment.",
            "Use toxicity_probability as the standalone 3D OOF score.",
            "Use the repository nested-OOF fusion protocol for performance claims; fitting and scoring a meta-model on this same OOF table is biased.",
            "A plain outer split of this table is not sufficient if upstream base predictions for meta-training rows were fit using the outer evaluation fold.",
            "Use conformer_available and missing_conformer_mask explicitly; blank ensemble fields are not zeros.",
            "Do not retune the retained 3D threshold inside this package.",
            "Never fit fusion parameters on Test1/Test2.",
        ],
        "schema": {
            "oof_3d_predictions.csv": list(contract_rows[0]),
            "oof_3d_quality.csv": list(quality_rows[0]),
        },
    }
    save_json(report, output / "handoff_contract.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-oof", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--diagnostic-rows", required=True)
    parser.add_argument("--model-decision", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    build_three_d_team_handoff(
        args.retained_oof,
        args.fold_manifest,
        args.diagnostic_rows,
        args.model_decision,
        args.output_root,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
