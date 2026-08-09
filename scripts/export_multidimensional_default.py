# -*- coding: utf-8 -*-
"""Export the canonical public 1D/3D fusion with strict alignment checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghxtox.multidimensional_default import assemble
from ghxtox.utils import (
    DEFAULT_MULTIDIMENSIONAL_MODEL_NAME,
    DEFAULT_MULTIDIMENSIONAL_THRESHOLD,
    DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D)
    parser.add_argument("--threshold", type=float, default=DEFAULT_MULTIDIMENSIONAL_THRESHOLD)
    parser.add_argument("--allow-noncanonical-operating-point", action="store_true")
    parser.add_argument(
        "--one-d-dir",
        default="runs/source_aware_multiconformer_frozen_test_20260808_r4_corrected",
    )
    parser.add_argument(
        "--structure-dir",
        default="runs/source_aware_multiconformer_frozen_test_20260808_r4_corrected",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/default_multidimensional_fusion_full2795",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "artifact_type": "ghxtox_multidimensional_public_default",
        "schema_version": 1,
        "model": DEFAULT_MULTIDIMENSIONAL_MODEL_NAME,
        "weight_1d": float(args.weight),
        "weight_3d": float(1.0 - args.weight),
        "threshold": float(args.threshold),
        "components": {
            "1d": "calibrated three-seed LoRA-ProtT5 + bidirectional Mamba",
            "3d": (
                "pure structure_probability from the pLDDT-aware 3D expert "
                "with the retained audited multi-conformer residual"
            ),
        },
    }
    for dataset in ("test1", "test2"):
        one_d_path = Path(args.one_d_dir) / f"{dataset}_declared_1d_probabilities.csv"
        structure_path = Path(args.structure_dir) / f"{dataset}_structure_predictions.csv"
        prediction_path = output_dir / f"{dataset}_predictions.csv"
        metrics_path = output_dir / f"{dataset}_metrics.json"
        result = assemble(
            args.weight,
            args.threshold,
            one_d_path,
            structure_path,
            prediction_path,
            metrics_path,
            allow_noncanonical=args.allow_noncanonical_operating_point,
        )
        summary[dataset] = {
            "n": result["n"],
            "metrics": result["metrics"],
            "one_d_source": str(one_d_path),
            "structure_source": str(structure_path),
        }
        metrics = result["metrics"]
        print(
            f"{dataset} n={result['n']} BACC={metrics['balanced_accuracy']:.4f} "
            f"MCC={metrics['mcc']:.4f} AUROC={metrics['auroc']:.4f} "
            f"AUPRC={metrics['auprc']:.4f}"
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
