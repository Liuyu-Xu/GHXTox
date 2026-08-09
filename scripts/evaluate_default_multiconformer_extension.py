"""Evaluate the frozen public fusion with an optional multi-conformer 3D residual.

This script never refits the 1D expert, fusion weight, calibration, or threshold.
It only runs the already-trained 3D checkpoint on processed caches carrying
audited conformer descriptors and then assembles the canonical 0.72/0.28
fusion from the frozen calibrated 1D probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ghxtox.multidimensional_default import _metrics, assemble
from ghxtox.structure_inference import predict_structure
from ghxtox.utils import (
    DEFAULT_MULTIDIMENSIONAL_THRESHOLD,
    DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D,
)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def _sequence_sources(pepflow_manifest: Path, af2_manifest: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for source, path in (("pepflow", pepflow_manifest), ("af2", af2_manifest)):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                sequence = row["sequence"].strip().upper()
                previous = result.setdefault(sequence, source)
                if previous != source:
                    raise ValueError(f"Sequence occurs in multiple conformer sources: {sequence}")
    return result


def _write_structure(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_index",
                "sample_id",
                "sequence",
                "label",
                "structure_probability",
                "base_structure_probability",
                "conformer_available",
                "conformer_logit_delta",
            ]
        )
        for index, sample_id in enumerate(result["sample_ids"]):
            writer.writerow(
                [
                    index,
                    sample_id,
                    result["sequences"][index],
                    result["labels"][index],
                    f"{float(result['probabilities'][index]):.10g}",
                    f"{float(result['base_probabilities'][index]):.10g}",
                    int(bool(result["conformer_available"][index])),
                    f"{float(result['conformer_deltas'][index]):.10g}",
                ]
            )


def _read_fusion(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        [row["sample_id"] for row in rows],
        np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
        np.asarray([float(row["p_1d"]) for row in rows]),
        np.asarray([float(row["toxicity_probability"]) for row in rows]),
    )


def _normalize_1d_probability(
    source: Path,
    probability_key: str,
    destination: Path,
) -> Path:
    """Write the declared 1D probability column into the assembler contract."""

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{source}: no 1D probability rows")
    required = {"sample_id", "label", probability_key}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{source}: missing declared 1D columns {sorted(missing)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "sequence", "label", "toxicity_probability"])
        for row in rows:
            probability = float(row[probability_key])
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"{source}: invalid {probability_key} for {row['sample_id']}"
                )
            writer.writerow(
                [
                    row["sample_id"],
                    row.get("sequence", ""),
                    int(row["label"]),
                    f"{probability:.10g}",
                ]
            )
    return destination


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    points = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(points[0]),
        "q25": float(points[1]),
        "median": float(points[2]),
        "q75": float(points[3]),
        "max": float(points[4]),
    }


def _residual_diagnostics(
    labels: np.ndarray,
    sources: np.ndarray,
    available: np.ndarray,
    logit_delta: np.ndarray,
    structure_shift: np.ndarray,
    fusion_shift: np.ndarray,
) -> dict:
    """Summarize residual direction without using labels for model selection.

    These statistics are diagnostic only. They make a one-sided offset or a
    label-opposed correction visible instead of allowing a near-null aggregate
    metric change to hide the behavior of the residual head.
    """

    def summarize(mask: np.ndarray) -> dict:
        selected_delta = logit_delta[mask]
        selected_structure = structure_shift[mask]
        selected_fusion = fusion_shift[mask]
        selected_labels = labels[mask]
        negative = selected_delta < -1e-12
        positive = selected_delta > 1e-12
        zero = ~(negative | positive)
        label_directed = ((selected_labels == 1) & positive) | (
            (selected_labels == 0) & negative
        )
        return {
            "n": int(mask.sum()),
            "negative_logit_delta": int(negative.sum()),
            "zero_logit_delta": int(zero.sum()),
            "positive_logit_delta": int(positive.sum()),
            "label_directed_count": int(label_directed.sum()),
            "label_directed_fraction": (
                float(label_directed.mean()) if label_directed.size else None
            ),
            "logit_delta_quantiles": _quantiles(selected_delta),
            "structure_probability_shift_quantiles": _quantiles(selected_structure),
            "fusion_probability_shift_quantiles": _quantiles(selected_fusion),
            "mean_fusion_probability_shift": (
                float(selected_fusion.mean()) if selected_fusion.size else None
            ),
        }

    covered = available.astype(bool)
    return {
        "interpretation": (
            "Frozen post-hoc diagnostic only; labels were not used to train, "
            "select, calibrate, rescale, or threshold the residual."
        ),
        "covered": summarize(covered),
        "covered_positive": summarize(covered & (labels == 1)),
        "covered_negative": summarize(covered & (labels == 0)),
        "pepflow": summarize(covered & (sources == "pepflow")),
        "af2": summarize(covered & (sources == "af2")),
    }


def evaluate_dataset(
    name: str,
    processed: Path,
    calibrated_1d: Path,
    checkpoint: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    sequence_sources: dict[str, str],
    one_d_probability_key: str,
) -> dict:
    structure = predict_structure(
        processed, checkpoint, device=device, batch_size=batch_size
    )
    structure_csv = output_root / f"{name}_structure_predictions.csv"
    fusion_csv = output_root / f"{name}_predictions.csv"
    metrics_json = output_root / f"{name}_metrics.json"
    _write_structure(structure_csv, structure)
    normalized_1d = _normalize_1d_probability(
        calibrated_1d,
        one_d_probability_key,
        output_root / f"{name}_declared_1d_probabilities.csv",
    )
    result = assemble(
        DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D,
        DEFAULT_MULTIDIMENSIONAL_THRESHOLD,
        normalized_1d,
        structure_csv,
        fusion_csv,
        metrics_json,
    )
    sample_ids, labels, p_1d, probabilities = _read_fusion(fusion_csv)
    structure_index = {
        str(sample_id): index for index, sample_id in enumerate(structure["sample_ids"])
    }
    base_3d = np.asarray(
        [structure["base_probabilities"][structure_index[sample_id]] for sample_id in sample_ids]
    )
    base_fusion = (
        DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D * p_1d
        + (1.0 - DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D) * base_3d
    )
    base_metrics = _metrics(labels, base_fusion, DEFAULT_MULTIDIMENSIONAL_THRESHOLD)
    metric_delta = {
        key: float(result["metrics"][key] - base_metrics[key]) for key in base_metrics
    }
    aligned_sequences = [
        structure["sequences"][structure_index[sample_id]] for sample_id in sample_ids
    ]
    aligned_delta = np.asarray(
        [structure["conformer_deltas"][structure_index[sample_id]] for sample_id in sample_ids]
    )
    aligned_available = np.asarray(
        [structure["conformer_available"][structure_index[sample_id]] for sample_id in sample_ids],
        dtype=bool,
    )
    aligned_sources = np.asarray(
        [sequence_sources.get(sequence, "none") for sequence in aligned_sequences]
    )
    aligned_structure = np.asarray(
        [structure["probabilities"][structure_index[sample_id]] for sample_id in sample_ids]
    )
    source_ablation = {"none": base_metrics}
    for source in ("pepflow", "af2", "both"):
        if source == "both":
            mask = np.asarray([sequence in sequence_sources for sequence in aligned_sequences])
        else:
            mask = np.asarray(
                [sequence_sources.get(sequence) == source for sequence in aligned_sequences]
            )
        source_3d = 1.0 / (1.0 + np.exp(-(_logit(base_3d) + aligned_delta * mask)))
        source_fusion = (
            DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D * p_1d
            + (1.0 - DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D) * source_3d
        )
        source_ablation[source] = _metrics(
            labels, source_fusion, DEFAULT_MULTIDIMENSIONAL_THRESHOLD
        )
    available = aligned_available
    covered_metrics = (
        _metrics(labels[available], probabilities[available], DEFAULT_MULTIDIMENSIONAL_THRESHOLD)
        if available.any() and np.unique(labels[available]).size == 2
        else None
    )
    residual_diagnostics = _residual_diagnostics(
        labels=labels,
        sources=aligned_sources,
        available=available,
        logit_delta=aligned_delta,
        structure_shift=aligned_structure - base_3d,
        fusion_shift=probabilities - base_fusion,
    )
    return {
        "n": int(labels.size),
        "covered_records": int(available.sum()),
        "coverage_fraction": float(available.mean()),
        "metrics": result["metrics"],
        "same_checkpoint_zero_conformer_baseline_metrics": base_metrics,
        "metric_delta_from_zero_conformer_baseline": metric_delta,
        "prediction_flips_from_zero_conformer_baseline": int(
            np.sum(
                (probabilities >= DEFAULT_MULTIDIMENSIONAL_THRESHOLD)
                != (base_fusion >= DEFAULT_MULTIDIMENSIONAL_THRESHOLD)
            )
        ),
        "source_ablation": source_ablation,
        "residual_diagnostics": residual_diagnostics,
        "covered_metrics": covered_metrics,
        "mean_absolute_logit_delta_available": float(
            np.abs(np.asarray(structure["conformer_deltas"])[available]).mean()
            if available.any()
            else 0.0
        ),
        "structure_predictions": str(structure_csv),
        "fusion_predictions": str(fusion_csv),
        "one_d_probability_source": str(calibrated_1d),
        "one_d_probability_key": one_d_probability_key,
        "normalized_one_d_probabilities": str(normalized_1d),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test1-processed",
        default="data/processed/test1_chemical_sites_final_esm2_multiconformer.pt",
        type=Path,
    )
    parser.add_argument(
        "--test2-processed",
        default="data/processed/test2_chemical_sites_final_esm2_multiconformer.pt",
        type=Path,
    )
    parser.add_argument(
        "--test1-1d",
        default="runs/lora_mamba_full/test1_lora_mamba_calibrated.csv",
        type=Path,
    )
    parser.add_argument(
        "--test2-1d",
        default="runs/lora_mamba_full/test2_lora_mamba_calibrated.csv",
        type=Path,
    )
    parser.add_argument("--test1-1d-key", default="toxicity_probability")
    parser.add_argument("--test2-1d-key", default="toxicity_probability")
    parser.add_argument(
        "--checkpoint",
        default="runs/multisource_1_30_full2795_rank_stable_candidate/best_model.pt",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="runs/default_multidimensional_fusion_multiconformer_extension",
        type=Path,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--pepflow-manifest",
        default="data/test_conformers/pepflow/test1/cache/conformer_manifest.csv",
        type=Path,
    )
    parser.add_argument(
        "--af2-manifest",
        default="data/test_conformers/af2/test1/cache_16_30/conformer_manifest.csv",
        type=Path,
    )
    args = parser.parse_args()
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_sources = _sequence_sources(args.pepflow_manifest, args.af2_manifest)
    summary = {
        "protocol": {
            "status": "frozen_multiconformer_extension_evaluation",
            "test_labels_used_for_training_or_selection": False,
            "weight_1d": DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D,
            "weight_3d": 1.0 - DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D,
            "threshold": DEFAULT_MULTIDIMENSIONAL_THRESHOLD,
            "checkpoint": str(args.checkpoint),
        },
        "test1": evaluate_dataset(
            "test1",
            args.test1_processed,
            args.test1_1d,
            args.checkpoint,
            args.output_dir,
            device,
            args.batch_size,
            sequence_sources,
            args.test1_1d_key,
        ),
        "test2": evaluate_dataset(
            "test2",
            args.test2_processed,
            args.test2_1d,
            args.checkpoint,
            args.output_dir,
            device,
            args.batch_size,
            sequence_sources,
            args.test2_1d_key,
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
