"""Compare a multi-generator conformer residual by source and length stratum."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.conformer_cohort_analysis import _metric_deltas, _spearman
from ghxtox.utils import save_json


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_summary(
    row: dict[str, str],
    cache_root: Path,
) -> Path:
    indexed_id = str(row["sample_id"])
    fallback = cache_root / indexed_id / "ensemble_summary.npz"
    cache_text = str(row.get("cache_path") or "").strip()
    if cache_text:
        portable = Path(cache_text).parent / "ensemble_summary.npz"
        if portable.exists():
            return portable
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing ensemble summary for {indexed_id}: {fallback}")


def _manifest_entries(
    manifests: list[str | Path],
    cache_roots: list[str | Path],
) -> dict[str, dict[str, Any]]:
    if len(manifests) != len(cache_roots):
        raise ValueError("Each conformer manifest requires one cache root.")
    entries: dict[str, dict[str, Any]] = {}
    for manifest, root in zip(manifests, cache_roots):
        root_path = Path(root)
        for row in _read_csv(manifest):
            original_id = str(row.get("original_id") or row["sample_id"])
            entry = {
                "sample_id": original_id,
                "indexed_id": str(row["sample_id"]),
                "sequence": str(row["sequence"]),
                "label": int(row["label"]),
                "source": str(row.get("source") or "unknown"),
                "summary_path": _resolve_summary(row, root_path),
            }
            previous = entries.setdefault(original_id, entry)
            if previous != entry:
                raise ValueError(f"Inconsistent manifest rows for {original_id}.")
    return entries


def _bucket(length: int) -> str:
    if length <= 15:
        return "1-15"
    if length <= 20:
        return "16-20"
    if length <= 25:
        return "21-25"
    if length <= 30:
        return "26-30"
    return "31+"


def _summary_features(path: Path) -> dict[str, float]:
    with np.load(path) as payload:
        return {
            "mean_residue_rmsf": float(np.asarray(payload["residue_rmsf"]).mean()),
            "mean_conformer_rmsd": float(np.asarray(payload["conformer_rmsd"]).mean()),
            "mean_radius_gyration": float(np.asarray(payload["radius_gyration"]).mean()),
            "contact_uncertainty_8": float(
                np.mean(
                    np.asarray(payload["contact_occupancy_8"])
                    * (1.0 - np.asarray(payload["contact_occupancy_8"]))
                )
            ),
        }


def analyse_multisource(
    baseline_path: str | Path,
    candidate_path: str | Path,
    manifests: list[str | Path],
    cache_roots: list[str | Path],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = {row["sample_id"]: row for row in _read_csv(baseline_path)}
    candidate = {row["sample_id"]: row for row in _read_csv(candidate_path)}
    entries = _manifest_entries(manifests, cache_roots)
    if set(entries) - set(baseline) or set(entries) - set(candidate):
        raise ValueError("At least one conformer sample is absent from the prediction files.")

    rows: list[dict[str, Any]] = []
    for sample_id, entry in entries.items():
        first = baseline[sample_id]
        second = candidate[sample_id]
        label = int(entry["label"])
        first_probability = float(first["toxicity_probability"])
        second_probability = float(second["toxicity_probability"])
        length = len(str(entry["sequence"]))
        source = (
            "PepFlow"
            if "pepflow" in str(entry["source"]).lower()
            else "AF2/ColabFold"
            if "colabfold" in str(entry["source"]).lower()
            else str(entry["source"])
        )
        rows.append(
            {
                "sample_id": sample_id,
                "fold": int(first["fold"]),
                "label": label,
                "sequence": entry["sequence"],
                "length": length,
                "length_bucket": _bucket(length),
                "source": source,
                "baseline_probability": first_probability,
                "candidate_probability": second_probability,
                "probability_delta": second_probability - first_probability,
                "absolute_probability_delta": abs(second_probability - first_probability),
                "per_sample_brier_improvement": (first_probability - label) ** 2
                - (second_probability - label) ** 2,
                "baseline_correct": int((first_probability >= threshold) == bool(label)),
                "candidate_correct": int((second_probability >= threshold) == bool(label)),
                **_summary_features(Path(entry["summary_path"])),
            }
        )

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        labels = np.asarray([row["label"] for row in selected], dtype=np.int64)
        first = np.asarray([row["baseline_probability"] for row in selected])
        second = np.asarray([row["candidate_probability"] for row in selected])
        corrected = sum(
            row["baseline_correct"] == 0 and row["candidate_correct"] == 1 for row in selected
        )
        worsened = sum(
            row["baseline_correct"] == 1 and row["candidate_correct"] == 0 for row in selected
        )
        return {
            "samples": len(selected),
            "positive": int(labels.sum()),
            "negative": int(len(labels) - labels.sum()),
            "metrics": _metric_deltas(labels, first, second, threshold),
            "classification_transitions": {
                "baseline_wrong_candidate_correct": int(corrected),
                "baseline_correct_candidate_wrong": int(worsened),
                "net_corrections": int(corrected - worsened),
            },
        }

    report: dict[str, Any] = {
        "protocol": {
            "scope": "training-only group-aware OOF exploratory analysis",
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
            "fixed_threshold": float(threshold),
            "conformer_manifests": [str(value) for value in manifests],
            "selection_warning": "Strata are explanatory and must not be used to retune this candidate.",
        },
        "overall": summarize(rows),
        "by_source": {
            source: summarize([row for row in rows if row["source"] == source])
            for source in sorted({str(row["source"]) for row in rows})
        },
        "by_length": {
            bucket: summarize([row for row in rows if row["length_bucket"] == bucket])
            for bucket in ("1-15", "16-20", "21-25", "26-30")
            if any(row["length_bucket"] == bucket for row in rows)
        },
        "folds": {
            str(fold): summarize([row for row in rows if row["fold"] == fold])
            for fold in sorted({int(row["fold"]) for row in rows})
        },
    }
    structural_keys = (
        "length",
        "mean_residue_rmsf",
        "mean_conformer_rmsd",
        "mean_radius_gyration",
        "contact_uncertainty_8",
    )
    report["spearman_correlations"] = {
        key: {
            target: _spearman(
                np.asarray([row[key] for row in rows], dtype=np.float64),
                np.asarray([row[target] for row in rows], dtype=np.float64),
            )
            for target in ("absolute_probability_delta", "per_sample_brier_improvement")
        }
        for key in structural_keys
    }
    return report, rows


def _write_rows(rows: list[dict[str, Any]], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--conformer-manifest", nargs="+", required=True)
    parser.add_argument("--cache-root", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.677819)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-output")
    args = parser.parse_args()
    report, rows = analyse_multisource(
        args.baseline,
        args.candidate,
        args.conformer_manifest,
        args.cache_root,
        args.threshold,
    )
    save_json(report, args.output)
    if args.sample_output:
        _write_rows(rows, args.sample_output)


if __name__ == "__main__":
    main()
