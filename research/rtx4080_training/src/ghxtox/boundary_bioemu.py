"""Prepare and run the frozen Phase 33 BioEmu boundary cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.bioemu_pilot import (
    DEFAULT_SEED,
    MODEL_NAME,
    PACKAGE_VERSION,
    REQUESTED_CONFORMERS,
    assess_sample,
    build_sample_command,
    decide_feasibility,
)
from ghxtox.utils import save_json


FIXED_THRESHOLD = 0.677819
TARGET_SIZE = 600
MINIMUM_ROWS_PER_FOLD = 50
LENGTH_STRATA = ((11, 15), (16, 30), (31, 50))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_seed(sample_id: str, seed: int = DEFAULT_SEED) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return seed + int.from_bytes(digest[:4], "big") % 1_000_000


def _length_stratum(length: int) -> str:
    for lower, upper in LENGTH_STRATA:
        if lower <= length <= upper:
            return f"{lower}-{upper}"
    raise ValueError(f"Unsupported peptide length {length}; expected 11-50 aa.")


def _write_rows(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    fieldnames: list[str] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"Refusing to write an empty manifest: {output}")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames if fieldnames is not None else list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_fasta(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['sample_id']}\n{row['sequence']}\n")


def _required_sample_files_exist(sample_root: Path) -> bool:
    return all(
        (sample_root / name).is_file()
        for name in ("sequence.fasta", "topology.pdb", "samples.xtc")
    )


def _read_single_fasta(path: Path) -> str:
    return "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(">")
    )


def _load_previous_sequences(
    previous_manifest: str | Path | None,
) -> dict[str, str]:
    if previous_manifest is None:
        return {}
    return {
        str(row["sample_id"]): str(row["sequence"])
        for row in _read_csv(previous_manifest)
    }


def _verified_reusable_ids(
    previous_report: str | Path | None,
    previous_output_root: str | Path | None,
    previous_manifest: str | Path | None,
) -> set[str]:
    if (
        previous_report is None
        and previous_output_root is None
        and previous_manifest is None
    ):
        return set()
    if None in (previous_report, previous_output_root, previous_manifest):
        raise ValueError(
            "Previous report, output root, and manifest must be provided together."
        )
    report_path = Path(str(previous_report))
    root = Path(str(previous_output_root))
    sequences = _load_previous_sequences(previous_manifest)
    if not report_path.is_file() or not root.is_dir():
        raise FileNotFoundError("Previous BioEmu report or output root is missing.")
    old_protocol_path = root / "protocol.json"
    if not old_protocol_path.is_file():
        raise FileNotFoundError("Previous BioEmu protocol.json is missing.")
    old_protocol = json.loads(old_protocol_path.read_text(encoding="utf-8"))
    expected_protocol = {
        "package_version": PACKAGE_VERSION,
        "model_name": MODEL_NAME,
        "base_seed": DEFAULT_SEED,
        "num_samples_per_sequence": REQUESTED_CONFORMERS,
        "filter_samples": True,
    }
    mismatches = {
        key: (old_protocol.get(key), expected)
        for key, expected in expected_protocol.items()
        if old_protocol.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Previous BioEmu protocol is not reusable: {mismatches}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reusable = set()
    for row in report.get("samples", []):
        sample_id = str(row["sample_id"])
        sample_root = root / sample_id
        if not bool(row.get("usable")) or not _required_sample_files_exist(sample_root):
            continue
        expected = sequences.get(sample_id)
        if expected is None:
            raise ValueError(f"Previous manifest lacks {sample_id}.")
        if _read_single_fasta(sample_root / "sequence.fasta") != expected:
            raise ValueError(f"Previous BioEmu sequence mismatch for {sample_id}.")
        reusable.add(sample_id)
    return reusable


def prepare_boundary_cohort(
    handoff_csv: str | Path,
    output_dir: str | Path,
    *,
    previous_report: str | Path | None = None,
    previous_output_root: str | Path | None = None,
    previous_manifest: str | Path | None = None,
    target_size: int = TARGET_SIZE,
) -> dict[str, Any]:
    """Freeze a label-blind boundary cohort and write server-ready manifests."""

    if target_size != TARGET_SIZE:
        raise ValueError(f"Phase 33 target_size is frozen to {TARGET_SIZE}.")
    handoff_path = Path(handoff_csv)
    rows = _read_csv(handoff_path)
    required = {
        "source_index",
        "sample_id",
        "fold",
        "label",
        "sequence",
        "toxicity_probability",
        "logit",
        "prediction",
        "decision_threshold",
        "generator_source",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0] if rows else []))
        raise ValueError(f"Retained handoff is missing columns: {missing}")
    if len(rows) != 6387:
        raise ValueError(f"Expected 6387 retained rows, found {len(rows)}.")
    indices = [int(row["source_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("Retained handoff source_index values are not unique.")
    thresholds = {round(float(row["decision_threshold"]), 9) for row in rows}
    if thresholds != {round(FIXED_THRESHOLD, 9)}:
        raise ValueError(
            f"Expected only threshold {FIXED_THRESHOLD}, found {thresholds}."
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "frozen_before_phase33_generation",
        "artifact": "phase33_unified_bioemu_decision_boundary_cohort",
        "input": {
            "retained_handoff": str(handoff_path),
            "retained_handoff_sha256": _sha256(handoff_path),
            "rows": len(rows),
        },
        "selection": {
            "uses_training_labels": False,
            "uses_external_test_labels": False,
            "target_size": TARGET_SIZE,
            "fixed_threshold": FIXED_THRESHOLD,
            "rank": "abs(toxicity_probability - fixed_threshold)",
            "tie_breaker": "source_index ascending",
            "minimum_rows_per_outer_fold": MINIMUM_ROWS_PER_FOLD,
            "post_selection_rule_search_allowed": False,
        },
        "generation": {
            "bioemu_package": PACKAGE_VERSION,
            "bioemu_model": MODEL_NAME,
            "base_seed": DEFAULT_SEED,
            "requested_conformers_per_sequence": REQUESTED_CONFORMERS,
            "physical_filter": "filter_samples=True",
            "msa_protocol": "BioEmu bundled ColabFold/MMseqs2 default remote MSA",
            "resume_existing_completed_outputs": True,
            "full_6387_generation_authorized": False,
        },
        "evaluation_boundary": {
            "phase25_exact_fallback_outside_selected_cohort": True,
            "fixed_threshold": FIXED_THRESHOLD,
            "test1_test2_labels_sealed": True,
            "seed_threshold_subset_checkpoint_search_after_oof": False,
        },
    }
    if previous_report is not None:
        protocol["reuse_source"] = {
            "report": str(previous_report),
            "report_sha256": _sha256(str(previous_report)),
            "manifest": str(previous_manifest),
            "manifest_sha256": _sha256(str(previous_manifest)),
            "requires_same_package_model_seed_and_32_samples": True,
        }
    save_json(protocol, output_root / "protocol.json")

    # Membership is frozen using only this label-free projection.
    candidates = []
    label_lookup = {}
    source_lookup = {int(row["source_index"]): row for row in rows}
    for row in rows:
        probability = float(row["toxicity_probability"])
        sequence = str(row["sequence"])
        source_index = int(row["source_index"])
        sample_id = str(row["sample_id"])
        candidate = {
            "source_index": source_index,
            "sample_id": sample_id,
            "fold": int(row["fold"]),
            "sequence": sequence,
            "length": len(sequence),
            "length_bin": _length_stratum(len(sequence)),
            "phase25_probability": probability,
            "phase25_logit": float(row["logit"]),
            "decision_threshold": FIXED_THRESHOLD,
            "absolute_probability_margin": abs(probability - FIXED_THRESHOLD),
            "requested_conformers": REQUESTED_CONFORMERS,
            "base_seed": DEFAULT_SEED,
            "sample_seed": _sample_seed(sample_id),
            "bioemu_package": PACKAGE_VERSION,
            "bioemu_model": MODEL_NAME,
            "physical_filter": "filter_samples=True",
        }
        candidates.append(candidate)
        label_lookup[source_index] = {
            "label": int(row["label"]),
            "prediction": int(float(row["prediction"])),
        }
    candidates.sort(
        key=lambda row: (
            float(row["absolute_probability_margin"]),
            int(row["source_index"]),
        )
    )
    selected = [
        dict(row, selection_rank=rank)
        for rank, row in enumerate(candidates[:TARGET_SIZE], 1)
    ]
    fold_counts = Counter(int(row["fold"]) for row in selected)
    if (
        set(fold_counts) != {0, 1, 2, 3, 4}
        or min(fold_counts.values()) < MINIMUM_ROWS_PER_FOLD
    ):
        raise ValueError(
            f"Frozen boundary cohort has insufficient fold coverage: {fold_counts}"
        )

    reusable_ids = _verified_reusable_ids(
        previous_report,
        previous_output_root,
        previous_manifest,
    )
    selected_rows = [
        {**row, "reuse_existing": int(str(row["sample_id"]) in reusable_ids)}
        for row in selected
    ]
    generate_rows = [row for row in selected_rows if not row["reuse_existing"]]
    reuse_rows = [row for row in selected_rows if row["reuse_existing"]]
    _write_rows(selected_rows, output_root / "selected_600.csv")
    _write_fasta(selected_rows, output_root / "selected_600.fasta")
    _write_rows(generate_rows, output_root / "generate_missing.csv")
    _write_fasta(generate_rows, output_root / "generate_missing.fasta")
    _write_rows(
        reuse_rows,
        output_root / "reuse_existing.csv",
        fieldnames=list(selected_rows[0]),
    )

    previous = (
        json.loads(Path(str(previous_report)).read_text(encoding="utf-8"))
        if previous_report is not None
        else {"samples": []}
    )
    successful_previous = [
        row
        for row in previous.get("samples", [])
        if row.get("returncode") == 0
        and float(row.get("elapsed_seconds", 0.0)) > 0
        and int(row.get("storage_bytes", 0)) > 0
    ]
    median_seconds = (
        statistics.median(float(row["elapsed_seconds"]) for row in successful_previous)
        if successful_previous
        else None
    )
    mean_bytes = (
        statistics.mean(int(row["storage_bytes"]) for row in successful_previous)
        if successful_previous
        else None
    )
    diagnostics = [label_lookup[int(row["source_index"])] for row in selected_rows]
    summary = {
        "status": "ready_for_autodl_generation",
        "selection_was_label_blind": True,
        "external_test_labels_read": False,
        "selected": len(selected_rows),
        "reuse_existing": len(reuse_rows),
        "new_generation_required": len(generate_rows),
        "by_fold": {str(key): value for key, value in sorted(fold_counts.items())},
        "by_length": dict(Counter(row["length_bin"] for row in selected_rows)),
        "by_existing_generator": dict(
            Counter(
                source_lookup[int(row["source_index"])]["generator_source"]
                for row in selected_rows
            )
        ),
        "post_freeze_label_diagnostics": {
            "positive": sum(row["label"] for row in diagnostics),
            "negative": len(diagnostics) - sum(row["label"] for row in diagnostics),
            "phase25_errors": sum(
                row["label"] != row["prediction"] for row in diagnostics
            ),
            "note": "Labels were joined only after deterministic membership was frozen.",
        },
        "cost_projection_from_verified_pilot": {
            "median_seconds_per_sequence": median_seconds,
            "mean_storage_bytes_per_sequence": mean_bytes,
            "projected_new_gpu_hours": (
                median_seconds * len(generate_rows) / 3600
                if median_seconds is not None
                else None
            ),
            "projected_new_storage_gb": (
                mean_bytes * len(generate_rows) / 1e9
                if mean_bytes is not None
                else None
            ),
        },
        "files": {
            "protocol": "protocol.json",
            "selected_manifest": "selected_600.csv",
            "generation_manifest": "generate_missing.csv",
            "generation_fasta": "generate_missing.fasta",
            "reuse_manifest": "reuse_existing.csv",
        },
    }
    save_json(summary, output_root / "selection_summary.json")
    checksum_paths = [
        output_root / "protocol.json",
        output_root / "selected_600.csv",
        output_root / "selected_600.fasta",
        output_root / "generate_missing.csv",
        output_root / "generate_missing.fasta",
        output_root / "reuse_existing.csv",
        output_root / "selection_summary.json",
    ]
    (output_root / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return summary


def run_boundary_manifest(
    manifest: str | Path,
    output_root: str | Path,
    *,
    authorized: bool = False,
    dry_run: bool = False,
    continue_on_error: bool = False,
    python: str = sys.executable,
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = _read_csv(manifest)
    if not rows or len(rows) > TARGET_SIZE:
        raise ValueError(f"Boundary manifest must contain 1-{TARGET_SIZE} rows.")
    if "label" in rows[0]:
        raise ValueError("Server-facing Phase 33 manifest must not contain labels.")
    if not dry_run and not authorized:
        raise PermissionError("Formal Phase 33 generation requires --authorized.")
    if (ckpt_path is None) != (model_config_path is None):
        raise ValueError("ckpt_path and model_config_path must be provided together.")
    if not dry_run:
        installed = importlib.metadata.version("bioemu")
        if installed != PACKAGE_VERSION:
            raise RuntimeError(
                f"Expected bioemu=={PACKAGE_VERSION}, found {installed}."
            )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_protocol = {
        "status": "running" if not dry_run else "dry_run",
        "scope": "Phase 33 missing decision-boundary rows only",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "rows": len(rows),
        "bioemu_package": PACKAGE_VERSION,
        "bioemu_model": MODEL_NAME,
        "base_seed": DEFAULT_SEED,
        "num_samples_per_sequence": REQUESTED_CONFORMERS,
        "filter_samples": True,
        "labels_present": False,
        "resume_completed_outputs": True,
        "continue_on_error": continue_on_error,
        "full_6387_generation_authorized": False,
        "dry_run": dry_run,
    }
    save_json(run_protocol, root / "protocol.json")

    statuses = []
    for row in rows:
        sample_id = str(row["sample_id"])
        sample_root = root / sample_id
        sample_root.mkdir(parents=True, exist_ok=True)
        status_path = sample_root / "run_status.json"
        if status_path.is_file() and _required_sample_files_exist(sample_root):
            previous_status = json.loads(status_path.read_text(encoding="utf-8"))
            if previous_status.get("returncode") == 0:
                statuses.append({**previous_status, "resume_skipped": True})
                continue
        command = build_sample_command(
            row,
            sample_root,
            python=python,
            seed=DEFAULT_SEED,
            num_samples=REQUESTED_CONFORMERS,
            cache_embeds_dir=root / "shared_cache" / "embeds",
            cache_so3_dir=root / "shared_cache" / "so3",
            ckpt_path=ckpt_path,
            model_config_path=model_config_path,
        )
        status = {
            "sample_id": sample_id,
            "source_index": int(row["source_index"]),
            "fold": int(row["fold"]),
            "length_bin": str(row["length_bin"]),
            "length": int(row["length"]),
            "seed": _sample_seed(sample_id),
            "command": command,
            "returncode": None,
            "elapsed_seconds": 0.0,
            "dry_run": dry_run,
            "resume_skipped": False,
        }
        started = time.perf_counter()
        if not dry_run:
            with (sample_root / "sample.log").open("a", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            status["returncode"] = int(completed.returncode)
        status["elapsed_seconds"] = float(time.perf_counter() - started)
        save_json(status, status_path)
        statuses.append(status)
        if status["returncode"] not in (None, 0) and not continue_on_error:
            break
    result = {"protocol": run_protocol, "samples": statuses}
    save_json(result, root / "run_summary.json")
    return result


def assess_boundary_manifest(
    manifest: str | Path,
    output_root: str | Path,
    output_json: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    rows = _read_csv(manifest)
    root = Path(output_root)
    assessed = [assess_sample(row, root / str(row["sample_id"])) for row in rows]
    report = decide_feasibility(
        assessed,
        total_training_samples=len(rows),
        max_projected_gpu_hours=16.0,
        max_projected_storage_gb=2.0,
    )
    passed = bool(report["gate"]["all_conditions_passed"])
    report.update(
        {
            "decision": (
                "ready_to_import_phase33_boundary_structures"
                if passed
                else "resume_or_repair_phase33_boundary_generation"
            ),
            "boundary_specialist_training_authorized": passed,
            "full_generation_authorized": False,
            "external_test_labels_read": False,
        }
    )
    save_json(report, output_json)
    _write_rows(report["samples"], output_csv)
    return report


def _label_server_rows(
    server_rows: list[dict[str, str]],
    handoff_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    handoff_by_index = {int(row["source_index"]): row for row in handoff_rows}
    labeled = []
    for row in server_rows:
        source_index = int(row["source_index"])
        handoff = handoff_by_index.get(source_index)
        if handoff is None:
            raise ValueError(f"Missing Phase 25 handoff row {source_index}.")
        if (
            str(handoff["sample_id"]) != str(row["sample_id"])
            or str(handoff["sequence"]) != str(row["sequence"])
            or int(handoff["fold"]) != int(row["fold"])
        ):
            raise ValueError(f"Boundary manifest alignment mismatch at {source_index}.")
        labeled.append(
            {
                **row,
                "label": int(handoff["label"]),
                "toxicity_probability": float(handoff["toxicity_probability"]),
                "decision_threshold": FIXED_THRESHOLD,
            }
        )
    return labeled


def import_boundary_structures(
    boundary_dir: str | Path,
    server_root: str | Path,
    handoff_csv: str | Path,
    output_cache_root: str | Path,
    *,
    pilot_cache_root: str | Path,
    pilot_raw_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create a local 600-row BioEmu cache after the label-free server run."""

    from ghxtox.bioemu_conformer_cache import build_bioemu_conformer_cache

    boundary = Path(boundary_dir)
    server = Path(server_root)
    output = Path(output_cache_root)
    output.mkdir(parents=True, exist_ok=True)
    report_path = server / "generation_feasibility.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("decision") != "ready_to_import_phase33_boundary_structures":
        raise ValueError(
            "Phase 33 server outputs are not authorized for import: "
            f"{report.get('decision')}"
        )
    if not report.get("boundary_specialist_training_authorized"):
        raise ValueError("Phase 33 boundary specialist training is not authorized.")

    handoff_rows = _read_csv(handoff_csv)
    selected = _label_server_rows(
        _read_csv(boundary / "selected_600.csv"), handoff_rows
    )
    missing = _label_server_rows(
        _read_csv(boundary / "generate_missing.csv"), handoff_rows
    )
    reused = _label_server_rows(
        _read_csv(boundary / "reuse_existing.csv"), handoff_rows
    )
    if len(selected) != TARGET_SIZE:
        raise ValueError(
            f"Expected {TARGET_SIZE} selected rows, found {len(selected)}."
        )
    if len(missing) + len(reused) != TARGET_SIZE:
        raise ValueError("Generated plus reused manifests do not cover 600 rows.")

    selected_path = boundary / "selected_600_with_labels.csv"
    missing_path = boundary / "generate_missing_with_labels.csv"
    reused_path = boundary / "reuse_existing_with_labels.csv"
    _write_rows(selected, selected_path)
    _write_rows(missing, missing_path)
    _write_rows(reused, reused_path)

    generated_summary = build_bioemu_conformer_cache(
        missing_path,
        server,
        output,
        feasibility_json=report_path,
        source="BioEmu-v1.2",
    )
    generated_manifest_rows = _read_csv(output / "conformer_manifest.csv")
    for row in generated_manifest_rows:
        row["reused_from_pilot"] = 0

    pilot_cache = Path(pilot_cache_root)
    pilot_raw = Path(pilot_raw_root) if pilot_raw_root is not None else None
    reused_manifest_rows: list[dict[str, Any]] = []
    for row in reused:
        sample_id = str(row["sample_id"])
        source_summary = pilot_cache / sample_id / "ensemble_summary.npz"
        if not source_summary.is_file():
            raise FileNotFoundError(f"Missing reused BioEmu cache for {sample_id}.")
        sample_output = output / sample_id
        sample_output.mkdir(parents=True, exist_ok=True)
        target_summary = sample_output / "ensemble_summary.npz"
        shutil.copy2(source_summary, target_summary)
        with np.load(target_summary) as payload:
            conformer_count = int(payload["aligned_coords"].shape[0])
        source_pdb = ""
        source_xtc = ""
        if pilot_raw is not None:
            pdb = pilot_raw / sample_id / "topology.pdb"
            xtc = pilot_raw / sample_id / "samples.xtc"
            source_pdb = str(pdb) if pdb.is_file() else ""
            source_xtc = str(xtc) if xtc.is_file() else ""
        reused_manifest_rows.append(
            {
                "source_index": int(row["source_index"]),
                "sample_id": sample_id,
                "original_id": sample_id,
                "sequence": str(row["sequence"]),
                "label": int(row["label"]),
                "fold": int(row["fold"]),
                "length": int(row["length"]),
                "length_stratum": str(row["length_bin"]),
                "conformer_count": conformer_count,
                "source": "BioEmu-v1.2",
                "confidence_imputed": 1,
                "cache_path": str(target_summary),
                "source_pdb": source_pdb,
                "source_xtc": source_xtc,
                "reused_from_pilot": 1,
            }
        )

    combined_rows = sorted(
        [*generated_manifest_rows, *reused_manifest_rows],
        key=lambda row: int(row["source_index"]),
    )
    if len(combined_rows) != TARGET_SIZE:
        raise ValueError(
            f"Expected 600 combined BioEmu rows, found {len(combined_rows)}."
        )
    if len({int(row["source_index"]) for row in combined_rows}) != TARGET_SIZE:
        raise ValueError("Combined BioEmu manifest contains duplicate source indices.")
    selected_indices = {int(row["source_index"]) for row in selected}
    if {int(row["source_index"]) for row in combined_rows} != selected_indices:
        raise ValueError("Combined BioEmu manifest does not match selected cohort.")

    fieldnames = list(generated_manifest_rows[0])
    if "reused_from_pilot" not in fieldnames:
        fieldnames.append("reused_from_pilot")
    _write_rows(combined_rows, output / "conformer_manifest.csv", fieldnames=fieldnames)
    summary = {
        "status": "ready_for_phase33_boundary_specialist",
        "selected": len(selected),
        "generated": len(generated_manifest_rows),
        "reused": len(reused_manifest_rows),
        "cache_root": str(output),
        "conformer_manifest": str(output / "conformer_manifest.csv"),
        "server_report": str(report_path),
        "boundary_specialist_training_authorized": True,
        "external_test_labels_read": False,
        "total_filtered_conformers": int(
            sum(int(row["conformer_count"]) for row in combined_rows)
        ),
        "generated_total_filtered_conformers": int(
            generated_summary["total_filtered_conformers"]
        ),
        "length_strata": dict(Counter(row["length_stratum"] for row in combined_rows)),
        "folds": dict(Counter(str(row["fold"]) for row in combined_rows)),
    }
    save_json(summary, output / "import_summary.json")
    checksum_paths = [
        selected_path,
        missing_path,
        reused_path,
        output / "conformer_manifest.csv",
        output / "import_summary.json",
    ]
    (output / "import_checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {Path(path).name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--handoff-csv", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--previous-report")
    prepare.add_argument("--previous-output-root")
    prepare.add_argument("--previous-manifest")

    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--authorized", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--continue-on-error", action="store_true")
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--ckpt-path")
    run.add_argument("--model-config-path")

    assess = subparsers.add_parser("assess")
    assess.add_argument("--manifest", required=True)
    assess.add_argument("--output-root", required=True)
    assess.add_argument("--output-json", required=True)
    assess.add_argument("--output-csv", required=True)

    import_local = subparsers.add_parser("import-local")
    import_local.add_argument("--boundary-dir", required=True)
    import_local.add_argument("--server-root", required=True)
    import_local.add_argument("--handoff-csv", required=True)
    import_local.add_argument("--output-cache-root", required=True)
    import_local.add_argument("--pilot-cache-root", required=True)
    import_local.add_argument("--pilot-raw-root")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        summary = prepare_boundary_cohort(
            args.handoff_csv,
            args.output_dir,
            previous_report=args.previous_report,
            previous_output_root=args.previous_output_root,
            previous_manifest=args.previous_manifest,
        )
        print(json.dumps(summary, ensure_ascii=False))
    elif args.command == "run":
        report = run_boundary_manifest(
            args.manifest,
            args.output_root,
            authorized=args.authorized,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
            python=args.python,
            ckpt_path=args.ckpt_path,
            model_config_path=args.model_config_path,
        )
        print(json.dumps({"samples": len(report["samples"])}, ensure_ascii=False))
    elif args.command == "assess":
        report = assess_boundary_manifest(
            args.manifest,
            args.output_root,
            args.output_json,
            args.output_csv,
        )
        print(json.dumps({"decision": report["decision"]}, ensure_ascii=False))
    else:
        summary = import_boundary_structures(
            args.boundary_dir,
            args.server_root,
            args.handoff_csv,
            args.output_cache_root,
            pilot_cache_root=args.pilot_cache_root,
            pilot_raw_root=args.pilot_raw_root,
        )
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
