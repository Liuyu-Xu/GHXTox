"""Prepare, run, and assess the train-only BioEmu unified-generator pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.peptide_topology_audit import _risk_class
from ghxtox.utils import save_json


LENGTH_BINS = ((11, 15), (16, 30), (31, 50))
PACKAGE_VERSION = "1.4.1"
MODEL_NAME = "bioemu-v1.2"
DEFAULT_SEED = 20260731
REQUESTED_CONFORMERS = 32
SMOKE_SIZE = 6
PILOT_SIZE = 200


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _length_bin(length: int) -> str | None:
    for lower, upper in LENGTH_BINS:
        if lower <= length <= upper:
            return f"{lower}-{upper}"
    return None


def _allocate_quotas(
    cells: list[tuple[str, int, int]],
    target_size: int,
    seed: int,
) -> dict[tuple[str, int, int], int]:
    """Allocate 6/7-cell quotas while exactly balancing all attainable margins."""

    base, remainder = divmod(target_size, len(cells))
    length_values = list(dict.fromkeys(cell[0] for cell in cells))
    fold_values = list(dict.fromkeys(cell[1] for cell in cells))
    label_values = list(dict.fromkeys(cell[2] for cell in cells))

    def balanced(total: int, count: int) -> list[int]:
        quotient, extra = divmod(total, count)
        return [quotient + int(index < extra) for index in range(count)]

    length_targets = dict(zip(length_values, balanced(target_size, len(length_values))))
    fold_targets = dict(zip(fold_values, balanced(target_size, len(fold_values))))
    label_targets = dict(zip(label_values, balanced(target_size, len(label_values))))
    length_extra = {
        value: length_targets[value] - base * sum(cell[0] == value for cell in cells)
        for value in length_values
    }
    fold_extra = {
        value: fold_targets[value] - base * sum(cell[1] == value for cell in cells)
        for value in fold_values
    }
    label_extra = {
        value: label_targets[value] - base * sum(cell[2] == value for cell in cells)
        for value in label_values
    }
    if min((*length_extra.values(), *fold_extra.values(), *label_extra.values())) < 0:
        raise ValueError("Target size is too small for exact marginal balancing.")

    by_fold = {
        fold: [cell for cell in cells if cell[1] == fold] for fold in fold_values
    }
    chosen: set[tuple[str, int, int]] = set()

    def assign(fold_index: int) -> bool:
        if fold_index == len(fold_values):
            return (
                sum(length_extra.values()) == 0
                and sum(label_extra.values()) == 0
                and len(chosen) == remainder
            )
        fold = fold_values[fold_index]
        needed = fold_extra[fold]
        options = list(combinations(by_fold[fold], needed))
        options.sort(
            key=lambda option: _stable_rank(
                "|".join(",".join(map(str, cell)) for cell in option),
                seed + fold,
            )
        )
        for option in options:
            option_lengths = Counter(cell[0] for cell in option)
            option_labels = Counter(cell[2] for cell in option)
            if any(
                length_extra[value] < count for value, count in option_lengths.items()
            ) or any(
                label_extra[value] < count for value, count in option_labels.items()
            ):
                continue
            for cell in option:
                chosen.add(cell)
                length_extra[cell[0]] -= 1
                label_extra[cell[2]] -= 1
            if assign(fold_index + 1):
                return True
            for cell in option:
                chosen.remove(cell)
                length_extra[cell[0]] += 1
                label_extra[cell[2]] += 1
        return False

    if not assign(0):
        raise ValueError(
            "Could not construct an exactly balanced near-uniform pilot allocation."
        )
    return {cell: base + int(cell in chosen) for cell in cells}


def select_bioemu_pilot(
    fold_manifest: str | Path,
    output_manifest: str | Path,
    output_fasta: str | Path,
    smoke_manifest: str | Path,
    smoke_fasta: str | Path,
    summary_json: str | Path | None = None,
    target_size: int = PILOT_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select an exact, deterministic near-balanced label/length/fold pilot."""

    source_rows = _read_csv(fold_manifest)
    candidates: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in source_rows:
        sequence = str(row["sequence"])
        length_bin = _length_bin(len(sequence))
        if length_bin is None:
            continue
        label = int(row["label"])
        fold = int(row["fold"])
        cysteines = sequence.count("C")
        item = {
            "source_index": int(row["source_index"]),
            "sample_id": str(row["sample_id"]),
            "label": label,
            "fold": fold,
            "group_id": str(row["group_id"]),
            "sequence": sequence,
            "length": len(sequence),
            "length_bin": length_bin,
            "cysteines": cysteines,
            "topology_risk": _risk_class(cysteines, len(sequence), False, False),
            "selection_seed": seed,
            "requested_conformers": REQUESTED_CONFORMERS,
            "bioemu_package": PACKAGE_VERSION,
            "bioemu_model": MODEL_NAME,
            "msa_protocol": "BioEmu bundled ColabFold/MMseqs2 default remote MSA",
            "physical_filter": "BioEmu filter_samples=True",
        }
        candidates.setdefault((length_bin, fold, label), []).append(item)

    folds = sorted({int(row["fold"]) for row in source_rows})
    cells = [
        (f"{low}-{high}", fold, label)
        for low, high in LENGTH_BINS
        for fold in folds
        for label in (0, 1)
    ]
    quotas = _allocate_quotas(cells, target_size, seed)
    selected: list[dict[str, Any]] = []
    for cell in cells:
        values = candidates.get(cell, [])
        values.sort(key=lambda row: _stable_rank(str(row["sample_id"]), seed))
        quota = quotas[cell]
        if len(values) < quota:
            raise ValueError(
                f"Pilot cell {cell} has {len(values)} candidates but needs {quota}."
            )
        for rank, row in enumerate(values[:quota], start=1):
            selected.append({**row, "cell_quota": quota, "cell_rank": rank})
    selected.sort(key=lambda row: int(row["source_index"]))
    if len(selected) != target_size:
        raise AssertionError(
            f"Expected {target_size} selected rows, got {len(selected)}."
        )

    smoke: list[dict[str, Any]] = []
    for low, high in LENGTH_BINS:
        bin_name = f"{low}-{high}"
        for label in (0, 1):
            values = [
                row
                for row in selected
                if row["length_bin"] == bin_name and row["label"] == label
            ]
            values.sort(key=lambda row: _stable_rank(str(row["sample_id"]), seed + 1))
            smoke.append({**values[0], "smoke": 1})
    smoke.sort(key=lambda row: (str(row["length_bin"]), int(row["label"])))

    _write_rows(selected, output_manifest)
    _write_fasta(selected, output_fasta)
    _write_rows(smoke, smoke_manifest)
    _write_fasta(smoke, smoke_fasta)
    if summary_json is not None:
        save_json(
            {
                "protocol": {
                    "scope": "train-only BioEmu unified-generator feasibility pilot",
                    "fold_manifest": str(fold_manifest),
                    "target_size": target_size,
                    "seed": seed,
                    "length_bins": [f"{lower}-{upper}" for lower, upper in LENGTH_BINS],
                    "stratification": "exact label/fold balance and nearest-possible length balance",
                    "external_test_labels_read": False,
                    "requested_conformers_per_sequence": REQUESTED_CONFORMERS,
                    "bioemu_package": PACKAGE_VERSION,
                    "bioemu_model": MODEL_NAME,
                    "physical_filter": "filter_samples=True",
                    "full_generation_authorized": False,
                },
                "pilot": {
                    "samples": len(selected),
                    "by_length": dict(Counter(row["length_bin"] for row in selected)),
                    "by_label": {
                        str(key): value
                        for key, value in Counter(
                            row["label"] for row in selected
                        ).items()
                    },
                    "by_fold": {
                        str(key): value
                        for key, value in Counter(
                            row["fold"] for row in selected
                        ).items()
                    },
                    "cell_quota_range": [
                        min(row["cell_quota"] for row in selected),
                        max(row["cell_quota"] for row in selected),
                    ],
                },
                "smoke": {
                    "samples": len(smoke),
                    "sample_ids": [row["sample_id"] for row in smoke],
                    "by_length": dict(Counter(row["length_bin"] for row in smoke)),
                    "by_label": {
                        str(key): value
                        for key, value in Counter(row["label"] for row in smoke).items()
                    },
                },
            },
            summary_json,
        )
    return selected, smoke


def _write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_fasta(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['sample_id']}\n{row['sequence']}\n")


def _sample_seed(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return seed + int.from_bytes(digest[:4], "big") % 1_000_000


def build_sample_command(
    row: dict[str, Any],
    output_dir: str | Path,
    python: str = sys.executable,
    seed: int = DEFAULT_SEED,
    num_samples: int = REQUESTED_CONFORMERS,
    cache_embeds_dir: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
) -> list[str]:
    if (ckpt_path is None) != (model_config_path is None):
        raise ValueError("ckpt_path and model_config_path must be provided together.")
    command = [
        python,
        "-m",
        "bioemu.sample",
        f"--sequence={row['sequence']}",
        f"--num_samples={num_samples}",
        f"--output_dir={output_dir}",
        f"--base_seed={_sample_seed(str(row['sample_id']), seed)}",
        "--filter_samples=True",
    ]
    if ckpt_path is not None:
        command.extend(
            [
                f"--ckpt_path={ckpt_path}",
                f"--model_config_path={model_config_path}",
            ]
        )
    else:
        command.append(f"--model_name={MODEL_NAME}")
    if cache_embeds_dir:
        command.append(f"--cache_embeds_dir={cache_embeds_dir}")
    if cache_so3_dir:
        command.append(f"--cache_so3_dir={cache_so3_dir}")
    return command


def run_manifest(
    manifest: str | Path,
    output_root: str | Path,
    *,
    limit: int,
    allow_pilot: bool = False,
    dry_run: bool = False,
    python: str = sys.executable,
    seed: int = DEFAULT_SEED,
    num_samples: int = REQUESTED_CONFORMERS,
    cache_embeds_dir: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows = _read_csv(manifest)
    if len(rows) > PILOT_SIZE:
        raise ValueError(
            "BioEmu runner refuses manifests larger than the frozen 200-row pilot."
        )
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if not allow_pilot and limit > SMOKE_SIZE:
        raise ValueError(
            "Only six smoke rows are authorized unless allow_pilot is explicit."
        )
    selected = rows[: min(limit, len(rows))]
    if not dry_run:
        installed = importlib.metadata.version("bioemu")
        if installed != PACKAGE_VERSION:
            raise RuntimeError(
                f"Expected bioemu=={PACKAGE_VERSION}, found {installed}."
            )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "manifest": str(manifest),
        "rows_requested": len(selected),
        "scope": "200-sample train-only BioEmu pilot; no full-dataset mode exists",
        "package_version": PACKAGE_VERSION,
        "model_name": MODEL_NAME,
        "base_seed": seed,
        "num_samples_per_sequence": num_samples,
        "filter_samples": True,
        "msa_protocol": "bundled ColabFold/MMseqs2 default remote MSA",
        "checkpoint_source": "explicit_local"
        if ckpt_path is not None
        else "named_model",
        "ckpt_path": str(ckpt_path) if ckpt_path is not None else None,
        "model_config_path": (
            str(model_config_path) if model_config_path is not None else None
        ),
        "dry_run": dry_run,
    }
    save_json(protocol, root / "protocol.json")

    statuses: list[dict[str, Any]] = []
    for row in selected:
        sample_id = str(row["sample_id"])
        sample_root = root / sample_id
        sample_root.mkdir(parents=True, exist_ok=True)
        command = build_sample_command(
            row,
            sample_root,
            python=python,
            seed=seed,
            num_samples=num_samples,
            cache_embeds_dir=cache_embeds_dir,
            cache_so3_dir=cache_so3_dir,
            ckpt_path=ckpt_path,
            model_config_path=model_config_path,
        )
        status = {
            "sample_id": sample_id,
            "source_index": int(row["source_index"]),
            "length_bin": str(row["length_bin"]),
            "length": int(row["length"]),
            "label": int(row["label"]),
            "seed": _sample_seed(sample_id, seed),
            "command": command,
            "returncode": None,
            "elapsed_seconds": 0.0,
            "dry_run": dry_run,
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
        save_json(status, sample_root / "run_status.json")
        statuses.append(status)
        if status["returncode"] not in (None, 0):
            break
    save_json({"protocol": protocol, "samples": statuses}, root / "run_summary.json")
    return statuses


def _read_single_fasta(path: Path) -> str:
    sequence = "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith(">")
    )
    return sequence


def _pdb_residue_count(path: Path) -> int:
    residues = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            residues.add((line[21:22], line[22:27]))
    return len(residues)


def _directory_bytes(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def assess_sample(row: dict[str, str], sample_root: Path) -> dict[str, Any]:
    expected_sequence = str(row["sequence"])
    status_path = sample_root / "run_status.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {}
    )
    result: dict[str, Any] = {
        "sample_id": str(row["sample_id"]),
        "length_bin": str(row["length_bin"]),
        "length": len(expected_sequence),
        "label": int(row["label"]) if row.get("label") not in (None, "") else None,
        "elapsed_seconds": float(status.get("elapsed_seconds", 0.0)),
        "storage_bytes": _directory_bytes(sample_root) if sample_root.exists() else 0,
        "returncode": status.get("returncode"),
        "raw_conformers": 0,
        "valid_conformers": 0,
        "filtered_conformers": 0,
        "sequence_match": False,
        "residue_count_match": False,
        "median_ca_rmsd_nm": float("nan"),
        "median_radius_gyration_nm": float("nan"),
        "max_adjacent_ca_distance_nm": float("nan"),
        "error": "",
    }
    try:
        sequence_path = sample_root / "sequence.fasta"
        topology_path = sample_root / "topology.pdb"
        trajectory_path = sample_root / "samples.xtc"
        result["sequence_match"] = (
            sequence_path.exists()
            and _read_single_fasta(sequence_path) == expected_sequence
        )
        result["residue_count_match"] = topology_path.exists() and _pdb_residue_count(
            topology_path
        ) == len(expected_sequence)
        raw_positions = []
        for path in sorted(sample_root.glob("batch_*.npz")):
            with np.load(path) as payload:
                if str(payload["sequence"].item()) != expected_sequence:
                    raise ValueError(f"Raw batch sequence mismatch in {path.name}.")
                raw_positions.append(np.asarray(payload["pos"], dtype=np.float32))
        result["raw_conformers"] = int(sum(len(values) for values in raw_positions))
        if topology_path.exists() and trajectory_path.exists():
            import mdtraj as md  # BioEmu runtime dependency; intentionally lazy.

            trajectory = md.load_xtc(str(trajectory_path), top=str(topology_path))
            ca_indices = trajectory.topology.select("name CA")
            if len(ca_indices) != len(expected_sequence):
                raise ValueError(
                    f"Expected {len(expected_sequence)} CA atoms, found {len(ca_indices)}."
                )
            result["valid_conformers"] = int(trajectory.n_frames)
            if trajectory.n_frames:
                rmsd = md.rmsd(trajectory, trajectory, 0, atom_indices=ca_indices)
                ca = trajectory.xyz[:, ca_indices, :]
                centered = ca - ca.mean(axis=1, keepdims=True)
                radius = np.sqrt(np.mean(np.sum(centered**2, axis=2), axis=1))
                adjacent = np.linalg.norm(ca[:, 1:, :] - ca[:, :-1, :], axis=2)
                result["median_ca_rmsd_nm"] = float(np.median(rmsd))
                result["median_radius_gyration_nm"] = float(np.median(radius))
                result["max_adjacent_ca_distance_nm"] = float(np.max(adjacent))
        result["filtered_conformers"] = max(
            0, int(result["raw_conformers"]) - int(result["valid_conformers"])
        )
    except (
        Exception
    ) as error:  # Persist the reason for machine-readable feasibility decisions.
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def decide_feasibility(
    rows: list[dict[str, Any]],
    *,
    total_training_samples: int = 6387,
    max_projected_gpu_hours: float = 96.0,
    max_projected_storage_gb: float = 500.0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one assessed smoke row is required.")

    def usable(row: dict[str, Any]) -> bool:
        length = int(row["length"])
        diversity = float(row["median_ca_rmsd_nm"])
        radius = float(row["median_radius_gyration_nm"])
        max_gap = float(row["max_adjacent_ca_distance_nm"])
        return bool(
            row.get("returncode") == 0
            and not row.get("error")
            and row.get("sequence_match")
            and row.get("residue_count_match")
            and int(row.get("valid_conformers", 0)) >= 8
            and np.isfinite(diversity)
            and 0.05 <= diversity <= 2.5
            and np.isfinite(radius)
            and radius <= max(1.0, 0.08 * length)
            and np.isfinite(max_gap)
            and max_gap <= 0.50
        )

    evaluations = [{**row, "usable": usable(row)} for row in rows]
    strata = {}
    for low, high in LENGTH_BINS:
        name = f"{low}-{high}"
        selected = [row for row in evaluations if row["length_bin"] == name]
        strata[name] = {
            "samples": len(selected),
            "usable": int(sum(row["usable"] for row in selected)),
            "success_rate": float(np.mean([row["usable"] for row in selected]))
            if selected
            else 0.0,
            "mean_valid_conformers": (
                float(np.mean([row["valid_conformers"] for row in selected]))
                if selected
                else 0.0
            ),
        }

    runtime_complete = all(row.get("returncode") == 0 for row in rows)
    nonzero_seconds = [
        float(row["elapsed_seconds"])
        for row in rows
        if row.get("returncode") == 0 and row["elapsed_seconds"] > 0
    ]
    bytes_per_sample = [
        int(row["storage_bytes"])
        for row in rows
        if row.get("returncode") == 0 and row["storage_bytes"] > 0
    ]
    projected_hours = (
        float(np.median(nonzero_seconds) * total_training_samples / 3600)
        if nonzero_seconds
        else float("inf")
    )
    projected_storage_gb = (
        float(np.mean(bytes_per_sample) * total_training_samples / 1e9)
        if bytes_per_sample
        else float("inf")
    )
    overall_rate = float(np.mean([row["usable"] for row in evaluations]))
    all_strata_stable = all(
        value["samples"] > 0 and value["success_rate"] >= 0.90
        for value in strata.values()
    )
    cost_feasible = bool(
        runtime_complete
        and nonzero_seconds
        and bytes_per_sample
        and projected_hours <= max_projected_gpu_hours
        and projected_storage_gb <= max_projected_storage_gb
    )
    passed = (
        runtime_complete
        and overall_rate >= 0.90
        and all_strata_stable
        and cost_feasible
    )
    if not runtime_complete:
        decision = "retry_bioemu_smoke_after_infrastructure_fix"
        next_stage = "rerun the same frozen six-row BioEmu smoke"
    elif passed:
        decision = "continue_bioemu_200_pilot"
        next_stage = "complete the frozen 200-sample BioEmu pilot"
    else:
        decision = "switch_to_esmflow_small_pilot"
        next_stage = "prepare a small ESMFlow feasibility pilot"
    return {
        "decision": decision,
        "full_generation_authorized": False,
        "next_authorized_stage": next_stage,
        "gate": {
            "runtime_completed_without_errors": runtime_complete,
            "overall_success_rate_at_least_0_90": overall_rate >= 0.90,
            "every_length_stratum_at_least_0_90": all_strata_stable,
            "projected_cost_within_limits": cost_feasible,
            "all_conditions_passed": passed,
        },
        "overall_success_rate": overall_rate,
        "length_strata": strata,
        "cost_projection": {
            "training_samples": total_training_samples,
            "median_seconds_per_sequence": (
                float(np.median(nonzero_seconds)) if nonzero_seconds else None
            ),
            "projected_gpu_hours": projected_hours,
            "max_projected_gpu_hours": max_projected_gpu_hours,
            "mean_storage_bytes_per_sequence": (
                float(np.mean(bytes_per_sample)) if bytes_per_sample else None
            ),
            "projected_storage_gb": projected_storage_gb,
            "max_projected_storage_gb": max_projected_storage_gb,
        },
        "filter_reason": (
            "raw minus valid frames were removed by BioEmu filter_samples=True; "
            "the official output groups steric clashes and chain discontinuities"
        ),
        "samples": evaluations,
    }


def assess_manifest(
    manifest: str | Path,
    output_root: str | Path,
    output_json: str | Path,
    output_csv: str | Path,
    *,
    limit: int,
    max_projected_gpu_hours: float = 96.0,
    max_projected_storage_gb: float = 500.0,
) -> dict[str, Any]:
    manifest_rows = _read_csv(manifest)[:limit]
    root = Path(output_root)
    rows = [assess_sample(row, root / str(row["sample_id"])) for row in manifest_rows]
    report = decide_feasibility(
        rows,
        max_projected_gpu_hours=max_projected_gpu_hours,
        max_projected_storage_gb=max_projected_storage_gb,
    )
    save_json(report, output_json)
    _write_rows(rows, output_csv)
    return report


def _add_common_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=SMOKE_SIZE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--fold-manifest", required=True)
    select.add_argument("--output-manifest", required=True)
    select.add_argument("--output-fasta", required=True)
    select.add_argument("--smoke-manifest", required=True)
    select.add_argument("--smoke-fasta", required=True)
    select.add_argument("--summary-json")
    select.add_argument("--target-size", type=int, default=PILOT_SIZE)
    select.add_argument("--seed", type=int, default=DEFAULT_SEED)

    run = subparsers.add_parser("run")
    _add_common_manifest_args(run)
    run.add_argument("--allow-pilot", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--num-samples", type=int, default=REQUESTED_CONFORMERS)
    run.add_argument("--cache-embeds-dir")
    run.add_argument("--cache-so3-dir")
    run.add_argument("--ckpt-path")
    run.add_argument("--model-config-path")

    assess = subparsers.add_parser("assess")
    _add_common_manifest_args(assess)
    assess.add_argument("--output-json", required=True)
    assess.add_argument("--output-csv", required=True)
    assess.add_argument("--max-projected-gpu-hours", type=float, default=96.0)
    assess.add_argument("--max-projected-storage-gb", type=float, default=500.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "select":
        selected, smoke = select_bioemu_pilot(
            args.fold_manifest,
            args.output_manifest,
            args.output_fasta,
            args.smoke_manifest,
            args.smoke_fasta,
            summary_json=args.summary_json,
            target_size=args.target_size,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "pilot": len(selected),
                    "smoke": len(smoke),
                    "pilot_length_bins": Counter(row["length_bin"] for row in selected),
                    "pilot_labels": Counter(row["label"] for row in selected),
                },
                ensure_ascii=False,
                default=dict,
            )
        )
    elif args.command == "run":
        run_manifest(
            args.manifest,
            args.output_root,
            limit=args.limit,
            allow_pilot=args.allow_pilot,
            dry_run=args.dry_run,
            python=args.python,
            seed=args.seed,
            num_samples=args.num_samples,
            cache_embeds_dir=args.cache_embeds_dir,
            cache_so3_dir=args.cache_so3_dir,
            ckpt_path=args.ckpt_path,
            model_config_path=args.model_config_path,
        )
    else:
        report = assess_manifest(
            args.manifest,
            args.output_root,
            args.output_json,
            args.output_csv,
            limit=args.limit,
            max_projected_gpu_hours=args.max_projected_gpu_hours,
            max_projected_storage_gb=args.max_projected_storage_gb,
        )
        print(json.dumps({"decision": report["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
