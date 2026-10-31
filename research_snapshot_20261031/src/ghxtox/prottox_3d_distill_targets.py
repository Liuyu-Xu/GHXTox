"""Generate outer-specific single-structure 3D-expert teacher targets.

The five teacher checkpoints are the strict-outer-fold GHXToxModel scratch
bases retained in topology_strict_oof_v1_final.tar.gz. For outer fold k, the
teacher_k checkpoint is applied only to records whose outer fold is not k, so
the resulting targets are outer-safe for student validation fold k.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ghxtox.data import collate_peptides
from ghxtox.models import GHXToxModel


ROLE_TO_OUTPUT = {"train": "fitting", "validation": "inner_validation", "calibration": "calibration"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _indices_sha256(indices: list[int]) -> str:
    canonical = json.dumps(sorted(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    if config["model"].get("soft_cys_topology", False):
        raise ValueError("Teacher checkpoint unexpectedly contains a topology branch")
    model = GHXToxModel(config).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatible.unexpected_keys or incompatible.missing_keys:
        raise ValueError(f"Teacher checkpoint/config mismatch for {checkpoint_path}")
    return model, config


def _collate(records):
    return collate_peptides(records, include_atom=False)


def generate_targets(protocol_path, output_dir, device_name="cuda", preflight_only=False):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8-sig"))
    if protocol.get("status") != "frozen_before_target_generation":
        raise ValueError("Protocol is not frozen before target generation")
    if protocol.get("scope") != "training_oof_only":
        raise ValueError("Only training-OOF teacher targets are supported")
    teacher = protocol["teacher"]
    train_path = Path(teacher["processed_train"])
    outer_manifest_path = Path(teacher["outer_fold_manifest"])
    nested_manifest_path = Path(teacher["nested_manifest"])
    checkpoint_root = Path(teacher["checkpoint_root"])

    print("Hashing training cache and manifests...", flush=True)
    actual_train_hash = sha256_file(train_path)
    actual_outer_hash = sha256_file(outer_manifest_path)
    actual_nested_hash = sha256_file(nested_manifest_path)
    if actual_train_hash.lower() != str(teacher["processed_train_sha256"]).lower():
        raise ValueError("Processed training cache hash mismatch")
    if actual_outer_hash.lower() != str(teacher["outer_fold_manifest_sha256"]).lower():
        raise ValueError("Outer fold manifest hash mismatch")
    if actual_nested_hash.lower() != str(teacher["nested_manifest_sha256"]).lower():
        raise ValueError("Nested manifest hash mismatch")

    records = torch.load(train_path, map_location="cpu", weights_only=False)["records"]
    outer_rows = _read_csv(outer_manifest_path)
    nested_rows = _read_csv(nested_manifest_path)
    if len(outer_rows) != len(records) or len(nested_rows) != len(records) * 5:
        raise ValueError("Manifest row counts do not match records")
    outer_fold_by_index = {}
    for row in outer_rows:
        index = int(row["source_index"])
        fold = int(row["fold"])
        if records[index]["sample_id"] != row["sample_id"] or records[index]["sequence"] != row["sequence"]:
            raise ValueError("Outer manifest identity mismatch")
        outer_fold_by_index[index] = fold
    nested_role_by_key = {}
    for row in nested_rows:
        nested_role_by_key[(int(row["outer_fold"]), int(row["source_index"]))] = str(row["role"])
    if set(nested_role_by_key) != {(k, i) for k in range(5) for i in range(len(records))}:
        raise ValueError("Nested manifest does not assign every outer-fold/source pair")

    provenances = {}
    for fold in range(5):
        fold_dir = checkpoint_root / f"fold{fold}"
        provenance = json.loads((fold_dir / "provenance.json").read_text(encoding="utf-8-sig"))
        if sha256_file(fold_dir / "best_model.pt").lower() != str(provenance["checkpoint_sha256"]).lower():
            raise ValueError(f"Checkpoint hash mismatch for fold {fold}")
        if str(provenance["training_data_sha256"]).lower() != actual_train_hash.lower():
            raise ValueError(f"Training-cache provenance mismatch for fold {fold}")
        if str(provenance["fold_manifest_sha256"]).lower() != actual_outer_hash.lower():
            raise ValueError(f"Fold-manifest provenance mismatch for fold {fold}")
        if provenance.get("external_test_labels_used") is not False or provenance.get("all_supervised_ancestors_included") is not True:
            raise ValueError(f"Provenance flags are invalid for fold {fold}")
        if provenance.get("selection_indices") not in (None, []):
            raise ValueError(f"Teacher fold {fold} used non-empty selection indices")
        fitting = set(provenance["fitting_indices"])
        calibration = set(provenance["unused_calibration_indices"])
        if fitting & calibration:
            raise ValueError(f"Fold {fold}: fitting and calibration indices overlap")
        if any(outer_fold_by_index[i] == fold for i in fitting | calibration):
            raise ValueError(f"Fold {fold}: provenance touched held-out outer fold samples")
        for role_name, expected in (("train", fitting), ("calibration", calibration)):
            actual = {i for (k, i), role in nested_role_by_key.items() if k == fold and role == role_name}
            if actual != expected:
                raise ValueError(f"Fold {fold}: nested role {role_name} does not match provenance")
        validation = {i for (k, i), role in nested_role_by_key.items() if k == fold and role == "validation"}
        if validation & (fitting | calibration):
            raise ValueError(f"Fold {fold}: inner validation leaked into fitting/calibration")
        provenances[fold] = provenance
        print(f"fold={fold} provenance/outer-safety preflight passed", flush=True)

    if preflight_only:
        return {"preflight": "passed", "records": len(records), "teacher_folds": sorted(provenances)}

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing target directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    manifest_rows = []
    all_role_counts = {}
    for fold in range(5):
        provenance = provenances[fold]
        target_indices = sorted(i for i, f in outer_fold_by_index.items() if f != fold)
        roles = {i: ROLE_TO_OUTPUT[nested_role_by_key[(fold, i)]] for i in target_indices}
        role_counts = {role: sum(1 for value in roles.values() if value == role) for role in ("fitting", "inner_validation", "calibration")}
        all_role_counts[str(fold)] = role_counts
        model, _config = _load_model(checkpoint_root / f"fold{fold}" / "best_model.pt", device)
        model.eval()
        loader = DataLoader(
            [records[i] for i in target_indices],
            batch_size=int(protocol["teacher_targets"].get("batch_size", 16)),
            collate_fn=_collate,
            shuffle=False,
        )
        csv_path = output / f"teacher_outer{fold}_targets.csv"
        fieldnames = protocol["teacher_targets"]["columns"]
        offset = 0
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            with torch.inference_mode():
                for batch in loader:
                    batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
                    logits = model(batch)["logits"]
                    logits_cpu = logits.cpu().tolist()
                    probability_cpu = torch.sigmoid(logits).cpu().tolist()
                    for local, logit, probability in zip(range(len(logits_cpu)), logits_cpu, probability_cpu, strict=True):
                        index = target_indices[offset + local]
                        record = records[index]
                        writer.writerow(
                            {
                                "source_index": index,
                                "sample_id": record["sample_id"],
                                "sequence": record["sequence"],
                                "outer_fold": fold,
                                "teacher_sample_role": roles[index],
                                "teacher_probability": f"{probability:.9g}",
                                "teacher_logit": f"{logit:.9g}",
                                "checkpoint_sha256": provenance["checkpoint_sha256"],
                                "teacher_train_indices_sha256": _indices_sha256(provenance["fitting_indices"]),
                                "outer_manifest_sha256": provenance["fold_manifest_sha256"],
                                "nested_manifest_sha256": provenance["nested_manifest_sha256"],
                                "teacher_config_sha256": provenance["base_config_sha256"],
                            }
                        )
                    offset += len(logits_cpu)
        if offset != len(target_indices):
            raise RuntimeError(f"Fold {fold}: emitted {offset}/{len(target_indices)} teacher rows")
        manifest_rows.append(
            {
                "outer_fold": fold,
                "file": csv_path.name,
                "rows": len(target_indices),
                "role_counts": role_counts,
                "sha256": sha256_file(csv_path),
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(f"fold={fold} rows={len(target_indices)} roles={role_counts} done", flush=True)

    summary = {
        "artifact_type": "prottox_3d_distill_teacher_targets",
        "version": 1,
        "protocol": str(protocol_path),
        "teacher_definition": teacher["definition"],
        "records_total": len(records),
        "targets": manifest_rows,
        "role_counts_by_outer_fold": all_role_counts,
        "selection_scope": "training OOF only; Test1/Test2 excluded",
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    (output / "target_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = generate_targets(args.protocol, args.output_dir, device_name=args.device, preflight_only=args.preflight_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
