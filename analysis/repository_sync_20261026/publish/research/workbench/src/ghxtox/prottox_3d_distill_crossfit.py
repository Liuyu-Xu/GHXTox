"""Build one audited cross-fitted teacher target row per training sample.

A sample in outer fold k receives its single teacher soft target from
teacher_k (the strict-outer checkpoint that never saw fold k). The result is a
6387-row table used only for final full-data student refit.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ghxtox.models import GHXToxModel
from ghxtox.prottox_3d_distill_targets import ROLE_TO_OUTPUT, _collate, _indices_sha256, _read_csv, sha256_file
from ghxtox.prottox_tsk import sequence_sha256


def _load_teacher_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    import copy
    config = copy.deepcopy(checkpoint["config"])
    if config["model"].get("soft_cys_topology", False):
        raise ValueError("Teacher checkpoint unexpectedly contains a topology branch")
    model = GHXToxModel(config).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatible.unexpected_keys or incompatible.missing_keys:
        raise ValueError(f"Teacher checkpoint/config mismatch for {checkpoint_path}")
    return model


def generate_crossfit_targets(protocol_path, output_csv, device_name="cuda"):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8-sig"))
    teacher = protocol["teacher"]
    train_path = Path(teacher["processed_train"])
    outer_manifest_path = Path(teacher["outer_fold_manifest"])
    checkpoint_root = Path(teacher["checkpoint_root"])
    actual_train_hash = sha256_file(train_path)
    actual_outer_hash = sha256_file(outer_manifest_path)
    if actual_train_hash.lower() != str(teacher["processed_train_sha256"]).lower():
        raise ValueError("Processed training cache hash mismatch")
    if actual_outer_hash.lower() != str(teacher["outer_fold_manifest_sha256"]).lower():
        raise ValueError("Outer fold manifest hash mismatch")
    records = torch.load(train_path, map_location="cpu", weights_only=False)["records"]
    outer_rows = _read_csv(outer_manifest_path)
    if len(outer_rows) != len(records):
        raise ValueError("Outer manifest row count does not match records")
    fold_by_index = {}
    for row in outer_rows:
        index = int(row["source_index"])
        if records[index]["sample_id"] != row["sample_id"] or records[index]["sequence"] != row["sequence"]:
            raise ValueError("Outer manifest identity mismatch")
        fold_by_index[index] = int(row["fold"])
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    rows = []
    for fold in range(5):
        fold_dir = checkpoint_root / f"fold{fold}"
        provenance = json.loads((fold_dir / "provenance.json").read_text(encoding="utf-8-sig"))
        checkpoint_path = fold_dir / "best_model.pt"
        if sha256_file(checkpoint_path).lower() != str(provenance["checkpoint_sha256"]).lower():
            raise ValueError(f"Checkpoint hash mismatch for fold {fold}")
        if provenance.get("external_test_labels_used") is not False or provenance.get("all_supervised_ancestors_included") is not True:
            raise ValueError(f"Provenance flags are invalid for fold {fold}")
        if provenance.get("selection_indices") not in (None, []):
            raise ValueError(f"Teacher fold {fold} used non-empty selection indices")
        held = {i for i, f in fold_by_index.items() if f == fold}
        if set(provenance["outer_test_indices"]) != held:
            raise ValueError(f"Provenance outer-test indices do not match manifest for fold {fold}")
        target_indices = sorted(held)
        model = _load_teacher_model(checkpoint_path, device)
        model.eval()
        loader = DataLoader([records[i] for i in target_indices], batch_size=int(protocol["teacher_targets"].get("batch_size", 16)), collate_fn=_collate, shuffle=False)
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
                logits = model(batch)["logits"]
                logits_cpu = logits.cpu().tolist()
                probability_cpu = torch.sigmoid(logits).cpu().tolist()
                for local, logit, probability in zip(range(len(logits_cpu)), logits_cpu, probability_cpu, strict=True):
                    index = target_indices[offset + local]
                    record = records[index]
                    rows.append(
                        {
                            "source_index": index,
                            "sample_id": record["sample_id"],
                            "sequence_hash": sequence_sha256(str(record["sequence"])),
                            "sequence": record["sequence"],
                            "outer_fold": fold,
                            "teacher_sample_role": "outer_holdout",
                            "teacher_probability": f"{probability:.9g}",
                            "teacher_logit": f"{logit:.9g}",
                            "checkpoint_sha256": provenance["checkpoint_sha256"],
                            "teacher_train_indices_sha256": _indices_sha256(provenance["fitting_indices"]),
                            "outer_manifest_sha256": provenance["fold_manifest_sha256"],
                            "teacher_config_sha256": provenance["base_config_sha256"],
                        }
                    )
                offset += len(logits_cpu)
        if offset != len(target_indices):
            raise RuntimeError(f"Fold {fold}: emitted {offset}/{len(target_indices)} rows")
        print(f"crossfit fold={fold} rows={len(target_indices)} done", flush=True)
    rows.sort(key=lambda row: int(row["source_index"]))
    if len(rows) != len(records):
        raise ValueError(f"Cross-fitted table has {len(rows)} rows; expected {len(records)}")
    seen = {int(row["source_index"]) for row in rows}
    if seen != set(range(len(records))) or len(seen) != len(rows):
        raise ValueError("Cross-fitted table must contain every source index exactly once")
    for row, record in zip(rows, records, strict=True):
        if int(row["outer_fold"]) != fold_by_index[int(row["source_index"])]:
            raise ValueError("Cross-fitted teacher fold does not match the sample's outer fold")
        if row["sample_id"] != record["sample_id"] or row["sequence_hash"] != sequence_sha256(str(record["sequence"])):
            raise ValueError("Cross-fitted table identity mismatch with training records")
        if not all(float(row[key]) == float(row[key]) for key in ("teacher_probability", "teacher_logit")):
            raise ValueError("Cross-fitted teacher outputs must be finite")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_index",
        "sample_id",
        "sequence_hash",
        "sequence",
        "outer_fold",
        "teacher_sample_role",
        "teacher_probability",
        "teacher_logit",
        "checkpoint_sha256",
        "teacher_train_indices_sha256",
        "outer_manifest_sha256",
        "teacher_config_sha256",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "artifact_type": "prottox_3d_distill_crossfit_teacher_targets",
        "version": 1,
        "rows": len(rows),
        "file": str(output_path),
        "sha256": sha256_file(output_path),
        "audit": {
            "rows_equal_records": len(rows) == len(records),
            "no_duplicate_source_index": len(seen) == len(rows),
            "every_teacher_fold_matches_sample_outer_fold": True,
            "all_roles_are_outer_holdout": all(row["teacher_sample_role"] == "outer_holdout" for row in rows),
            "all_outputs_finite": True,
        },
        "selection_scope": "training data only; Test1/Test2 excluded",
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    (output_path.parent / "crossfit_targets_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = generate_crossfit_targets(args.protocol, args.output_csv, device_name=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
