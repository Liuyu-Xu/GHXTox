"""Fixed-epoch, group-held-out prototype. No external test input is accepted."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ghxtox.data import collate_peptides
from ghxtox.models import GHXToxModel
from ghxtox.soft_cys_topology import cached_candidate_features
from ghxtox.utils import move_batch_to_device, set_seed


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_peak_rss_bytes():
    """OS process-lifetime peak resident memory; includes native tensor allocations."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak_working", "working", "peak_paged", "paged",
                                                    "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not get_info(process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return int(counters.peak_working)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def validate_manifest(records, rows):
    if len(rows) != len(records):
        raise ValueError("Fold manifest must contain exactly one row per training record")
    seen, groups, sequences, assignments = set(), {}, {}, {}
    for row in rows:
        i, fold = int(row["source_index"]), int(row["fold"])
        if i in seen or not 0 <= i < len(records) or fold not in range(5):
            raise ValueError("Duplicate/out-of-range manifest index or fold")
        seen.add(i)
        record = records[i]
        if str(record["sample_id"]) != row["sample_id"] or record["sequence"] != row["sequence"]:
            raise ValueError("Manifest identity mismatch")
        if int(record["label"]) != int(row["label"]) or int(row["label"]) not in (0, 1):
            raise ValueError("Manifest label mismatch")
        group = row.get("group", row.get("group_id", ""))
        if not group:
            raise ValueError("Empty sequence group")
        for mapping, key in ((groups, group), (sequences, record["sequence"])):
            if key in mapping and mapping[key] != fold:
                raise ValueError("Sequence/group crosses held-out folds")
            mapping[key] = fold
        assignments[i] = fold
    if set(assignments.values()) != set(range(5)):
        raise ValueError("All five folds must be populated")
    return assignments


def validate_checkpoint_provenance(provenance, checkpoint_path, train_hash, manifest_hash, held_indices, size):
    if str(provenance.get("checkpoint_sha256", "")).lower() != sha256(checkpoint_path).lower():
        raise ValueError("Checkpoint provenance hash mismatch")
    if (str(provenance.get("training_data_sha256", "")).lower() != str(train_hash).lower()
            or str(provenance.get("fold_manifest_sha256", "")).lower() != str(manifest_hash).lower()):
        raise ValueError("Checkpoint data/manifest provenance mismatch")
    if provenance.get("external_test_labels_used") is not False:
        raise ValueError("Checkpoint provenance must exclude external test labels")
    if provenance.get("all_supervised_ancestors_included") is not True:
        raise ValueError("Provenance must include all supervised checkpoint ancestors")
    for key in ("fitting_indices", "selection_indices"):
        values = provenance.get(key)
        if not isinstance(values, list) or any(type(i) is not int or not 0 <= i < size for i in values):
            raise ValueError(f"Invalid checkpoint {key}")
        if set(values) & set(held_indices):
            raise ValueError(f"Held-out fold influenced checkpoint {key}")
    if not provenance["fitting_indices"]:
        raise ValueError("Retained supervised checkpoint requires fitting provenance")


def collate_topology(records):
    batch = collate_peptides(records, include_atom=False)
    if all("soft_cys_cache" in record for record in records):
        batch["soft_cys_cache"] = [record["soft_cys_cache"] for record in records]
    return batch


def freeze_backbone(model):
    """Frozen dropout/normalization as well as parameters; gradients through base remain."""
    if model.soft_cys_topology is None:
        raise ValueError("Topology branch is disabled")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("soft_cys_topology."))
    model.eval()
    model.soft_cys_topology.train()
    return [p for p in model.parameters() if p.requires_grad]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8-sig"))
    if protocol.get("status") != "frozen_before_candidate_oof":
        raise ValueError("Resolve template inputs and freeze protocol before execution")
    if protocol.get("scope") != "training_oof_only" or protocol.get("checkpoint_selection") != "fixed_epochs":
        raise ValueError("Only training OOF with fixed epochs is supported")
    for key in ("epochs", "batch_size", "stability_repeats"):
        if type(protocol[key]) is not int or protocol[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "weight_decay", "stability_sigma"):
        if not math.isfinite(protocol[key]) or protocol[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if protocol["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    if set(protocol["model_overrides"]) - {"topology_max_distance", "topology_max_candidates", "topology_temperature"}:
        raise ValueError("Prototype overrides must not change retained backbone or fusion")
    variants = protocol["variants"]
    if set(variants) != {"local", "matching", "topology", "null"} or len(variants) != 4:
        raise ValueError("All four prespecified variants are required")
    train_path, manifest_path = protocol["processed_train"], protocol["fold_manifest"]
    train_hash, manifest_hash = sha256(train_path), sha256(manifest_path)
    if protocol.get("training_data_sha256") != train_hash or protocol.get("fold_manifest_sha256") != manifest_hash:
        raise ValueError("Freeze actual training/manifest hashes in protocol before running")
    records = torch.load(train_path, map_location="cpu", weights_only=False)["records"]
    with open(manifest_path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assignment = validate_manifest(records, rows)
    checkpoints = protocol["checkpoints"]
    if {entry["fold"] for entry in checkpoints} != set(range(5)) or len(checkpoints) != 5:
        raise ValueError("Exactly five fold-specific checkpoints are required")
    # Validate every fold before any fitting or output creation.
    checkpoint_provenance = {}
    for entry in checkpoints:
        provenance = json.loads(Path(entry["provenance"]).read_text(encoding="utf-8-sig"))
        held = [i for i in assignment if assignment[i] == entry["fold"]]
        validate_checkpoint_provenance(provenance, entry["path"], train_hash, manifest_hash, held, len(records))
        checkpoint_provenance[entry["fold"]] = provenance
    if args.preflight_only:
        print("Five-fold input identity and checkpoint provenance preflight passed")
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    started = time.perf_counter()
    distance = float(protocol["model_overrides"]["topology_max_distance"])
    print(f"Building soft-Cys candidate cache for {len(records)} training records...", flush=True)
    for record_index, record in enumerate(records, start=1):
        record["soft_cys_cache"] = cached_candidate_features(record, distance)
        if record_index % 500 == 0 or record_index == len(records):
            print(f"candidate_cache={record_index}/{len(records)}", flush=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    predictions, resource_rows = [], []
    for entry in sorted(checkpoints, key=lambda item: item["fold"]):
        fold = entry["fold"]
        train_indices = sorted(checkpoint_provenance[fold]["fitting_indices"])
        held_indices = sorted(i for i in assignment if assignment[i] == fold)
        checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=False)
        config = copy.deepcopy(checkpoint["config"])
        if config["model"].get("soft_cys_topology", False):
            raise ValueError("Expected a retained backbone without topology branch")
        state = checkpoint["model_state"]
        held_loader = DataLoader([records[i] for i in held_indices], batch_size=protocol["batch_size"],
                                 collate_fn=collate_topology, shuffle=False)
        for variant in ["baseline"] + variants:
            print(f"fold={fold} variant={variant} starting", flush=True)
            tick = time.perf_counter()
            set_seed(protocol["seed"] + fold)
            variant_config = copy.deepcopy(config)
            if variant != "baseline":
                variant_config["model"].update(protocol["model_overrides"])
                variant_config["model"].update(soft_cys_topology=True, topology_mode=variant)
            model = GHXToxModel(variant_config).to(device)
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys or any(not key.startswith("soft_cys_topology.") for key in incompatible.missing_keys):
                raise ValueError("Backbone checkpoint/config mismatch")
            if variant not in {"baseline", "null"}:
                params = freeze_backbone(model)
                optimizer = torch.optim.AdamW(params, lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
                loader = DataLoader([records[i] for i in train_indices], batch_size=protocol["batch_size"],
                                    shuffle=True, collate_fn=collate_topology)
                for epoch in range(protocol["epochs"]):
                    loss_sum, sample_count = 0.0, 0
                    for batch in loader:
                        batch = move_batch_to_device(batch, device)
                        optimizer.zero_grad(set_to_none=True)
                        result = model(batch)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(result["logits"], batch["labels"])
                        loss.backward()
                        optimizer.step()
                        loss_sum += float(loss) * len(result["logits"])
                        sample_count += len(result["logits"])
                    print(
                        f"fold={fold} variant={variant} epoch={epoch + 1}/{protocol['epochs']} "
                        f"train_loss={loss_sum / sample_count:.4f}",
                        flush=True,
                    )
            model.eval()
            offset = 0
            with torch.no_grad():
                for batch in held_loader:
                    batch = move_batch_to_device(batch, device)
                    result = model(batch)
                    probability = result["logits"].sigmoid()
                    drift = torch.zeros_like(probability)
                    pairing_drift = torch.zeros_like(probability)
                    if variant == "topology":
                        # Rebuild from perturbed geometry, deliberately bypassing cached features.
                        for _ in range(protocol["stability_repeats"]):
                            perturbed = dict(batch)
                            perturbed.pop("soft_cys_cache", None)
                            scale = protocol["stability_sigma"] * (1 - batch["plddt"]).unsqueeze(-1)
                            noise = torch.randn_like(batch["coords"]) * scale * batch["mask"].unsqueeze(-1)
                            perturbed["coords"] = batch["coords"] + noise
                            perturbed["chemical_site_coords"] = batch["chemical_site_coords"] + noise.unsqueeze(2)
                            if "backbone_coords" in batch:
                                perturbed["backbone_coords"] = batch["backbone_coords"] + noise.unsqueeze(2)
                            changed = model(perturbed)
                            drift += (changed["logits"].sigmoid() - probability).abs() / protocol["stability_repeats"]
                            for b, (before, after) in enumerate(zip(result["topology_evidence"], changed["topology_evidence"])):
                                if before["pairing"].numel():
                                    pairing_drift[b] += (before["pairing"] - after["pairing"]).abs().mean() / protocol["stability_repeats"]
                    for b in range(len(probability)):
                        i = held_indices[offset + b]
                        row = {"source_index": i, "sample_id": records[i]["sample_id"],
                               "sequence": records[i]["sequence"], "label": int(records[i]["label"]),
                               "fold": fold, "variant": variant, "probability": float(probability[b]),
                               "probability_perturbation_mae": float(drift[b]) if variant == "topology" else "",
                               "pairing_perturbation_mae": float(pairing_drift[b]) if variant == "topology" else ""}
                        predictions.append(row)
                    if "topology_evidence" in result:
                        with (out / "candidate_topology_evidence.jsonl").open("a", encoding="utf-8") as handle:
                            for b, evidence in enumerate(result["topology_evidence"]):
                                exported = {"source_index": held_indices[offset + b], "fold": fold,
                                            "variant": variant, "interpretation": "candidate_pairing_not_native_bonds",
                                            "residue_indices_zero_based": evidence["residue_indices"].cpu().tolist(),
                                            "pairing": evidence["pairing"].cpu().tolist(), "free": evidence["free"].cpu().tolist(),
                                            "candidates_compact_cys_indices": evidence["candidates"] if variant == "topology" else None,
                                            "candidate_weights": evidence["weights"].cpu().tolist() if variant == "topology" else None}
                                handle.write(json.dumps(exported, allow_nan=False) + "\n")
                    offset += len(probability)
            torch.save({"config": variant_config, "model_state": model.cpu().state_dict(),
                        "fold": fold, "fitting_indices": train_indices, "selection_indices": [],
                        "protocol_sha256": sha256(args.protocol)}, out / f"fold{fold}_{variant}.pt")
            resource_rows.append({"fold": fold, "variant": variant, "topology_fitting_samples": len(train_indices), "seconds": time.perf_counter() - tick})
            print(f"fold={fold} variant={variant} complete", flush=True)
            del model
        with (out / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
    audit = {"protocol_sha256": sha256(args.protocol), "seconds": time.perf_counter() - started,
             "runs": resource_rows, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
             "process_lifetime_peak_rss_bytes": process_peak_rss_bytes(),
             "status": "complete", "external_test_labels_used": False}
    (out / "run_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Topology OOF complete: {out}", flush=True)


if __name__ == "__main__":
    main()
