"""Run one fixed-epoch topology-branch smoke without reading outer-test labels."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ghxtox.metrics import binary_metrics
from ghxtox.models import GHXToxModel
from ghxtox.nested_folds import load_nested_indices
from ghxtox.topology_oof import (
    collate_topology,
    freeze_backbone,
    process_peak_rss_bytes,
    sha256,
    validate_checkpoint_provenance,
)
from ghxtox.utils import move_batch_to_device, set_seed


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _base_provenance(
    checkpoint: Path,
    processed: Path,
    nested_manifest: Path,
    outer_fold: int,
    roles: dict[str, list[int]],
    run_config: dict,
) -> dict:
    train = run_config["train"]
    if train.get("selection_mode") != "fixed_epoch":
        raise ValueError("A strict smoke base must use fixed_epoch selection")
    if "initial_checkpoint" in train or "trainable_prefixes" in train:
        raise ValueError("The smoke base must be scratch-trained, with no inherited supervised trunk")
    return {
        "artifact_type": "ghxtox_strict_nested_scratch_base_provenance",
        "version": 1,
        "fold": outer_fold,
        "checkpoint_sha256": sha256(checkpoint),
        "training_data_sha256": sha256(processed),
        "fold_manifest_sha256": sha256(nested_manifest),
        "checkpoint": str(checkpoint),
        "processed_train": str(processed),
        "nested_manifest": str(nested_manifest),
        "external_test_labels_used": False,
        "all_supervised_ancestors_included": True,
        "supervised_ancestor_lineage": [],
        "initialization": "scratch",
        "fitting_indices": roles["train"],
        "selection_indices": [],
        "diagnostic_validation_indices": roles["validation"],
        "unused_calibration_indices": roles["calibration"],
        "outer_test_indices": roles["test"],
        "selection_policy": "fixed_epoch=1; inner-validation metrics are diagnostic only",
        "eligible_for_strict_outer_fold_topology_oof": True,
    }


def _evaluate(model, records, indices, device, batch_size):
    loader = DataLoader([records[i] for i in indices], batch_size=batch_size, shuffle=False, collate_fn=collate_topology)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            logits.append(output["logits"].detach().cpu())
            labels.append(batch["labels"].detach().cpu())
    return binary_metrics(torch.cat(logits), torch.cat(labels))


def _outer_probabilities(model, records, indices, device, batch_size):
    loader = DataLoader([records[i] for i in indices], batch_size=batch_size, shuffle=False, collate_fn=collate_topology)
    rows, offset = [], 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # The outer-test labels remain in the source tensor but are removed before inference.
            batch["labels"] = None
            batch = move_batch_to_device(batch, device)
            probabilities = model(batch)["logits"].sigmoid().cpu().tolist()
            for probability in probabilities:
                index = indices[offset]
                record = records[index]
                rows.append({"source_index": index, "sample_id": record["sample_id"],
                             "sequence": record["sequence"], "fold": 0, "probability": probability})
                offset += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--nested-manifest", required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs != 1:
        raise ValueError("This local smoke is deliberately fixed to one epoch")
    if args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("batch-size and learning-rate must be positive")
    processed, nested, base_path, output = map(Path, (args.processed, args.nested_manifest, args.base_checkpoint, args.output_dir))
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing output: {output}")
    payload = torch.load(processed, map_location="cpu", weights_only=False)
    records = payload["records"]
    roles = load_nested_indices(nested, args.outer_fold, len(records))
    checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("Base checkpoint is missing config or model_state")
    base_config = checkpoint["config"]
    provenance = _base_provenance(base_path, processed, nested, args.outer_fold, roles, base_config)
    validate_checkpoint_provenance(provenance, base_path, provenance["training_data_sha256"],
                                   provenance["fold_manifest_sha256"], roles["test"], len(records))
    topology_config = copy.deepcopy(base_config)
    topology_config["model"].update({"soft_cys_topology": True, "topology_mode": "topology",
                                     "topology_max_distance": 8.0, "topology_max_candidates": 8,
                                     "topology_temperature": 1.0})
    set_seed(int(topology_config.get("seed", 2026)))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output.mkdir(parents=True, exist_ok=False)
    (output / "base_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "topology_config.json").write_text(json.dumps(topology_config, indent=2) + "\n", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    model = GHXToxModel(topology_config).to(device)
    incompatibility = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatibility.unexpected_keys or any(not key.startswith("soft_cys_topology.") for key in incompatibility.missing_keys):
        raise ValueError("Base checkpoint does not match topology model contract")
    trainable = freeze_backbone(model)
    frozen_snapshot = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("soft_cys_topology.")
    }
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0001)
    loader = DataLoader([records[i] for i in roles["train"]], batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_topology)
    started = time.perf_counter()
    model.soft_cys_topology.train()
    loss_sum, samples = 0.0, 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)["logits"]
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        loss_sum += float(loss) * len(logits)
        samples += len(logits)
    changed_frozen = [
        name for name, parameter in model.named_parameters()
        if name in frozen_snapshot and not torch.equal(parameter.detach().cpu(), frozen_snapshot[name])
    ]
    if changed_frozen:
        raise RuntimeError(f"Frozen base parameters changed: {changed_frozen[:5]}")
    validation = _evaluate(model, records, roles["validation"], device, args.batch_size)
    outer = _outer_probabilities(model, records, roles["test"], device, args.batch_size)
    with (output / "outer_test_probabilities.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outer[0]))
        writer.writeheader()
        writer.writerows(outer)
    topology_provenance = {
        "artifact_type": "ghxtox_topology_branch_smoke_provenance",
        "version": 1,
        "base_provenance_sha256": sha256(output / "base_provenance.json"),
        "fitting_indices": roles["train"], "diagnostic_validation_indices": roles["validation"],
        "outer_test_indices": roles["test"], "external_test_labels_used": False,
        "all_non_topology_parameters_frozen": True,
        "epochs": 1, "learning_rate": args.learning_rate, "batch_size": args.batch_size,
    }
    torch.save({"model_state": model.cpu().state_dict(), "config": topology_config,
                "base_provenance": provenance, "topology_provenance": topology_provenance}, output / "topology_branch.pt")
    audit = {"status": "complete", "purpose": "local feasibility smoke; no performance conclusion",
             "outer_test_labels_read": False, "train_loss": loss_sum / samples,
             "frozen_non_topology_parameters_changed": 0,
             "validation_metrics": validation, "elapsed_seconds": time.perf_counter() - started,
             "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
             "process_lifetime_peak_rss_bytes": process_peak_rss_bytes(), "outer_predictions": len(outer)}
    (output / "run_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
