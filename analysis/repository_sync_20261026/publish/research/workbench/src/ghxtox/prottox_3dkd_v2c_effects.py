"""Generate teacher component intervention logits for V2-C (口径 A)."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ghxtox.data import collate_peptides
from ghxtox.models import GHXToxModel
from ghxtox.prottox_3d_distill_targets import _read_csv, sha256_file


def _collate(records):
    return collate_peptides(records, include_atom=False)


def _load_teacher(path, device, *, no_chem: bool):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    if config["model"].get("soft_cys_topology", False):
        raise ValueError("Teacher checkpoint unexpectedly has topology branch")
    if no_chem:
        config["model"]["chemical_site_branch"] = False
    model = GHXToxModel(config).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if no_chem:
        unexpected_chemical = [key for key in incompatible.unexpected_keys if not key.startswith("chemical_site_branch.")]
        if unexpected_chemical:
            raise ValueError(f"No-chem teacher has unexpected non-chemical keys: {unexpected_chemical[:5]}")
        if incompatible.missing_keys:
            raise ValueError("No-chem teacher checkpoint is missing required keys")
    else:
        if incompatible.unexpected_keys or incompatible.missing_keys:
            raise ValueError("Full teacher checkpoint has unexpected/missing keys")
    return model, config


def _forward_logits(model, records, device, *, no_geo: bool):
    original_forward = None
    if no_geo:
        original_forward = model.fusion.forward

        def zero_spatial_forward(*args, **kwargs):
            if "spatial_h" in kwargs:
                kwargs["spatial_h"] = torch.zeros_like(kwargs["spatial_h"])
            return original_forward(*args, **kwargs)

        model.fusion.forward = zero_spatial_forward
    logits = []
    loader = DataLoader(records, batch_size=16, collate_fn=_collate, shuffle=False)
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
                logits.append(model(batch)["logits"].detach().cpu())
    finally:
        if original_forward is not None:
            model.fusion.forward = original_forward
    return torch.cat(logits, dim=0)


def generate_effects(protocol_path, output_dir, device_name="cuda"):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8-sig"))
    teacher = protocol["teacher"]
    records = torch.load(Path(teacher["processed_train"]), map_location="cpu", weights_only=False)["records"]
    outer_rows = _read_csv(Path(teacher["outer_fold_manifest"]))
    if len(outer_rows) != len(records):
        raise ValueError("Manifest/record count mismatch")
    fold_by_index = {int(row["source_index"]): int(row["fold"]) for row in outer_rows}
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    summaries = []
    for fold in range(5):
        fold_dir = Path(teacher["checkpoint_root"]) / f"fold{fold}"
        provenance = json.loads((fold_dir / "provenance.json").read_text(encoding="utf-8-sig"))
        if sha256_file(fold_dir / "best_model.pt").lower() != str(provenance["checkpoint_sha256"]).lower():
            raise ValueError(f"Checkpoint hash mismatch fold {fold}")
        full_model, _config = _load_teacher(fold_dir / "best_model.pt", device, no_chem=False)
        no_chem_model, _config2 = _load_teacher(fold_dir / "best_model.pt", device, no_chem=True)
        z_full = _forward_logits(full_model, records, device, no_geo=False)
        z_no_chem = _forward_logits(no_chem_model, records, device, no_geo=False)
        z_no_geo = _forward_logits(full_model, records, device, no_geo=True)
        z_no_chem_no_geo = _forward_logits(no_chem_model, records, device, no_geo=True)
        payload = {
            "outer_fold": fold,
            "sample_id": [str(record["sample_id"]) for record in records],
            "label": [int(record["label"]) for record in records],
            "outer_fold_by_sample": [fold_by_index[i] for i in range(len(records))],
            "z_full": z_full,
            "z_no_chem": z_no_chem,
            "z_no_geo": z_no_geo,
            "z_no_chem_no_geo": z_no_chem_no_geo,
            "checkpoint_sha256": provenance["checkpoint_sha256"],
        }
        path = output / f"v2c_component_effects_outer{fold}.pt"
        torch.save(payload, path)
        summaries.append({"outer_fold": fold, "file": path.name, "sha256": sha256_file(path), "checkpoint_sha256": provenance["checkpoint_sha256"]})
        print(f"outer fold {fold} component effects generated", flush=True)
    manifest = {
        "artifact_type": "prottox_3dkd_v2c_component_effects",
        "version": "v2c-1",
        "definition": protocol["intervention_definitions"],
        "exports": summaries,
        "selection_scope": "training diagnostic only; Test1/Test2 excluded",
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = generate_effects(args.protocol, args.output_dir, device_name=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
