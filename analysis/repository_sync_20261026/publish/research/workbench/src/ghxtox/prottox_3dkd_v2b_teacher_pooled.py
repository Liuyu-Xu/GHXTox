"""Export teacher pooled representations (pre-classifier) for V2-B diagnostics."""
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


def _load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    if config["model"].get("soft_cys_topology", False):
        raise ValueError("Teacher checkpoint unexpectedly contains a topology branch")
    model = GHXToxModel(config).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatible.unexpected_keys or incompatible.missing_keys:
        raise ValueError(f"Checkpoint/config mismatch: {path}")
    return model


def _collate(records):
    return collate_peptides(records, include_atom=False)


def export_pooled(protocol_path, output_dir, device_name="cuda"):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8-sig"))
    teacher = protocol["teacher"]
    records = torch.load(Path(teacher["processed_train"]), map_location="cpu", weights_only=False)["records"]
    manifest_rows = _read_csv(Path(teacher["outer_fold_manifest"]))
    if len(manifest_rows) != len(records):
        raise ValueError("Manifest/record count mismatch")
    fold_by_index = {int(row["source_index"]): int(row["fold"]) for row in manifest_rows}
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
        model = _load_model(fold_dir / "best_model.pt", device)
        model.eval()
        loader = DataLoader(records, batch_size=16, collate_fn=_collate, shuffle=False)
        embeddings = []
        with torch.inference_mode():
            for batch in loader:
                batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
                result = model(batch)
                embeddings.append(result["embedding"].detach().cpu().float())
        embedding = torch.cat(embeddings, dim=0)
        if embedding.shape[0] != len(records):
            raise RuntimeError("Teacher pooled embedding row mismatch")
        payload = {
            "outer_fold": fold,
            "embedding": embedding,
            "source_index": list(range(len(records))),
            "sample_id": [str(record["sample_id"]) for record in records],
            "sequence": [str(record["sequence"]) for record in records],
            "label": [int(record["label"]) for record in records],
            "outer_fold_by_sample": [fold_by_index[i] for i in range(len(records))],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
        }
        path = output / f"teacher_pooled_outer{fold}.pt"
        torch.save(payload, path)
        summaries.append({"outer_fold": fold, "rows": len(records), "dim": int(embedding.shape[1]), "file": path.name, "sha256": sha256_file(path), "checkpoint_sha256": provenance["checkpoint_sha256"]})
        print(f"outer fold {fold} pooled exported dim={embedding.shape[1]}", flush=True)
    manifest = {
        "artifact_type": "prottox_3dkd_v2b_teacher_pooled",
        "version": "diagnostic-v1",
        "exports": summaries,
        "definition": "result['embedding'] = masked-mean of fused sequence/spatial nodes before classifier input/global concatenation",
        "selection_scope": "training data only; Test1/Test2 excluded",
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
    result = export_pooled(args.protocol, args.output_dir, device_name=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
