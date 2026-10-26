"""Build a read-only prototype profile map for the frozen MPCE-3DKD product.

The map is generated from the canonical batch-size-1 product cache and the
frozen single-forward product checkpoints.  It describes, for each prototype,
the model-associated PhysChem profile of training samples that are closest to
it.  It does not retrain or modify the model, does not change probabilities,
and is used only for auxiliary display/interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ghxtox.models import PHYS_CHEM_FEATURE_NAMES
from ghxtox.mpce_3dkd_predict import load_ensemble_manifest, load_product_checkpoint
from ghxtox.prottox_tsk import file_sha256
from ghxtox.train_mpce_net import MPCEB, PROTOTYPES_PER_CLASS


ROOT = Path(r"D:\dachuang2026\GHXTox")
RUNS = ROOT / "runs"
CACHE = RUNS / "mpce_3dkd_product_single_cache.pt"
MANIFEST = RUNS / "mpce_3dkd_product_single_refit_v1" / "ensemble_manifest.json"
OUTPUT = RUNS / "mpce_3dkd_product_single_refit_v1" / "prototype_profile_map.json"


def _profile_text(cys_count: float, net_charge: float, hydrophobic_fraction: float) -> str:
    parts = []
    if cys_count < 1.5:
        parts.append("low Cys count")
    elif cys_count <= 3.0:
        parts.append("moderate Cys count")
    else:
        parts.append("high Cys count")
    if net_charge <= 0.5:
        parts.append("low/negative net charge")
    elif net_charge <= 1.5:
        parts.append("moderate net charge")
    else:
        parts.append("high net charge")
    if hydrophobic_fraction <= 0.30:
        parts.append("lower hydrophobicity")
    elif hydrophobic_fraction <= 0.38:
        parts.append("moderate hydrophobicity")
    else:
        parts.append("higher hydrophobicity")
    return "model-associated profile: " + " / ".join(parts) + " (not an experimental toxicity cause)"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    records = cache["records"]
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    manifest = load_ensemble_manifest(MANIFEST)
    checkpoint_paths = {int(item["seed"]): Path(item["path"]) for item in manifest["checkpoints"]}

    global_phys_mean = phys.mean(dim=0).numpy()
    per_seed = {}
    for seed in (20260905, 20260906):
        model, tri_mean, tri_std, phys_mean, phys_std, metadata = load_product_checkpoint(checkpoint_paths[seed], device)
        with torch.inference_mode():
            normalized_tri = (tri.to(device) - tri_mean) / tri_std
            normalized_phys = (phys.to(device) - phys_mean) / phys_std
            z_final, z_global, z_proto, cosine = model(normalized_tri, normalized_phys)
            cosine_np = cosine.cpu().numpy()
        global_proto = cosine_np.argmax(axis=1)
        usage = {
            int(i): int((global_proto == i).sum())
            for i in range(2 * PROTOTYPES_PER_CLASS)
        }
        profiles = {}
        for prototype in range(2 * PROTOTYPES_PER_CLASS):
            selected = global_proto == prototype
            if not bool(selected.any()):
                profiles[prototype] = None
                continue
            phys_mean_selected = phys[selected].mean(dim=0).numpy()
            similarities = cosine_np[selected, prototype]
            profiles[prototype] = {
                "usage_count": int(selected.sum()),
                "physchem_mean": {
                    str(name): float(value)
                    for name, value in zip(PHYS_CHEM_FEATURE_NAMES, phys_mean_selected, strict=True)
                },
                "similarity_stats": {
                    "mean": float(np.mean(similarities)),
                    "q25": float(np.percentile(similarities, 25)),
                    "q50": float(np.percentile(similarities, 50)),
                    "q75": float(np.percentile(similarities, 75)),
                },
                "model_associated_pattern": _profile_text(
                    float(phys_mean_selected[6]),
                    float(phys_mean_selected[1]),
                    float(phys_mean_selected[4]),
                ),
            }
        per_seed[str(seed)] = {
            "usage_count": usage,
            "assignment": "global closest prototype across all 8 prototypes",
            "prototypes": profiles,
        }

    payload = {
        "artifact_type": "mpce_3dkd_prototype_profile_map",
        "prototype_profile_version": "1.0",
        "model": "MPCE-3DKD-Product single-forward refit",
        "source": {"cache": str(CACHE), "manifest": str(MANIFEST)},
        "per_seed": per_seed,
        "global_physchem_mean": {
            str(name): float(value)
            for name, value in zip(PHYS_CHEM_FEATURE_NAMES, global_phys_mean, strict=True)
        },
        "note": "Descriptions are model-associated prototype summaries, not toxicity causes or decision rules.",
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["prototype_profile_sha256"] = file_sha256(OUTPUT)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
