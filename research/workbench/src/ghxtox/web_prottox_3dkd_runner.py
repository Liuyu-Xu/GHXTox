"""JSON-stdin web runner for the frozen ProtTox-3DKD deployment ensemble."""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from ghxtox.local_web_api import _validate_sequence
from ghxtox.prottox_3dkd_predict import MODEL_NAME, MODEL_VERSION, predict_sequences


def run(request: dict[str, Any]) -> dict[str, Any]:
    sequence = _validate_sequence(str(request["sequence"]))
    manifest = os.environ.get("GHXTOX_PROTTOX_3DKD_MANIFEST", "").strip()
    if not manifest:
        raise RuntimeError("Set GHXTOX_PROTTOX_3DKD_MANIFEST to the frozen ensemble_manifest.json before web prediction.")
    result = predict_sequences(
        [sequence],
        manifest,
        model_path=os.environ.get("GHXTOX_PROTTOX_3DKD_MODEL_PATH") or None,
        device=os.environ.get("GHXTOX_PROTTOX_3DKD_DEVICE", "cuda"),
    )[0]
    probability = float(result["probability"])
    margin = abs(probability - 0.5)
    confidence = "High" if margin >= 0.25 else "Moderate" if margin >= 0.10 else "Low"
    phys = result["physchem"]
    return {
        "id": f"prottox_3dkd_{uuid.uuid4().hex}",
        "name": str(request.get("name") or "sequence"),
        "sequence": sequence,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "modelVersion": f"{MODEL_NAME} {MODEL_VERSION} · frozen dual-seed MLP ensemble; 3D expert is training-only",
        "probability": probability,
        "decisionThreshold": 0.5,
        "label": result["label"],
        "binaryLabel": result["label"],
        "confidence": confidence,
        "route": "final",
        "analysisLevel": "final",
        "routeReason": "Frozen ProtTox-3DKD inference: ProtT5 -> TriStat + PhysChem -> two-seed MLP probability mean.",
        "sequenceEvidence": {
            "basicFraction": phys["basicFraction"],
            "hydrophobicFraction": phys["hydrophobicFraction"],
            "aromaticFraction": phys["aromaticFraction"],
            "cysteineCount": phys["cysteineCount"],
            "netCharge": phys["netCharge"],
            "flags": ["Frozen ProtT5 residue embeddings", "TriStat pooling", "Fixed physicochemical inputs"],
        },
        "structureEvidence": {
            "available": False,
            "meanPlddt": None,
            "confidenceBand": "Not used",
            "localGeometry": "No structure is generated at deployment.",
            "chemicalSites": "Not part of ProtTox-3DKD.",
            "cysteineTopology": "Only fixed Cys descriptors are used.",
            "residueConfidence": [],
        },
        "reviewNotes": [
            "Research prediction only; not an experimental toxicity conclusion.",
            "Interpret low-confidence predictions cautiously and validate candidates experimentally.",
        ],
        "isMock": False,
        "recommendFullAnalysis": False,
        "provenance": {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "ensemble": result["provenance"],
        },
    }


def main() -> None:
    try:
        print(json.dumps(run(json.load(sys.stdin)), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
