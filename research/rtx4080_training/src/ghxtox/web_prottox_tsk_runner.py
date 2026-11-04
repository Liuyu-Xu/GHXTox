"""JSON-stdin runner for a one-stage local ProtTox-TSK web prediction."""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ghxtox.local_web_api import _validate_sequence
from ghxtox.predict_prottox_tsk import predict_sequences
from ghxtox.prottox_tsk import MODEL_NAME, MODEL_VERSION, file_sha256


def run(request: dict[str, Any]) -> dict[str, Any]:
    sequence = _validate_sequence(str(request["sequence"]))
    checkpoint = os.environ.get("GHXTOX_PROTTOX_TSK_CHECKPOINT", "").strip()
    if not checkpoint:
        raise RuntimeError("Set GHXTOX_PROTTOX_TSK_CHECKPOINT to a trained final_model.pt before web prediction.")
    result = predict_sequences([sequence], checkpoint, model_path=os.environ.get("GHXTOX_PROTTOX_TSK_MODEL_PATH") or None, device=os.environ.get("GHXTOX_PROTTOX_TSK_DEVICE", "cuda"))[0]
    probability = float(result["probability"])
    margin = abs(probability - 0.5)
    confidence = "High" if margin >= 0.25 else "Moderate" if margin >= 0.10 else "Low"
    phys = result["physchem"]
    return {
        "id": f"prottox_tsk_{uuid.uuid4().hex}", "name": str(request.get("name") or "sequence"), "sequence": sequence,
        "createdAt": datetime.now(timezone.utc).isoformat(), "modelVersion": f"{MODEL_NAME} {MODEL_VERSION} · Frozen ProtT5 + TriStat + PhysChem + {result['provenance']['head_kind'].upper()}",
        "probability": probability, "decisionThreshold": 0.5, "label": result["label"], "binaryLabel": result["label"], "confidence": confidence,
        "route": "final", "analysisLevel": "final", "routeReason": "One-stage ProtTox-TSK inference; no GHXTox Quick/Full, 3D, or fusion workflow was run.",
        "sequenceEvidence": {"basicFraction": phys["basicFraction"], "hydrophobicFraction": phys["hydrophobicFraction"], "aromaticFraction": phys["aromaticFraction"], "cysteineCount": phys["cysteineCount"], "netCharge": phys["netCharge"], "flags": ["Frozen ProtT5 residue embeddings", "TriStat pooling", "Fixed physicochemical inputs"]},
        "structureEvidence": {"available": False, "meanPlddt": None, "confidenceBand": "Not used", "localGeometry": "Not part of ProtTox-TSK.", "chemicalSites": "Not part of ProtTox-TSK.", "cysteineTopology": "Only the fixed Cys descriptors were used.", "residueConfidence": []},
        "reviewNotes": ["This is a one-stage research prediction, not an experimental toxicity conclusion.", "Interpret low-confidence predictions cautiously and validate candidates experimentally."], "isMock": False, "recommendFullAnalysis": False,
        "provenance": {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "checkpoint": checkpoint, "checkpoint_sha256": file_sha256(checkpoint), **result["provenance"]},
    }


def main() -> None:
    try:
        print(json.dumps(run(json.load(sys.stdin)), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
