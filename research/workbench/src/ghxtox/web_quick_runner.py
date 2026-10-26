"""Run the frozen sequence-only GHXTox Quick Prediction workflow.

Input is a JSON request on stdin and output is a browser PredictionResult on
stdout.  The runner uses the audited three-seed LoRA-ProtT5 + BiMamba ensemble
only.  It deliberately does not generate a structure or invoke the 3D expert.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ghxtox.local_web_api import _asset_root, _validate_sequence
from ghxtox.multidimensional_default import _logit, _predict_1d_seed
from ghxtox.lora_mamba_1d import PUBLIC_1D_SEEDS, load_test_records
from ghxtox.web_quick_policy import apply_policy, load_policy


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise RuntimeError(f"Quick asset mismatch: {destination}")
        return
    os.link(source, destination)


def materialize_quick_assets(asset_root: Path | None = None) -> dict[str, str]:
    root, source_root = _repo_root(), asset_root or _asset_root()
    mappings = {
        source_root / "full_seed42.pt": root / "runs/lora_mamba_full/full_seed42.pt",
        source_root / "full_seed123.pt": root / "runs/lora_mamba_full/full_seed123.pt",
        source_root / "full_seed2025.pt": root / "runs/lora_mamba_full/full_seed2025.pt",
        source_root / "train_cdhit080_fallback_folds.csv": root / "data/folds/train_cdhit080_fallback_folds.csv",
    }
    for source, destination in mappings.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required Quick asset is missing: {source}")
        _link(source, destination)
    return {str(destination.relative_to(root)).replace("\\", "/"): _sha256(destination) for destination in mappings.values()}


def _verify_policy_inputs(policy: dict[str, Any], root: Path, asset_root: Path) -> None:
    expected = {entry["sha256"] for entry in policy.get("inputs", [])}
    observed = {
        _sha256(asset_root / "train_cdhit080_fallback_folds.csv"),
        _sha256(root / "runs/lora_mamba_full/oof_seed42.json"),
        _sha256(root / "runs/lora_mamba_full/oof_seed123.json"),
        _sha256(root / "runs/lora_mamba_full/oof_seed2025.json"),
    }
    if expected != observed:
        raise RuntimeError("Quick policy inputs do not match the frozen OOF assets.")


def _evidence(sequence: str) -> dict[str, Any]:
    length = len(sequence)
    count = lambda letters: sum(residue in letters for residue in sequence)
    protonated = lambda pka: 1 / (1 + 10 ** (7 - pka))
    deprotonated = lambda pka: 1 / (1 + 10 ** (pka - 7))
    charge = round(protonated(9.69) + count("K") * protonated(10.5) + count("R") * protonated(12.4) + count("H") * protonated(6.0) - deprotonated(2.34) - count("D") * deprotonated(3.86) - count("E") * deprotonated(4.25) - count("C") * deprotonated(8.33) - count("Y") * deprotonated(10.07), 2)
    return {"basicFraction": count("KRH") / length, "hydrophobicFraction": count("AVILMFWY") / length, "aromaticFraction": count("FWY") / length, "cysteineCount": count("C"), "netCharge": charge, "flags": ["sequence-only Quick Prediction"]}


def _run(request: dict[str, Any]) -> dict[str, Any]:
    sequence = _validate_sequence(str(request["sequence"]))
    name, root = str(request.get("name") or "sequence"), _repo_root()
    asset_root = _asset_root()
    assets = materialize_quick_assets(asset_root)
    policy_path = root / "configs/web_quick_sequence_policy_v1.json"
    policy = load_policy(policy_path)
    _verify_policy_inputs(policy, root, asset_root)
    config = json.loads((root / "configs/lora_mamba_1d_autodl.json").read_text(encoding="utf-8-sig"))
    work_root = Path(os.environ.get("GHXTOX_WEB_WORK_DIR", root / "runs/web_local_jobs"))
    job_dir = work_root / f"quick_{uuid.uuid4().hex}"
    job_dir.mkdir(parents=True, exist_ok=False)
    adapter = job_dir / "sequence_records.pt"
    torch.save({"sample_ids": ["web_quick_1"], "sequences": [sequence], "labels": [0]}, adapter)
    records = load_test_records(adapter)
    from transformers import T5Tokenizer
    device = torch.device(os.environ.get("GHXTOX_WEB_DEVICE", "cuda"))
    tokenizer = T5Tokenizer.from_pretrained(str(config["plm"]), do_lower_case=False)
    checkpoints = [root / f"runs/lora_mamba_full/full_seed{seed}.pt" for seed in PUBLIC_1D_SEEDS]
    seed_probabilities = [_predict_1d_seed(records, checkpoint, config, device, tokenizer, seed) for checkpoint, seed in zip(checkpoints, PUBLIC_1D_SEEDS)]
    mean_probability = float(np.mean(np.stack(seed_probabilities), axis=0)[0])
    calibration = policy["model"]["calibration"]
    probability = float(1 / (1 + np.exp(-(float(calibration["slope"]) * _logit([mean_probability])[0] + float(calibration["intercept"])))))
    decision = apply_policy(probability, policy)
    review = ["Quick Prediction is a sequence-only computational screen."]
    if decision["abstained"]:
        review.append("The frozen sequence OOF reject band classified this quick result as uncertain; run Full GHXTox Analysis.")
    else:
        review.append("Full GHXTox Analysis remains available when structural context is needed; experimental evidence is required for toxicity confirmation.")
    return {"id": job_dir.name, "name": name, "sequence": sequence, "createdAt": datetime.now(timezone.utc).isoformat(), "modelVersion": "GHXTox Sequence-only Quick v1 · three-seed LoRA-ProtT5 + BiMamba", "probability": probability, "pSeq": probability, "preliminaryPrediction": decision["label"], "decisionThreshold": float(policy["decision"]["threshold"]), "label": decision["label"], "binaryLabel": decision["binary_label"], "risk": "Intermediate", "confidence": decision["confidence"], "route": "fast", "analysisLevel": "quick", "routeReason": "Frozen sequence-only three-seed 1D workflow; no structure was generated.", "sequenceEvidence": _evidence(sequence), "structureEvidence": {"available": False, "meanPlddt": None, "confidenceBand": "Not requested", "localGeometry": "Not included in Quick Prediction.", "chemicalSites": "Not included in Quick Prediction.", "cysteineTopology": "Cys count is reported, but topology requires Full Analysis.", "residueConfidence": []}, "reviewNotes": review, "isMock": False, "recommendFullAnalysis": bool(decision["abstained"]), "provenance": {"assets_sha256": assets, "quick_policy": str(policy_path), "quick_policy_sha256": _sha256(policy_path), "quick_margin": decision["margin"], "requested_route": request.get("requested_route", "fast")}}


def main() -> None:
    try:
        print(json.dumps(_run(json.load(sys.stdin)), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
