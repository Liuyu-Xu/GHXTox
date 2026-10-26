"""Run one canonical GHXTox prediction for the private localhost web API.

The program receives one JSON object on standard input and emits one
``PredictionResult`` object on standard output.  It deliberately executes the
same complete sequence -> structure -> ESM2 feature -> 1D/3D fusion path as
the current default.  It is an operator command, not a public web service.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from ghxtox.esmfold_predict import generate_esmfold_structures
from ghxtox.local_web_api import CANONICAL_MODEL, CANONICAL_THRESHOLD, _asset_root, _validate_sequence
from ghxtox.multidimensional_default import predict
from ghxtox.plm_embed import attach_esm2_embeddings
from ghxtox.preprocess import preprocess_fasta


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_canonical_asset(source: Path, destination: Path) -> None:
    """Create a hard link once, refusing to replace a different local asset."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise RuntimeError(f"Canonical asset mismatch: {destination}")
        return
    os.link(source, destination)


def materialize_canonical_assets(asset_root: Path | None = None) -> dict[str, str]:
    """Bind recovered immutable assets to the canonical public-predictor paths."""

    root = _repo_root()
    source_root = asset_root or _asset_root()
    mappings = {
        source_root / "full_seed42.pt": root / "runs/lora_mamba_full/full_seed42.pt",
        source_root / "full_seed123.pt": root / "runs/lora_mamba_full/full_seed123.pt",
        source_root / "full_seed2025.pt": root / "runs/lora_mamba_full/full_seed2025.pt",
        source_root / "best_model.pt": root / "runs/multisource_1_30_full2795_rank_stable_candidate/best_model.pt",
        source_root / "train_cdhit080_fallback_folds.csv": root / "data/folds/train_cdhit080_fallback_folds.csv",
    }
    for source, destination in mappings.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required recovered asset is missing: {source}")
        _link_canonical_asset(source, destination)
    return {str(destination.relative_to(root)).replace("\\", "/"): _sha256(destination) for destination in mappings.values()}


def _sequence_evidence(sequence: str) -> dict[str, Any]:
    length = len(sequence)
    count = lambda letters: sum(residue in letters for residue in sequence)
    basic, acidic = count("KRH"), count("DE")
    cysteines = count("C")
    # Henderson–Hasselbalch estimate at pH 7.0, including N- and C-termini.
    protonated = lambda pka: 1 / (1 + 10 ** (7 - pka))
    deprotonated = lambda pka: 1 / (1 + 10 ** (pka - 7))
    net_charge = round(
        protonated(9.69) + count("K") * protonated(10.5) + count("R") * protonated(12.4) + count("H") * protonated(6.0)
        - deprotonated(2.34) - count("D") * deprotonated(3.86) - count("E") * deprotonated(4.25)
        - count("C") * deprotonated(8.33) - count("Y") * deprotonated(10.07),
        2,
    )
    flags: list[str] = []
    if basic / length >= 0.25:
        flags.append("basic-residue enriched")
    if count("AVILMFWY") / length >= 0.42:
        flags.append("hydrophobic-residue enriched")
    if cysteines >= 2:
        flags.append(f"{cysteines} cysteines: review flag only; topology branch is not enabled")
    if not flags:
        flags.append("no simple composition flag triggered")
    return {
        "basicFraction": basic / length,
        "hydrophobicFraction": count("AVILMFWY") / length,
        "aromaticFraction": count("FWY") / length,
        "cysteineCount": cysteines,
        "netCharge": net_charge,
        "flags": flags,
    }


def _confidence(probability: float, mean_plddt: float) -> str:
    margin = abs(probability - CANONICAL_THRESHOLD)
    if margin >= 0.27 and mean_plddt >= 75:
        return "High"
    if margin >= 0.09:
        return "Moderate"
    return "Review"


def _run(request: dict[str, Any]) -> dict[str, Any]:
    sequence = _validate_sequence(str(request["sequence"]))
    name = str(request.get("name") or "sequence")
    root = _repo_root()
    assets = materialize_canonical_assets()
    work_root = Path(os.environ.get("GHXTOX_WEB_WORK_DIR", root / "runs/web_local_jobs"))
    job_dir = work_root / f"job_{uuid.uuid4().hex}"
    job_dir.mkdir(parents=True, exist_ok=False)
    fasta = job_dir / "input.fasta"
    fasta.write_text(f">web_input\n{sequence}\n", encoding="utf-8")

    # ESMFold writes the exact cache consumed by preprocess_fasta.  A fresh
    # prediction must not fall back to heuristic coordinates.
    structure_cache = job_dir / "structures"
    generate_esmfold_structures(
        SimpleNamespace(
            fasta=str(fasta), pdb_dir=str(job_dir / "esmfold_pdb"), cache_dir=str(structure_cache),
            device=os.environ.get("GHXTOX_WEB_DEVICE", "cuda"), max_length=128,
            chunk_size=int(os.environ.get("GHXTOX_WEB_ESMFOLD_CHUNK", "64")),
            skip_existing=False, no_pdb=True, no_npz=False, on_error="error", on_mismatch="error", log_every=1,
        )
    )
    raw_processed = job_dir / "processed_raw.pt"
    records = preprocess_fasta(
        fasta, raw_processed, structure_mode="cached", structure_cache_dir=structure_cache,
        max_length=128, missing_structure="error",
    )
    if len(records) != 1 or records[0]["sequence"] != sequence:
        raise RuntimeError("Web preprocessing did not preserve the submitted sequence.")
    processed = job_dir / "processed_esm2.pt"
    attach_esm2_embeddings(
        input_path=raw_processed,
        output_path=processed,
        model_name="esm2_t33_650M_UR50D",
        device_name=os.environ.get("GHXTOX_WEB_DEVICE", "cuda"),
        batch_size=1,
        cache_dir=Path(os.environ.get("GHXTOX_WEB_ESM2_CACHE", root / "data/plm/esm2_t33_650M_UR50D")),
        skip_existing=True,
    )
    # The public predictor retains labels for offline metric bookkeeping.  The
    # web sample has no observed label, so use a neutral placeholder which is
    # never exposed as a metric or training target.
    adapter = job_dir / "sequence_records.pt"
    torch.save({"sample_ids": ["web_input_1"], "sequences": [sequence], "labels": [0]}, adapter)
    output_csv, output_metrics = job_dir / "prediction.csv", job_dir / "prediction_metrics.json"
    predict(
        processed_path=processed,
        sequence_records_path=adapter,
        config_1d="configs/lora_mamba_1d_autodl.json",
        checkpoints=["runs/lora_mamba_full/full_seed42.pt", "runs/lora_mamba_full/full_seed123.pt", "runs/lora_mamba_full/full_seed2025.pt"],
        oof_jsons=["runs/lora_mamba_full/oof_seed42.json", "runs/lora_mamba_full/oof_seed123.json", "runs/lora_mamba_full/oof_seed2025.json"],
        manifest="data/folds/train_cdhit080_fallback_folds.csv",
        structure_checkpoint="runs/multisource_1_30_full2795_rank_stable_candidate/best_model.pt",
        weight=0.72,
        threshold=CANONICAL_THRESHOLD,
        device_name=os.environ.get("GHXTOX_WEB_DEVICE", "cuda"),
        batch_size=1,
        output_csv=output_csv,
        output_metrics=output_metrics,
        selective_policy=root / "runs/selective_prediction_oof_20260809_v1/selected_policy.json",
    )
    with output_csv.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    processed_payload = torch.load(processed, map_location="cpu", weights_only=False)
    plddt = processed_payload["records"][0]["plddt"].float().tolist()
    mean_plddt = float(sum(plddt) / len(plddt))
    probability = float(row["toxicity_probability"])
    binary_label = "Toxic" if probability >= CANONICAL_THRESHOLD else "Non-toxic"
    abstained = str(row.get("abstained", "0")).strip() in {"1", "true", "True"}
    label = "Uncertain" if abstained else binary_label
    confidence = "Review" if abstained else _confidence(probability, mean_plddt)
    review = ["Research screening output; review alongside experimental context and a secondary method."]
    if abstained:
        review.append("The frozen selective-prediction policy rejected this full computational result as uncertain; experimental validation is recommended.")
    elif confidence == "Review":
        review.append("Prediction is near the frozen decision threshold and warrants expert review.")
    if sequence.count("C") >= 2:
        review.append("Multiple cysteines are a review flag only; the non-promoted Soft-Cys topology branch was not used.")
    return {
        "id": job_dir.name,
        "name": name,
        "sequence": sequence,
        "createdAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "modelVersion": f"{CANONICAL_MODEL} · w1D=0.72",
        "probability": probability,
        "pSeq": float(row["p_1d"]),
        "pStructure": float(row["p_3d"]),
        "pFinal": probability,
        "finalPrediction": label,
        "decisionThreshold": CANONICAL_THRESHOLD,
        "label": label,
        "binaryLabel": binary_label,
        "risk": "High" if probability >= 0.70 else "Intermediate" if label == "Toxic" else "Low",
        "confidence": confidence,
        "route": "structure",
        "analysisLevel": "full",
        "routeReason": "Canonical fixed 0.72/0.28 sequence/structure fusion was run; no adaptive route policy was applied.",
        "sequenceEvidence": _sequence_evidence(sequence),
        "structureEvidence": {
            "available": True,
            "meanPlddt": mean_plddt,
            "confidenceBand": "High" if mean_plddt >= 80 else "Moderate" if mean_plddt >= 70 else "Low",
            "localGeometry": "ESMFold-derived single structure processed by the canonical 3D expert.",
            "chemicalSites": "Canonical 3D chemical-site branch included in the expert checkpoint.",
            "cysteineTopology": "Not used: Soft-Cys topology did not pass frozen OOF promotion.",
            "residueConfidence": plddt,
        },
        "reviewNotes": review,
        "isMock": False,
        "selectivePrediction": {
            "uncertaintyScore": float(row["uncertainty_score"]),
            "abstained": abstained,
            "policy": "frozen margin-only selective policy · 90% OOF coverage",
        },
        "provenance": {
            "assets_sha256": assets,
            "job_dir": str(job_dir),
            "prediction_csv": str(output_csv),
            "metrics_json": str(output_metrics),
            "requested_route": request.get("requested_route", "auto"),
        },
    }


def main() -> None:
    try:
        request = json.load(sys.stdin)
        result = _run(request)
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
