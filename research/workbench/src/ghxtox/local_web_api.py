"""Private localhost API boundary for the GHXTox research web interface.

The endpoint is deliberately conservative: it returns a prediction only when
an operator has configured a complete, canonical inference runner.  Merely
finding checkpoint files is insufficient because the public default also
requires fresh structure/feature preparation for the submitted sequence.
This prevents a partial 1D branch or synthetic UI value from being reported as
the audited multidimensional default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from threading import Lock
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ProtT5 Quick inference is memory-heavy; serialize local requests to protect the GPU.
QUICK_RUN_LOCK = Lock()
PROTTOX_TSK_RUN_LOCK = Lock()
PROTTOX_3DKD_RUN_LOCK = Lock()
MPCE_3DKD_RUN_LOCK = Lock()


CANONICAL_MODEL = "GHXTox-Multidimensional-Fusion"
CANONICAL_THRESHOLD = 0.47022256531284556
CANONICAL_WEIGHT_1D = 0.72
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _asset_root() -> Path:
    """Return the operator supplied asset root, or the local recovered bundle."""

    return Path(os.environ.get("GHXTOX_WEB_ASSET_ROOT", r"D:\\dachuang2026\\ziliao"))


def _full_runner_command() -> str | None:
    """Configured complete fusion runner; legacy variable remains supported."""

    value = os.environ.get("GHXTOX_WEB_FULL_RUNNER", "").strip() or os.environ.get("GHXTOX_WEB_RUNNER", "").strip()
    return value or None


def _quick_runner_command() -> str | None:
    """Configured sequence-only runner; it must not fall back to a mock response."""

    value = os.environ.get("GHXTOX_WEB_QUICK_RUNNER", "").strip()
    return value or None


def _prottox_tsk_runner_command() -> str | None:
    """Configured one-stage independent-model runner; never falls back to GHXTox."""

    value = os.environ.get("GHXTOX_PROTTOX_TSK_RUNNER", "").strip()
    return value or None


def _prottox_3dkd_runner_command() -> str | None:
    """Configured frozen ProtTox-3DKD deployment runner (dual-seed MLP ensemble)."""

    value = os.environ.get("GHXTOX_PROTTOX_3DKD_RUNNER", "").strip()
    return value or None


def _prottox_3dkd_batch_runner_command() -> str | None:
    """Configured batch runner for ProtTox-3DKD v1."""

    value = os.environ.get("GHXTOX_PROTTOX_3DKD_BATCH_RUNNER", "").strip()
    return value or None


def _mpce_3dkd_runner_command() -> str | None:
    """Configured MPCE-3DKD evidence runner (single detailed / batch base)."""

    value = os.environ.get("GHXTOX_MPCE_3DKD_RUNNER", "").strip()
    return value or None


def _runner_command() -> str | None:
    """Backward-compatible alias for the complete fusion runner."""

    return _full_runner_command()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetStatus:
    role: str
    path: str
    present: bool
    sha256: str | None


def canonical_assets(asset_root: Path | None = None) -> list[AssetStatus]:
    root = asset_root or _asset_root()
    entries = (
        ("1d_seed_42", root / "full_seed42.pt"),
        ("1d_seed_123", root / "full_seed123.pt"),
        ("1d_seed_2025", root / "full_seed2025.pt"),
        ("3d_expert", root / "best_model.pt"),
        ("1d_config", _repo_root() / "configs" / "lora_mamba_1d_autodl.json"),
        ("oof_seed_42", _repo_root() / "runs" / "lora_mamba_full" / "oof_seed42.json"),
        ("oof_seed_123", _repo_root() / "runs" / "lora_mamba_full" / "oof_seed123.json"),
        ("oof_seed_2025", _repo_root() / "runs" / "lora_mamba_full" / "oof_seed2025.json"),
        ("fold_manifest", Path(r"D:\\dachuang2026\\ziliao\\train_cdhit080_fallback_folds.csv")),
    )
    return [AssetStatus(role, str(path), path.is_file(), _sha256(path)) for role, path in entries]


def quick_assets(asset_root: Path | None = None) -> list[AssetStatus]:
    root = asset_root or _asset_root()
    repo = _repo_root()
    entries = (
        ("1d_seed_42", root / "full_seed42.pt"),
        ("1d_seed_123", root / "full_seed123.pt"),
        ("1d_seed_2025", root / "full_seed2025.pt"),
        ("1d_config", repo / "configs" / "lora_mamba_1d_autodl.json"),
        ("quick_policy", repo / "configs" / "web_quick_sequence_policy_v1.json"),
        ("oof_seed_42", repo / "runs" / "lora_mamba_full" / "oof_seed42.json"),
        ("oof_seed_123", repo / "runs" / "lora_mamba_full" / "oof_seed123.json"),
        ("oof_seed_2025", repo / "runs" / "lora_mamba_full" / "oof_seed2025.json"),
        ("fold_manifest", root / "train_cdhit080_fallback_folds.csv"),
    )
    return [AssetStatus(role, str(path), path.is_file(), _sha256(path)) for role, path in entries]


def quick_runtime_status(asset_root: Path | None = None, runner: str | None = None) -> dict[str, Any]:
    assets = quick_assets(asset_root)
    missing = [entry.role for entry in assets if not entry.present]
    command = runner if runner is not None else _quick_runner_command()
    if missing:
        state, reason = "unavailable", f"Missing Quick Prediction assets: {', '.join(missing)}."
    elif not command:
        state, reason = "configuration_required", "Quick assets are present, but no sequence-only Quick runner is configured."
    else:
        state, reason = "ready", "Frozen sequence-only Quick runner is configured."
    return {"service": "GHXTox private local API", "state": state, "reason": reason, "model": {"name": "GHXTox Sequence-only Quick", "sequence_only": True, "full_fusion_threshold_reused": False}, "assets": [asdict(asset) for asset in assets], "runner_configured": bool(command)}


def prottox_tsk_runtime_status(runner: str | None = None) -> dict[str, Any]:
    """Status for the isolated ProtTox-TSK route, without legacy assets."""

    checkpoint_text = os.environ.get("GHXTOX_PROTTOX_TSK_CHECKPOINT", "").strip()
    checkpoint = Path(checkpoint_text) if checkpoint_text else None
    command = runner if runner is not None else _prottox_tsk_runner_command()
    assets = [] if checkpoint is None else [AssetStatus("prottox_tsk_final_checkpoint", str(checkpoint), checkpoint.is_file(), _sha256(checkpoint))]
    if checkpoint is None:
        state, reason = "configuration_required", "Set GHXTOX_PROTTOX_TSK_CHECKPOINT after training a new ProtTox-TSK head. Historical GHXTox checkpoints are not accepted."
    elif not checkpoint.is_file():
        state, reason = "unavailable", "Configured ProtTox-TSK checkpoint does not exist."
    elif not command:
        state, reason = "configuration_required", "ProtTox-TSK checkpoint is present, but no one-stage runner is configured."
    else:
        state, reason = "ready", "Independent one-stage ProtTox-TSK runner is configured."
    return {"service": "GHXTox private local API", "state": state, "reason": reason, "model": {"name": "ProtTox-TSK", "legacy_ghxtox_components_enabled": False, "quick_full_workflow_enabled": False, "three_d_enabled": False}, "assets": [asdict(asset) for asset in assets], "runner_configured": bool(command)}


def prottox_3dkd_runtime_status(runner: str | None = None) -> dict[str, Any]:
    """Status for the frozen ProtTox-3DKD deployment route."""

    manifest_text = os.environ.get("GHXTOX_PROTTOX_3DKD_MANIFEST", "").strip()
    manifest = Path(manifest_text) if manifest_text else None
    command = runner if runner is not None else _prottox_3dkd_runner_command()
    assets = [] if manifest is None else [AssetStatus("prottox_3dkd_ensemble_manifest", str(manifest), manifest.is_file(), _sha256(manifest))]
    if manifest is None:
        state, reason = "configuration_required", "Set GHXTOX_PROTTOX_3DKD_MANIFEST to the frozen ProtTox-3DKD ensemble_manifest.json."
    elif not manifest.is_file():
        state, reason = "unavailable", "Configured ProtTox-3DKD ensemble manifest does not exist."
    elif not command:
        state, reason = "configuration_required", "ProtTox-3DKD manifest is present, but no deployment runner is configured."
    else:
        state, reason = "ready", "Frozen ProtTox-3DKD deployment runner is configured."
    return {
        "service": "GHXTox private local API",
        "state": state,
        "reason": reason,
        "model": {
            "name": "ProtTox-3DKD",
            "version": "v1.0",
            "ensemble": "20260905 + 20260906 mean probability",
            "deployment_3d_enabled": False,
            "quick_full_workflow_enabled": False,
        },
        "assets": [asdict(asset) for asset in assets],
        "runner_configured": bool(command),
    }


def mpce_3dkd_runtime_status(runner: str | None = None) -> dict[str, Any]:
    """Status for the MPCE-3DKD evidence deployment route."""

    manifest_text = os.environ.get("GHXTOX_MPCE_3DKD_MANIFEST", "").strip()
    manifest = Path(manifest_text) if manifest_text else None
    command = runner if runner is not None else _mpce_3dkd_runner_command()
    assets = [] if manifest is None else [AssetStatus("mpce_3dkd_ensemble_manifest", str(manifest), manifest.is_file(), _sha256(manifest))]
    if manifest is None:
        state, reason = "configuration_required", "Set GHXTOX_MPCE_3DKD_MANIFEST to the MPCE-3DKD single-forward ensemble_manifest.json."
    elif not manifest.is_file():
        state, reason = "unavailable", "Configured MPCE-3DKD ensemble manifest does not exist."
    elif not command:
        state, reason = "configuration_required", "MPCE-3DKD manifest is present, but no evidence runner is configured."
    else:
        state, reason = "ready", "Frozen MPCE-3DKD evidence runner is configured."
    return {
        "service": "GHXTox private local API",
        "state": state,
        "reason": reason,
        "model": {
            "name": "MPCE-3DKD",
            "version": "1.0",
            "ensemble": "20260905 + 20260906 mean probability",
            "deployment_3d_enabled": False,
            "evidence_schema_version": "1.0",
            "quick_full_workflow_enabled": False,
        },
        "assets": [asdict(asset) for asset in assets],
        "runner_configured": bool(command),
    }


def runtime_status(asset_root: Path | None = None, runner: str | None = None) -> dict[str, Any]:
    assets = canonical_assets(asset_root)
    missing = [entry.role for entry in assets if not entry.present]
    command = runner if runner is not None else _runner_command()
    if missing:
        state = "unavailable"
        reason = f"Missing canonical assets: {', '.join(missing)}."
    elif not command:
        state = "configuration_required"
        reason = (
            "Checkpoint assets are present, but no complete canonical runner is configured. "
            "A runner must prepare the submitted sequence's structure and feature caches before fusion."
        )
    else:
        state = "ready"
        reason = "Canonical runner is configured."
    return {
        "service": "GHXTox private local API",
        "state": state,
        "reason": reason,
        "model": {
            "name": CANONICAL_MODEL,
            "weight_1d": CANONICAL_WEIGHT_1D,
            "decision_threshold": CANONICAL_THRESHOLD,
            "soft_cys_default_enabled": False,
            "structure_utility_router_enabled": False,
        },
        "assets": [asdict(asset) for asset in assets],
        "runner_configured": bool(command),
    }


def _validate_sequence(sequence: str) -> str:
    cleaned = "".join(sequence.upper().split())
    if not 3 <= len(cleaned) <= 50:
        raise ValueError("Sequence length must be within the audited 3–50 aa dataset context.")
    invalid = sorted(set(cleaned) - STANDARD_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Sequence contains non-standard residues: {', '.join(invalid)}.")
    return cleaned


def _run_configured_runner(request: dict[str, Any], command: str | None, analysis: str) -> dict[str, Any]:
    if not command:
        raise RuntimeError(f"No {analysis} inference runner is configured.")
    completed = subprocess.run(
        command,
        shell=True,
        input=json.dumps(request),
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("GHXTOX_WEB_TIMEOUT_SECONDS", "1800")),
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Canonical runner failed (exit {completed.returncode}): {completed.stderr.strip()}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Canonical runner did not return a JSON prediction record.") from error
    if not isinstance(result, dict):
        raise RuntimeError("Canonical runner returned a non-object JSON value.")
    return result


def create_app():
    """Create the optional FastAPI app without making FastAPI a core dependency."""

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
    except ImportError as error:  # pragma: no cover - exercised by startup guidance
        raise RuntimeError("Install the API extra: pip install -e .[api]") from error

    class PredictionRequest(BaseModel):
        sequence: str = Field(min_length=3, max_length=50)
        name: str = Field(min_length=1, max_length=160)
        requested_route: str = "auto"

    class ApiPredictRequest(BaseModel):
        sequence: str = Field(min_length=1, max_length=50)

    class ApiEvidenceRequest(BaseModel):
        sequence: str = Field(min_length=1, max_length=50)
        analysis_level: str = "detailed"

    class ApiBatchItem(BaseModel):
        id: str = Field(min_length=1, max_length=160)
        sequence: str = Field(min_length=1, max_length=200)

    class ApiBatchRequest(BaseModel):
        sequences: list[ApiBatchItem] = Field(min_length=1, max_length=200)

    # FastAPI resolves postponed annotations via module globals.  The model is
    # deliberately imported late with FastAPI, so publish it before routes.
    globals()["PredictionRequest"] = PredictionRequest
    globals()["ApiPredictRequest"] = ApiPredictRequest
    globals()["ApiEvidenceRequest"] = ApiEvidenceRequest
    globals()["ApiBatchRequest"] = ApiBatchRequest

    app = FastAPI(title="GHXTox private local API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return {
            "mpce_3dkd": mpce_3dkd_runtime_status(),
            "prottox_3dkd": prottox_3dkd_runtime_status(),
            "prottox_tsk": prottox_tsk_runtime_status(),
            "quick": quick_runtime_status(),
            "full": runtime_status(),
        }

    def run_prediction(request: PredictionRequest, *, analysis: str) -> dict[str, Any]:
        try:
            sequence = _validate_sequence(request.sequence)
        except ValueError as error:
            raise HTTPException(status_code=422, detail={"code": "invalid_sequence", "message": str(error)}) from error
        if analysis == "quick":
            status, command = quick_runtime_status(), _quick_runner_command()
        elif analysis == "prottox-tsk":
            status, command = prottox_tsk_runtime_status(), _prottox_tsk_runner_command()
        elif analysis == "prottox-3dkd":
            status, command = prottox_3dkd_runtime_status(), _prottox_3dkd_runner_command()
        else:
            status, command = runtime_status(), _full_runner_command()
        if status["state"] != "ready":
            raise HTTPException(status_code=503, detail={"code": f"{analysis}_model_unavailable", "message": status["reason"], "status": status})
        try:
            if analysis == "quick":
                # The three-seed ProtT5 ensemble uses most of a laptop GPU.  Queue
                # simultaneous browser submissions instead of competing for VRAM.
                with QUICK_RUN_LOCK:
                    result = _run_configured_runner({"sequence": sequence, "name": request.name, "requested_route": "fast"}, command, analysis)
            elif analysis == "prottox-tsk":
                with PROTTOX_TSK_RUN_LOCK:
                    result = _run_configured_runner({"sequence": sequence, "name": request.name, "requested_route": "final"}, command, analysis)
            elif analysis == "prottox-3dkd":
                with PROTTOX_3DKD_RUN_LOCK:
                    result = _run_configured_runner({"sequence": sequence, "name": request.name, "requested_route": "final"}, command, analysis)
            else:
                result = _run_configured_runner({"sequence": sequence, "name": request.name, "requested_route": "structure"}, command, analysis)
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail={"code": f"{analysis}_runner_failed", "message": str(error)}) from error
        result.setdefault("isMock", False)
        result.setdefault("analysisLevel", analysis)
        result.setdefault("provenance", {"runtime": status, "requested_route": request.requested_route})
        return result

    @app.post("/v1/predictions/quick")
    @app.post("/predict/quick")
    def predict_quick(request: PredictionRequest) -> dict[str, Any]:
        return run_prediction(request, analysis="quick")

    @app.post("/v1/predictions/full")
    @app.post("/predict/full")
    def predict_full(request: PredictionRequest) -> dict[str, Any]:
        return run_prediction(request, analysis="full")

    @app.post("/v1/predictions/prottox-tsk")
    @app.post("/predict/prottox-tsk")
    def predict_prottox_tsk(request: PredictionRequest) -> dict[str, Any]:
        """One-stage independent model endpoint; no old GHXTox path is invoked."""
        return run_prediction(request, analysis="prottox-tsk")

    @app.post("/v1/predictions/prottox-3dkd")
    @app.post("/predict/prottox-3dkd")
    def predict_prottox_3dkd(request: PredictionRequest) -> dict[str, Any]:
        """Frozen deployment model endpoint: ProtTox-3DKD dual-seed MLP ensemble."""
        return run_prediction(request, analysis="prottox-3dkd")

    @app.post("/api/predict")
    def api_predict(payload: ApiPredictRequest) -> dict[str, Any]:
        """Minimal user-facing endpoint for frozen ProtTox-3DKD v1.

        The response intentionally does not expose teacher/KD/outer-fold or
        checkpoint provenance details to ordinary users.
        """
        try:
            sequence = _validate_sequence(payload.sequence)
        except ValueError as error:
            raise HTTPException(status_code=422, detail={"code": "invalid_sequence", "message": str(error)}) from error
        status = prottox_3dkd_runtime_status()
        command = _prottox_3dkd_runner_command()
        if status["state"] != "ready":
            raise HTTPException(status_code=503, detail={"code": "prottox-3dkd_unavailable", "message": status["reason"]})
        try:
            with PROTTOX_3DKD_RUN_LOCK:
                result = _run_configured_runner(
                    {"sequence": sequence, "name": "sequence", "requested_route": "final"},
                    command,
                    "prottox-3dkd",
                )
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail={"code": "prottox-3dkd_runner_failed", "message": str(error)}) from error
        probability = float(result["probability"])
        margin = abs(probability - 0.5)
        confidence = "High" if margin >= 0.25 else "Moderate" if margin >= 0.10 else "Low"
        evidence = result.get("sequenceEvidence", {})
        return {
            "toxicity_probability": probability,
            "prediction": result["label"],
            "confidence": confidence,
            "sequence_length": len(sequence),
            "model_version": "ProtTox-3DKD-v1",
            "sequence_evidence": {
                "net_charge_ph7": evidence.get("netCharge"),
                "basic_fraction": evidence.get("basicFraction"),
                "hydrophobic_fraction": evidence.get("hydrophobicFraction"),
                "aromatic_fraction": evidence.get("aromaticFraction"),
                "cysteine_count": evidence.get("cysteineCount"),
            },
            "warnings": [],
        }

    @app.post("/api/predict/batch")
    def api_predict_batch(payload: ApiBatchRequest) -> dict[str, Any]:
        """Per-item batch prediction for frozen ProtTox-3DKD v1."""
        status = prottox_3dkd_runtime_status()
        command = _prottox_3dkd_batch_runner_command()
        if status["state"] != "ready":
            raise HTTPException(status_code=503, detail={"code": "prottox-3dkd_unavailable", "message": status["reason"]})
        if not command:
            raise HTTPException(status_code=503, detail={"code": "prottox_3dkd_batch_unavailable", "message": "Set GHXTOX_PROTTOX_3DKD_BATCH_RUNNER before batch prediction."})
        try:
            with PROTTOX_3DKD_RUN_LOCK:
                result = _run_configured_runner(
                    {"sequences": [{"id": item.id, "sequence": item.sequence} for item in payload.sequences]},
                    command,
                    "prottox-3dkd-batch",
                )
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail={"code": "prottox_3dkd_batch_runner_failed", "message": str(error)}) from error
        return result

    @app.post("/api/predict/mpce-3dkd")
    def api_predict_mpce_evidence(payload: ApiEvidenceRequest) -> dict[str, Any]:
        """MPCE-3DKD evidence endpoint; single detailed evidence by default."""
        if payload.analysis_level not in {"base", "detailed"}:
            raise HTTPException(status_code=422, detail={"code": "invalid_analysis_level", "message": "analysis_level must be 'base' or 'detailed'."})
        try:
            sequence = _validate_sequence(payload.sequence)
        except ValueError as error:
            raise HTTPException(status_code=422, detail={"code": "invalid_sequence", "message": str(error)}) from error
        status = mpce_3dkd_runtime_status()
        command = _mpce_3dkd_runner_command()
        if status["state"] != "ready":
            raise HTTPException(status_code=503, detail={"code": "mpce_3dkd_unavailable", "message": status["reason"]})
        try:
            with MPCE_3DKD_RUN_LOCK:
                result = _run_configured_runner(
                    {"sequence": sequence, "name": "sequence", "analysis_level": payload.analysis_level},
                    command,
                    "mpce-3dkd",
                )
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail={"code": "mpce_3dkd_runner_failed", "message": str(error)}) from error
        return result

    @app.post("/api/predict/mpce-3dkd/batch")
    def api_predict_mpce_evidence_batch(payload: ApiBatchRequest) -> dict[str, Any]:
        """MPCE-3DKD batch endpoint; base evidence only (no perturbation screen)."""
        status = mpce_3dkd_runtime_status()
        command = _mpce_3dkd_runner_command()
        if status["state"] != "ready":
            raise HTTPException(status_code=503, detail={"code": "mpce_3dkd_unavailable", "message": status["reason"]})
        try:
            with MPCE_3DKD_RUN_LOCK:
                result = _run_configured_runner(
                    {"sequences": [{"id": item.id, "sequence": item.sequence} for item in payload.sequences]},
                    command,
                    "mpce-3dkd-batch",
                )
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail={"code": "mpce_3dkd_batch_runner_failed", "message": str(error)}) from error
        return result

    @app.post("/v1/predictions")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        """New-project default: the independent one-stage ProtTox-TSK route."""
        return run_prediction(request, analysis="prottox-tsk")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("Install the API extra first: pip install -e .[api]") from error
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
