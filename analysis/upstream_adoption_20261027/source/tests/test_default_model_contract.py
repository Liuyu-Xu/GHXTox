from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ghxtox.multidimensional_default import (
    DEFAULT_STRUCTURE_CHECKPOINT,
    assemble,
    build_parser,
    validate_public_default_config,
)
from ghxtox.utils import (
    DEFAULT_MULTIDIMENSIONAL_THRESHOLD,
    DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D,
    DEFAULT_PUBLIC_MODEL_NAME,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    path = ROOT / "configs" / "default_multidimensional_fusion.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_public_configuration_matches_the_frozen_model() -> None:
    config = _load_config()
    validate_public_default_config(config)

    assert config["public_model_name"] == DEFAULT_PUBLIC_MODEL_NAME
    assert config["model_status"] == "primary"
    assert config["architecture_version"] == "no_inner_transformer_20261023"
    assert config["fusion"]["weight_sequence"] == DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D
    assert config["fusion"]["weight_structure"] == 0.24
    assert config["fusion"]["threshold"] == DEFAULT_MULTIDIMENSIONAL_THRESHOLD
    assert config["composition"]["sequence_submodel"]["seeds"] == [42, 123, 2025]
    assert (
        config["composition"]["structural_submodel"]["checkpoint"]
        == DEFAULT_STRUCTURE_CHECKPOINT
    )
    assert config["inference"]["structures_per_peptide"] == 1
    assert config["inference"]["multi_conformer_residual"] == "disabled"
    assert config["composition"]["structural_submodel"]["architecture"] == "no_inner_transformer"
    architecture = json.loads((ROOT / "configs/default.json").read_text(encoding="utf-8"))
    assert architecture["model"]["num_sequence_layers"] == 0
    assert architecture["model"]["num_egnn_layers"] == 3
    assert architecture["model"]["chemical_site_branch"] is True


def test_public_parser_exposes_only_supported_inference_inputs() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--processed",
            "structural.pt",
            "--sequence-records",
            "sequence.pt",
            "--output-csv",
            "predictions.csv",
        ]
    )
    assert args.command == "predict"
    assert not hasattr(args, "conformer_cache")
    assert not hasattr(args, "conformer_manifest")


def test_all_default_entrypoints_select_no_inner_transformer() -> None:
    from ghxtox.evaluate import build_arg_parser as evaluate_parser
    from ghxtox.predict import build_arg_parser as structure_parser
    from ghxtox.train import build_arg_parser as train_parser

    predict_args = build_parser().parse_args([
        "predict", "--processed", "structure.pt", "--sequence-records", "sequence.pt",
        "--output-csv", "predictions.csv",
    ])
    assert predict_args.structure_checkpoint == DEFAULT_STRUCTURE_CHECKPOINT
    assert structure_parser().parse_args([]).checkpoint == DEFAULT_STRUCTURE_CHECKPOINT
    assert evaluate_parser().parse_args([]).checkpoint == DEFAULT_STRUCTURE_CHECKPOINT
    assert train_parser().parse_args([]).config == "configs/default.json"
    training_config = json.loads((ROOT / "configs/default.json").read_text(encoding="utf-8"))
    assert training_config["model"]["num_sequence_layers"] == 0
    assert training_config["inference"]["default_checkpoint"] == DEFAULT_STRUCTURE_CHECKPOINT
    assert training_config["inference"]["fallback_checkpoint"] is None


def test_public_default_rejects_stale_structural_config() -> None:
    config = _load_config()
    config["composition"]["structural_submodel"]["config"] = "configs/default_3d_v1.json"
    with pytest.raises(ValueError, match="configs/default.json"):
        validate_public_default_config(config)


@pytest.mark.parametrize("seed", [42, 123, 2025])
def test_installed_public_sequence_weights_match_declared_config(seed) -> None:
    """When local weights exist, protect the real strict inference contract."""
    import torch
    from ghxtox.lora_mamba_1d import validate_public_checkpoint_payload

    checkpoint = ROOT / f"runs/lora_mamba_full/full_seed{seed}.pt"
    if not checkpoint.is_file():
        pytest.skip("Local deployment weights are not distributed with the source package")
    config = json.loads((ROOT / "configs/lora_mamba_1d_autodl.json").read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_public_checkpoint_payload(payload, config, expected_seed=seed, checkpoint_path=checkpoint)


def test_fixed_probability_fusion(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence.csv"
    structure_path = tmp_path / "structure.csv"
    output_path = tmp_path / "predictions.csv"

    _write_rows(
        sequence_path,
        ["sample_id", "label", "toxicity_probability"],
        [
            {"sample_id": "p1", "label": 1, "toxicity_probability": 0.8},
            {"sample_id": "p2", "label": 0, "toxicity_probability": 0.2},
        ],
    )
    _write_rows(
        structure_path,
        ["sample_id", "label", "structure_probability"],
        [
            {"sample_id": "p1", "label": 1, "structure_probability": 0.5},
            {"sample_id": "p2", "label": 0, "structure_probability": 0.4},
        ],
    )

    summary = assemble(
        DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D,
        DEFAULT_MULTIDIMENSIONAL_THRESHOLD,
        sequence_path,
        structure_path,
        output_path,
    )

    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed = np.asarray([float(row["toxicity_probability"]) for row in rows])
    expected = np.asarray([0.76 * 0.8 + 0.24 * 0.5, 0.76 * 0.2 + 0.24 * 0.4])

    assert np.allclose(observed, expected)
    assert summary["n"] == 2
