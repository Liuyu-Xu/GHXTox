from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

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
    assert config["fusion"]["weight_sequence"] == DEFAULT_MULTIDIMENSIONAL_WEIGHT_1D
    assert config["fusion"]["weight_structure"] == 0.28
    assert config["fusion"]["threshold"] == DEFAULT_MULTIDIMENSIONAL_THRESHOLD
    assert config["composition"]["sequence_submodel"]["seeds"] == [42, 123, 2025]
    assert (
        config["composition"]["structural_submodel"]["checkpoint"]
        == DEFAULT_STRUCTURE_CHECKPOINT
    )
    assert config["inference"]["structures_per_peptide"] == 1
    assert config["inference"]["multi_conformer_residual"] == "disabled"


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
    expected = np.asarray([0.72 * 0.8 + 0.28 * 0.5, 0.72 * 0.2 + 0.28 * 0.4])

    assert np.allclose(observed, expected)
    assert summary["n"] == 2
