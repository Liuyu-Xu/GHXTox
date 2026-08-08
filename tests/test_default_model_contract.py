from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pytest

from ghxtox._legacy.hybrid_default import validate_legacy_hybrid_config
from ghxtox.multidimensional_default import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_MANIFEST,
    DEFAULT_OOF_JSONS,
    DEFAULT_STRUCTURE_CHECKPOINT,
    DEFAULT_1D_CONFIG,
    _validate_canonical_predict_assets,
    assemble,
    build_parser,
    validate_public_default_config,
)
from ghxtox.utils import (
    DEFAULT_LEGACY_HYBRID_CONFIG,
    DEFAULT_MULTIDIMENSIONAL_CONFIG,
    DEFAULT_PUBLIC_CONFIG,
    DEFAULT_PUBLIC_MODEL_NAME,
)
from scripts.refreeze_lora_mamba_fusion import load_prob_by_sample_id


ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_multidimensional_model_is_the_only_public_default() -> None:
    current = _load("configs/default_multidimensional_fusion.json")
    validate_public_default_config(current)
    assert DEFAULT_PUBLIC_CONFIG == DEFAULT_MULTIDIMENSIONAL_CONFIG
    assert current["public_model_name"] == DEFAULT_PUBLIC_MODEL_NAME
    marked = []
    for path in (ROOT / "configs").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("canonical_public_default") is True:
            marked.append(path.name)
    assert marked == ["default_multidimensional_fusion.json"]
    structure_source = current["composition"]["3d_expert"]["test_probability_column"]
    assert "default_multidimensional_fusion_multiconformer_extension" in structure_source
    assert "default_3d_hybrid" not in structure_source


def test_3d_training_config_points_to_current_default() -> None:
    expert = _load("configs/default.json")
    pointer = expert["public_default"]
    assert pointer == {
        "config": "configs/default_multidimensional_fusion.json",
        "model_name": "GHXTox-Multidimensional-Fusion",
        "entrypoint": "ghxtox-predict",
        "role": "canonical public default; this file is only the 3D expert training config consumed by it",
    }


def test_previous_hybrid_is_explicitly_legacy() -> None:
    legacy = _load(DEFAULT_LEGACY_HYBRID_CONFIG)
    validate_legacy_hybrid_config(legacy)
    assert legacy["canonical_public_default"] is False
    assert legacy["status"] == "legacy_baseline_only"


def test_current_default_does_not_import_legacy_pipeline() -> None:
    source = (ROOT / "src/ghxtox/multidimensional_default.py").read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "ghxtox.structure_inference" in imported
    assert "ghxtox.hybrid_default" not in imported
    assert not any(module.startswith("ghxtox._legacy") for module in imported)


def test_public_runtime_transitively_excludes_legacy_modules() -> None:
    package_root = ROOT / "src/ghxtox"
    pending = ["ghxtox.multidimensional_default"]
    visited: set[str] = set()
    legacy_edges: list[tuple[str, str]] = []
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        relative = module.split(".")[1:]
        path = package_root.joinpath(*relative).with_suffix(".py")
        if not path.exists():
            path = package_root.joinpath(*relative, "__init__.py")
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("ghxtox"):
                    imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(
                    alias.name for alias in node.names if alias.name.startswith("ghxtox")
                )
        for dependency in imported:
            if dependency.startswith("ghxtox._legacy") or dependency == "ghxtox.hybrid_default":
                legacy_edges.append((module, dependency))
            pending.append(dependency)
    assert legacy_edges == []


def test_full_runner_uses_only_pure_3d_oof() -> None:
    script = (ROOT / "scripts/run_lora_mamba_full.sh").read_text(encoding="utf-8")
    assert "runs/conformer_ensemble_residual_oof/full_calibrated_predictions.csv" in script
    assert "runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv" not in script


def test_public_exporter_uses_retained_multiconformer_structure_outputs() -> None:
    script = (ROOT / "scripts/export_multidimensional_default.py").read_text(
        encoding="utf-8"
    )
    assert "default_multidimensional_fusion_multiconformer_extension" in script
    assert 'f"{dataset}_structure_predictions.csv"' in script
    assert "runs/default_3d_hybrid" not in script


def test_console_entrypoints_are_explicit_and_compatible() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ghxtox-predict = "ghxtox.multidimensional_default:main"' in project
    assert 'ghxtox-multidimensional-predict = "ghxtox.multidimensional_default:main"' in project
    assert 'ghxtox-structure-predict = "ghxtox.predict:main"' in project
    assert "ghxtox-prott5-predict" not in project
    assert "ghxtox-hybrid-predict" not in project
    scripts = project.split("[project.scripts]", 1)[1]
    assert "ghxtox._legacy" not in scripts


def test_previous_default_code_is_physically_isolated() -> None:
    assert not (ROOT / "src/ghxtox/hybrid_default.py").exists()
    isolated = ROOT / "src/ghxtox/_legacy"
    assert (isolated / "hybrid_default.py").is_file()
    assert (isolated / "prott5_default.py").is_file()
    assert (isolated / "plm_textcnn.py").is_file()
    assert (isolated / "cross_modal_adapter.py").is_file()


def test_generated_install_metadata_is_ignored_and_not_authoritative() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "*.egg-info/" in ignore
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ghxtox-predict = "ghxtox.multidimensional_default:main"' in project


def test_public_predict_parser_exposes_multiconformer_extension() -> None:
    args = build_parser().parse_args([
        "predict",
        "--processed", "base.pt",
        "--sequence-records", "sequence.pt",
        "--conformer-manifest", "pepflow.csv", "af2.csv",
        "--conformer-cache-root", "pepflow_cache", "af2_cache",
        "--conformer-match-by-sequence",
        "--allow-unused-conformer-records",
        "--conformer-processed-output", "augmented.pt",
        "--output-csv", "predictions.csv",
    ])
    assert args.conformer_manifest == ["pepflow.csv", "af2.csv"]
    assert args.conformer_cache_root == ["pepflow_cache", "af2_cache"]
    assert args.conformer_match_by_sequence is True
    assert args.allow_unused_conformer_records is True
    assert args.conformer_processed_output == "augmented.pt"


def test_public_predict_assets_are_pinned_to_current_generation() -> None:
    _validate_canonical_predict_assets(
        DEFAULT_1D_CONFIG,
        DEFAULT_CHECKPOINTS,
        DEFAULT_OOF_JSONS,
        DEFAULT_MANIFEST,
        DEFAULT_STRUCTURE_CHECKPOINT,
    )
    with pytest.raises(ValueError, match="canonical 3D checkpoint"):
        _validate_canonical_predict_assets(
            DEFAULT_1D_CONFIG,
            DEFAULT_CHECKPOINTS,
            DEFAULT_OOF_JSONS,
            DEFAULT_MANIFEST,
            "runs/legacy/best_model.pt",
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_assemble_rejects_partial_intersection_and_label_mismatch(tmp_path: Path) -> None:
    one_d = tmp_path / "one_d.csv"
    three_d = tmp_path / "three_d.csv"
    output = tmp_path / "output.csv"
    _write_csv(one_d, [
        {"sample_id": "a", "sequence": "AAA", "label": 0, "toxicity_probability": 0.1},
        {"sample_id": "b", "sequence": "BBB", "label": 1, "toxicity_probability": 0.9},
    ])
    _write_csv(three_d, [
        {"sample_id": "a", "sequence": "AAA", "label": 0, "structure_probability": 0.2},
    ])
    with pytest.raises(ValueError, match="coverage differs"):
        assemble(0.72, 0.47022256531284556, one_d, three_d, output)

    _write_csv(three_d, [
        {"sample_id": "a", "sequence": "AAA", "label": 1, "structure_probability": 0.2},
        {"sample_id": "b", "sequence": "BBB", "label": 1, "structure_probability": 0.8},
    ])
    with pytest.raises(ValueError, match="label mismatch"):
        assemble(0.72, 0.47022256531284556, one_d, three_d, output)


def test_public_assembler_rejects_previous_hybrid_container(tmp_path: Path) -> None:
    one_d = tmp_path / "one_d.csv"
    three_d = tmp_path / "three_d.csv"
    output = tmp_path / "output.csv"
    _write_csv(one_d, [
        {"sample_id": "a", "sequence": "AAA", "label": 0, "toxicity_probability": 0.1},
        {"sample_id": "b", "sequence": "BBB", "label": 1, "toxicity_probability": 0.9},
    ])
    _write_csv(three_d, [
        {
            "sample_id": "a", "sequence": "AAA", "label": 0,
            "structure_probability": 0.2, "prott5_probability": 0.3,
            "interaction_delta": 0.0,
        },
        {
            "sample_id": "b", "sequence": "BBB", "label": 1,
            "structure_probability": 0.8, "prott5_probability": 0.7,
            "interaction_delta": 0.0,
        },
    ])
    with pytest.raises(ValueError, match="previous-hybrid model columns"):
        assemble(0.72, 0.47022256531284556, one_d, three_d, output)


def test_canonical_entrypoint_rejects_old_operating_point(tmp_path: Path) -> None:
    one_d = tmp_path / "one_d.csv"
    three_d = tmp_path / "three_d.csv"
    output = tmp_path / "output.csv"
    _write_csv(one_d, [
        {"sample_id": "a", "sequence": "AAA", "label": 0, "toxicity_probability": 0.1},
        {"sample_id": "b", "sequence": "BBB", "label": 1, "toxicity_probability": 0.9},
    ])
    _write_csv(three_d, [
        {"sample_id": "a", "sequence": "AAA", "label": 0, "structure_probability": 0.2},
        {"sample_id": "b", "sequence": "BBB", "label": 1, "structure_probability": 0.8},
    ])
    with pytest.raises(ValueError, match="non-canonical"):
        assemble(0.62, 0.4976932219075496, one_d, three_d, output)


def test_refreeze_rejects_a_previous_generation_mixed_oof(tmp_path: Path) -> None:
    mixed = tmp_path / "mixed.csv"
    _write_csv(mixed, [{
        "source_index": 0,
        "fold": 0,
        "sample_id": "a",
        "sequence": "AAA",
        "label": 0,
        "baseline_probability": 0.1,
        "expert_probability": 0.2,
        "toxicity_probability": 0.15,
    }])
    with pytest.raises(ValueError, match="mixed/fused OOF"):
        load_prob_by_sample_id(mixed, require_pure_3d=True)
