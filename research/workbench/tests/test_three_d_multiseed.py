import csv
import json
from pathlib import Path

from ghxtox.three_d_multiseed import (
    FIXED_THRESHOLD,
    _probability_metrics,
    aggregate_multiseed,
    compare_reference,
    prepare_protocol,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prediction_rows(probabilities: list[float]) -> list[dict[str, str | int | float]]:
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    lengths = [12, 12, 18, 18, 25, 25, 35, 35, 45, 45]
    return [
        {
            "source_index": index,
            "fold": index // 2,
            "sample_id": f"peptide_{index + 1}",
            "sequence": "A" * lengths[index],
            "label": labels[index],
            "toxicity_probability": probabilities[index],
        }
        for index in range(10)
    ]


def _write_predictions(path: Path, probabilities: list[float]) -> None:
    rows = _prediction_rows(probabilities)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_quality(path: Path) -> None:
    rows = [
        {
            "source_index": index,
            "fold": index // 2,
            "generator_source": "generated" if index % 4 < 2 else "not_generated",
        }
        for index in range(10)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_and_reference_gate(tmp_path: Path):
    base_config = tmp_path / "base.json"
    residual_config = tmp_path / "residual.json"
    _write_json(
        base_config, {"seed": 42, "train": {"initial_checkpoint": "old-{fold}"}}
    )
    _write_json(
        residual_config, {"seed": 42, "train": {"initial_checkpoint": "old-{fold}"}}
    )
    artifacts = {
        name: tmp_path / name
        for name in (
            "base.pt",
            "residual_train.pt",
            "residual_oof.pt",
            "base_folds.csv",
            "residual_folds.csv",
        )
    }
    for path in artifacts.values():
        path.write_bytes(path.name.encode())
    for fold in range(5):
        checkpoint = tmp_path / "control" / f"fold{fold}" / "best_model.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"fold-{fold}".encode())
    reference_oof = tmp_path / "reference.csv"
    probabilities = [0.90, 0.10, 0.85, 0.15, 0.80, 0.20, 0.75, 0.25, 0.70, 0.30]
    _write_predictions(reference_oof, probabilities)

    output_root = tmp_path / "output"
    protocol = prepare_protocol(
        output_root=output_root,
        base_config=base_config,
        residual_config=residual_config,
        base_processed=artifacts["base.pt"],
        residual_train_processed=artifacts["residual_train.pt"],
        residual_oof_processed=artifacts["residual_oof.pt"],
        base_fold_manifest=artifacts["base_folds.csv"],
        residual_fold_manifest=artifacts["residual_folds.csv"],
        reference_oof=reference_oof,
        base_control_pattern=str(tmp_path / "control" / "fold{fold}" / "best_model.pt"),
    )

    assert protocol["seeds"] == [42, 123, 2025]
    assert protocol["external_test_labels_read"] is False
    seed123 = json.loads(
        (output_root / "configs" / "residual_seed123.json").read_text(encoding="utf-8")
    )
    assert seed123["seed"] == 123
    assert seed123["train"]["initial_checkpoint"].endswith(
        "seed_123/base/fold{fold}/best_model.pt"
    )

    candidate_oof = tmp_path / "candidate.csv"
    _write_predictions(candidate_oof, probabilities)
    result = compare_reference(
        reference_oof, candidate_oof, tmp_path / "reference_gate.json"
    )
    assert result["new_seeds_authorized"] is True
    assert result["probability_comparison"]["maximum_absolute_difference"] == 0.0


def test_fixed_three_seed_aggregate_promotes_only_ensemble(tmp_path: Path):
    reference = tmp_path / "reference.csv"
    _write_predictions(
        reference, [0.90, 0.10, 0.85, 0.15, 0.80, 0.20, 0.75, 0.25, 0.60, 0.70]
    )
    seed_oofs = {}
    for seed, last_negative in ((42, 0.30), (123, 0.25), (2025, 0.20)):
        path = tmp_path / f"seed_{seed}.csv"
        _write_predictions(
            path, [0.98, 0.02, 0.97, 0.03, 0.96, 0.04, 0.95, 0.05, 0.90, last_negative]
        )
        seed_oofs[seed] = path

    quality = tmp_path / "quality.csv"
    _write_quality(quality)
    summary_path = tmp_path / "summary.json"
    decision = aggregate_multiseed(
        reference_oof=reference,
        seed_oofs=seed_oofs,
        output_csv=tmp_path / "ensemble.csv",
        output_summary=summary_path,
        output_decision=tmp_path / "decision.json",
        quality_csv=quality,
    )

    assert decision["best_single_seed_selection_authorized"] is False
    assert decision["external_test_labels_read"] is False
    assert decision["gate"]["all_conditions_passed"] is True
    with (tmp_path / "ensemble.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert float(rows[-1]["decision_threshold"]) == FIXED_THRESHOLD
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary["ensemble_by_generator"]) == {"generated", "not_generated"}
    assert "ensemble_vs_reference" in summary["decision_disagreements"]
    assert summary["runtime_and_storage"]["42"]["files"] >= 1


def test_probability_metrics_preserve_float64_ranking():
    metrics = _probability_metrics(
        labels=[1, 0],
        probabilities=[0.500000001, 0.5],
    )
    assert metrics["auprc"] == 1.0
