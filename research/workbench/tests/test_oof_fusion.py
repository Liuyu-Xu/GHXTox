from __future__ import annotations

import csv
import math

import pytest

from ghxtox.oof_fusion import apply_frozen_fusion, screen_oof_fusion


def _write_predictions(
    path,
    probabilities,
    *,
    with_folds,
    sample_prefix="sample",
    with_source_index=False,
    fold_offset=0,
):
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    fieldnames = ["sample_id", "sequence", "label", "toxicity_probability"]
    if with_folds:
        fieldnames.insert(0, "fold")
    if with_source_index:
        fieldnames.insert(0, "source_index")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (label, probability) in enumerate(zip(labels, probabilities)):
            row = {
                "sample_id": f"{sample_prefix}_{index}",
                "sequence": "AC" + "D" * index,
                "label": label,
                "toxicity_probability": probability,
            }
            if with_folds:
                row["fold"] = (index + fold_offset) % 5
            if with_source_index:
                row["source_index"] = index
            writer.writerow(row)


def test_screen_and_apply_oof_fusion(tmp_path) -> None:
    baseline = tmp_path / "baseline.csv"
    expert = tmp_path / "expert.csv"
    _write_predictions(
        baseline,
        [0.1, 0.7, 0.3, 0.6, 0.2, 0.8, 0.4, 0.55, 0.45, 0.9],
        with_folds=True,
    )
    _write_predictions(
        expert,
        [0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.2, 0.8, 0.4, 0.7],
        with_folds=False,
    )
    screened = screen_oof_fusion(
        baseline, expert, tmp_path / "screen", weight_step=0.25, threshold_step=0.1
    )
    assert screened["protocol"]["num_folds"] == 5
    assert 0.0 <= screened["frozen_parameters_from_all_oof"]["expert_weight"] <= 1.0
    applied = apply_frozen_fusion(
        baseline,
        expert,
        tmp_path / "applied",
        expert_weight=screened["frozen_parameters_from_all_oof"]["expert_weight"],
        threshold=screened["frozen_parameters_from_all_oof"]["threshold"],
    )
    assert applied["protocol"]["parameters_selected_from_test_labels"] is False
    assert (tmp_path / "applied" / "predictions.csv").exists()


def test_screen_accepts_verified_id_aliases_and_external_fold_manifest(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline.csv"
    expert = tmp_path / "expert.csv"
    manifest = tmp_path / "folds.csv"
    probabilities = [0.1, 0.7, 0.3, 0.6, 0.2, 0.8, 0.4, 0.55, 0.45, 0.9]
    _write_predictions(
        baseline, probabilities, with_folds=False, sample_prefix="peptide"
    )
    _write_predictions(expert, probabilities, with_folds=False, sample_prefix="train")
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_index", "sample_id", "sequence", "label", "fold"],
        )
        writer.writeheader()
        for index, label in enumerate([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]):
            writer.writerow(
                {
                    "source_index": index,
                    "sample_id": f"manifest_{index}",
                    "sequence": "AC" + "D" * index,
                    "label": label,
                    "fold": index % 5,
                }
            )
    screened = screen_oof_fusion(
        baseline,
        expert,
        tmp_path / "screen_alias",
        weight_step=0.25,
        threshold_step=0.1,
        fold_manifest=manifest,
    )
    assert screened["protocol"]["num_folds"] == 5
    assert screened["protocol"]["fold_manifest"] == str(manifest)


def test_screen_can_cap_an_auxiliary_expert_weight(tmp_path) -> None:
    baseline = tmp_path / "baseline.csv"
    expert = tmp_path / "expert.csv"
    _write_predictions(
        baseline,
        [0.1, 0.7, 0.3, 0.6, 0.2, 0.8, 0.4, 0.55, 0.45, 0.9],
        with_folds=True,
    )
    _write_predictions(
        expert,
        [0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.2, 0.8, 0.4, 0.7],
        with_folds=False,
    )
    screened = screen_oof_fusion(
        baseline,
        expert,
        tmp_path / "screen_capped",
        weight_step=0.1,
        threshold_step=0.1,
        expert_weight_max=0.3,
    )
    assert screened["protocol"]["expert_weight_max"] == 0.3
    assert screened["frozen_parameters_from_all_oof"]["expert_weight"] <= 0.3
    assert all(
        row["expert_weight"] <= 0.3 for row in screened["crossfit_fold_selection"]
    )

    logit_screened = screen_oof_fusion(
        baseline,
        expert,
        tmp_path / "screen_capped_logit",
        weight_step=0.1,
        threshold_step=0.1,
        expert_weight_max=0.3,
        fusion_space="logit",
    )
    assert logit_screened["protocol"]["fusion_space"] == "logit"
    assert logit_screened["frozen_parameters_from_all_oof"]["expert_weight"] <= 0.3

    frozen = logit_screened["frozen_parameters_from_all_oof"]
    applied = apply_frozen_fusion(
        baseline,
        expert,
        tmp_path / "applied_logit",
        expert_weight=frozen["expert_weight"],
        threshold=frozen["threshold"],
        fusion_space=frozen["fusion_space"],
    )
    assert applied["protocol"]["fusion_space"] == "logit"
    with (tmp_path / "applied_logit" / "predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        applied_rows = list(csv.DictReader(handle))
    weight = float(frozen["expert_weight"])
    baseline_probability = 0.1
    expert_probability = 0.2
    expected_logit = (1.0 - weight) * math.log(
        baseline_probability / (1.0 - baseline_probability)
    ) + weight * math.log(expert_probability / (1.0 - expert_probability))
    expected_probability = 1.0 / (1.0 + math.exp(-expected_logit))
    assert float(applied_rows[0]["toxicity_probability"]) == pytest.approx(
        expected_probability
    )


def test_alignment_rejects_fold_mismatch_and_invalid_probabilities(tmp_path) -> None:
    baseline = tmp_path / "baseline.csv"
    expert = tmp_path / "expert.csv"
    probabilities = [0.1, 0.7, 0.3, 0.6, 0.2, 0.8, 0.4, 0.55, 0.45, 0.9]
    _write_predictions(
        baseline,
        probabilities,
        with_folds=True,
        with_source_index=True,
    )
    _write_predictions(
        expert,
        probabilities,
        with_folds=True,
        with_source_index=True,
        fold_offset=1,
    )
    with pytest.raises(ValueError, match="column fold"):
        screen_oof_fusion(baseline, expert, tmp_path / "fold_mismatch")

    _write_predictions(
        expert,
        [float("nan"), *probabilities[1:]],
        with_folds=True,
        with_source_index=True,
    )
    with pytest.raises(ValueError, match="must all be finite"):
        screen_oof_fusion(baseline, expert, tmp_path / "nan")

    _write_predictions(
        expert,
        [1.1, *probabilities[1:]],
        with_folds=True,
        with_source_index=True,
    )
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        screen_oof_fusion(baseline, expert, tmp_path / "out_of_range")


def test_alignment_rejects_duplicate_source_indices(tmp_path) -> None:
    baseline = tmp_path / "baseline.csv"
    expert = tmp_path / "expert.csv"
    probabilities = [0.1, 0.7, 0.3, 0.6, 0.2, 0.8, 0.4, 0.55, 0.45, 0.9]
    _write_predictions(
        baseline,
        probabilities,
        with_folds=True,
        with_source_index=True,
    )
    _write_predictions(
        expert,
        probabilities,
        with_folds=True,
        with_source_index=True,
    )
    with expert.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[-1]["source_index"] = rows[0]["source_index"]
    with expert.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="duplicate source_index"):
        screen_oof_fusion(baseline, expert, tmp_path / "duplicate")
