from __future__ import annotations

import csv
import json

import torch
from torch.utils.data import DataLoader

from ghxtox._legacy.cross_modal_adapter import (
    CoupledFeatureDataset,
    ProtT5StructureAdapter,
    _build_balanced_conformer_sampler,
    _decision_margin_penalty,
    _initialize_adapter_from_checkpoint,
    _run_adapter_epoch,
    aggregate_oof_runs,
    attach_conformer_features_to_adapter,
    collate_coupled,
    summarize_oof_ablation,
)
from ghxtox.geometry_pretrain import (
    GeometrySSLModel,
    cross_view_geometry_loss,
    geometry_ssl_loss,
)


def _payload() -> dict:
    return {
        "format_version": 1,
        "token_embeddings": torch.arange(30, dtype=torch.float16).reshape(5, 6),
        "spatial_embeddings": torch.arange(20, dtype=torch.float16).reshape(5, 4),
        "geometry_ssl_embeddings": torch.arange(40, dtype=torch.float16).reshape(5, 8),
        "plddt": torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]),
        "offsets": torch.tensor([0, 2, 5]),
        "base_probabilities": torch.tensor([0.25, 0.75]),
        "labels": torch.tensor([0, 1]),
        "folds": torch.tensor([0, 1]),
        "sample_ids": ["peptide_1", "peptide_2"],
        "sequences": ["AC", "DEF"],
        "metadata": {"token_dim": 6, "spatial_dim": 4, "geometry_ssl_dim": 8},
    }


def test_coupled_feature_collation_preserves_residue_alignment() -> None:
    dataset = CoupledFeatureDataset(_payload())
    batch = collate_coupled([dataset[0], dataset[1]])
    assert batch["tokens"].shape == (2, 3, 6)
    assert batch["spatial"].shape == (2, 3, 4)
    assert batch["geometry_ssl"].shape == (2, 3, 8)
    assert batch["mask"].tolist() == [[True, True, False], [True, True, True]]
    assert torch.equal(batch["tokens"][1, :3], torch.arange(12, 30).reshape(3, 6))


def test_adapter_starts_as_and_can_hard_fallback_to_baseline() -> None:
    torch.manual_seed(3)
    model = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
    )
    batch = collate_coupled(
        [CoupledFeatureDataset(_payload())[0], CoupledFeatureDataset(_payload())[1]]
    )
    model.eval()
    initial = model(
        tokens=batch["tokens"],
        spatial=batch["spatial"],
        plddt=batch["plddt"],
        mask=batch["mask"],
        base_probability=batch["base_probability"],
    )
    assert torch.allclose(
        torch.sigmoid(initial["logits"]), batch["base_probability"], atol=1e-7
    )
    with torch.no_grad():
        model.delta_head[-1].bias.fill_(2.0)
    fallback = model(
        tokens=batch["tokens"],
        spatial=batch["spatial"],
        plddt=batch["plddt"],
        mask=batch["mask"],
        base_probability=batch["base_probability"],
        structure_available=torch.zeros(2),
    )
    assert torch.allclose(
        torch.sigmoid(fallback["logits"]), batch["base_probability"], atol=1e-7
    )
    assert torch.count_nonzero(fallback["interaction_delta"]) == 0


def test_local_chemical_gate_preserves_zero_initialized_baseline() -> None:
    torch.manual_seed(4)
    model = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        chemical_dim=12,
    )
    batch = collate_coupled(
        [CoupledFeatureDataset(_payload())[0], CoupledFeatureDataset(_payload())[1]]
    )
    output = model(
        tokens=batch["tokens"],
        spatial=batch["spatial"],
        plddt=batch["plddt"],
        mask=batch["mask"],
        base_probability=batch["base_probability"],
        chemical=torch.randn(2, 3, 12),
    )
    assert torch.allclose(
        torch.sigmoid(output["logits"]), batch["base_probability"], atol=1e-7
    )


def test_geometry_ssl_gate_preserves_zero_initialized_baseline() -> None:
    torch.manual_seed(5)
    model = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        geometry_ssl_dim=8,
    )
    dataset = CoupledFeatureDataset(_payload())
    batch = collate_coupled([dataset[0], dataset[1]])
    output = model(
        tokens=batch["tokens"],
        spatial=batch["spatial"],
        plddt=batch["plddt"],
        mask=batch["mask"],
        base_probability=batch["base_probability"],
        geometry_ssl=batch["geometry_ssl"],
    )
    assert torch.allclose(
        torch.sigmoid(output["logits"]), batch["base_probability"], atol=1e-7
    )


def test_bidirectional_coupling_has_exact_initial_and_missing_structure_fallback() -> (
    None
):
    torch.manual_seed(6)
    model = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        bidirectional_coupling=True,
    )
    dataset = CoupledFeatureDataset(_payload())
    batch = collate_coupled([dataset[0], dataset[1]])
    initial = model(
        tokens=batch["tokens"],
        spatial=batch["spatial"],
        plddt=batch["plddt"],
        mask=batch["mask"],
        base_probability=batch["base_probability"],
    )
    assert torch.equal(
        initial["bidirectional_interaction_delta"],
        torch.zeros_like(batch["base_probability"]),
    )
    assert torch.allclose(
        torch.sigmoid(initial["logits"]), batch["base_probability"], atol=1e-7
    )
    assert initial["residue_mutual_gate"].shape == batch["mask"].shape
    assert initial["residue_mutual_agreement"].shape == batch["mask"].shape
    assert torch.count_nonzero(initial["residue_mutual_gate"][~batch["mask"]]) == 0
    assert torch.all(initial["residue_mutual_gate"] >= 0)
    assert torch.all(initial["residue_mutual_gate"] <= 1)
    assert torch.all(initial["residue_mutual_agreement"] >= 0)
    assert torch.all(initial["residue_mutual_agreement"] <= 1)

    with torch.no_grad():
        model.bidirectional_delta_head[-1].bias.fill_(2.0)
    active = model(
        tokens=batch["tokens"],
        spatial=batch["spatial"],
        plddt=batch["plddt"],
        mask=batch["mask"],
        base_probability=batch["base_probability"],
    )
    assert torch.all(active["bidirectional_interaction_delta"] > 0)
    fallback = model(
        tokens=batch["tokens"],
        spatial=batch["spatial"],
        plddt=batch["plddt"],
        mask=batch["mask"],
        base_probability=batch["base_probability"],
        structure_available=torch.zeros(2),
    )
    assert torch.equal(
        fallback["interaction_delta"], torch.zeros_like(batch["base_probability"])
    )
    assert torch.allclose(
        torch.sigmoid(fallback["logits"]), batch["base_probability"], atol=1e-7
    )


def test_hierarchical_three_head_adapter_has_exact_initial_fallback() -> None:
    torch.manual_seed(6)
    model = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        geometry_ssl_dim=8,
        hierarchical_fusion=True,
        num_motifs=2,
    )
    dataset = CoupledFeatureDataset(_payload())
    batch = collate_coupled([dataset[0], dataset[1]])
    output = model(
        tokens=batch["tokens"],
        spatial=batch["spatial"],
        plddt=batch["plddt"],
        mask=batch["mask"],
        base_probability=batch["base_probability"],
        geometry_ssl=batch["geometry_ssl"],
    )
    assert output["sequence_aux_logits"].shape == (2,)
    assert output["structure_aux_logits"].shape == (2,)
    assert torch.allclose(
        torch.sigmoid(output["logits"]), batch["base_probability"], atol=1e-7
    )


def test_geometry_ssl_objectives_are_finite() -> None:
    torch.manual_seed(7)
    model = GeometrySSLModel(
        structure_feature_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    batch = {
        "residue_features": torch.randn(2, 5, 42),
        "structure_features": torch.randn(2, 5, 16),
        "coords": torch.randn(2, 5, 3) * 4.0,
        "plddt": torch.full((2, 5), 0.8),
        "mask": torch.tensor(
            [[True, True, True, True, False], [True, True, True, True, True]]
        ),
    }
    output = model(**batch)
    loss, report = geometry_ssl_loss(output, batch)
    assert output["hidden"].shape == (2, 5, 16)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in report.values())


def test_cross_view_geometry_objectives_are_finite() -> None:
    torch.manual_seed(8)
    model = GeometrySSLModel(
        structure_feature_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )
    mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]]
    )
    anchor_coords = torch.randn(2, 5, 3) * 3.0
    conformers = torch.stack(
        [
            anchor_coords,
            anchor_coords + torch.randn_like(anchor_coords) * 0.2,
        ],
        dim=1,
    )
    batch = {
        "residue_features": torch.randn(2, 5, 42),
        "structure_features": torch.randn(2, 5, 16),
        "coords": anchor_coords,
        "plddt": torch.full((2, 5), 0.8),
        "mask": mask,
        "view_structure_features": torch.randn(2, 5, 16),
        "view_coords": conformers[:, 1],
        "view_plddt": torch.full((2, 5), 0.75),
        "conformer_graph_residue_coords": conformers,
        "conformer_graph_conformer_mask": torch.ones(2, 2, dtype=torch.bool),
    }
    loss, report = cross_view_geometry_loss(model, batch)
    assert torch.isfinite(loss)
    assert report["consistency"] >= 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in report.values())


def test_conformer_features_preserve_alignment_and_exact_missing_fallback() -> None:
    payload = _payload()
    payload.update(
        {
            "conformer_residue_features": torch.arange(40, dtype=torch.float16).reshape(
                5, 8
            ),
            "conformer_global_features": torch.arange(24, dtype=torch.float16).reshape(
                2, 12
            ),
            "conformer_source_features": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float16
            ),
            "conformer_available": torch.tensor([True, False]),
        }
    )
    dataset = CoupledFeatureDataset(payload)
    batch = collate_coupled([dataset[0], dataset[1]])
    assert batch["conformer_residue"].shape == (2, 3, 8)
    assert batch["conformer_global"].shape == (2, 12)
    assert batch["conformer_available"].tolist() == [True, False]

    model = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        conformer_residue_dim=8,
        conformer_global_dim=12,
        conformer_source_dim=3,
        source_specific_rank=4,
    )
    with torch.no_grad():
        model.delta_head[-1].bias.fill_(2.0)
    output = model(
        tokens=batch["tokens"],
        spatial=batch["spatial"],
        plddt=batch["plddt"],
        mask=batch["mask"],
        base_probability=batch["base_probability"],
        conformer_residue=batch["conformer_residue"],
        conformer_global=batch["conformer_global"],
        conformer_source=batch["conformer_source"],
        conformer_available=batch["conformer_available"],
    )
    assert torch.count_nonzero(output["residue_conformer_gate"][1]) == 0
    assert output["global_conformer_gate"][1] == 0

    hard_fallback = model(
        tokens=batch["tokens"],
        spatial=batch["spatial"],
        plddt=batch["plddt"],
        mask=batch["mask"],
        base_probability=batch["base_probability"],
        structure_available=torch.zeros(2),
        conformer_residue=batch["conformer_residue"],
        conformer_global=batch["conformer_global"],
        conformer_source=batch["conformer_source"],
        conformer_available=batch["conformer_available"],
    )
    assert torch.allclose(
        torch.sigmoid(hard_fallback["logits"]),
        batch["base_probability"],
        atol=1e-7,
    )


def test_source_length_residual_is_zero_initialized_and_source_specific() -> None:
    payload = _payload()
    payload.update(
        {
            "conformer_residue_features": torch.randn(5, 8),
            "conformer_global_features": torch.randn(2, 12),
            "conformer_source_features": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            "conformer_available": torch.ones(2, dtype=torch.bool),
        }
    )
    batch = collate_coupled(
        [CoupledFeatureDataset(payload)[0], CoupledFeatureDataset(payload)[1]]
    )
    model = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        conformer_residue_dim=8,
        conformer_global_dim=12,
        conformer_source_dim=3,
        source_specific_rank=4,
    )
    arguments = {
        "tokens": batch["tokens"],
        "spatial": batch["spatial"],
        "plddt": batch["plddt"],
        "mask": batch["mask"],
        "base_probability": batch["base_probability"],
        "conformer_residue": batch["conformer_residue"],
        "conformer_global": batch["conformer_global"],
        "conformer_source": batch["conformer_source"],
        "conformer_available": batch["conformer_available"],
    }
    initial = model(**arguments)
    assert torch.allclose(torch.sigmoid(initial["logits"]), batch["base_probability"])
    assert torch.count_nonzero(initial["conformer_interaction_delta"]) == 0

    with torch.no_grad():
        model.conformer_source_length_head[-1].bias.copy_(
            torch.tensor([1.0, -1.0, 0.0])
        )
    output = model(**arguments)
    assert output["conformer_interaction_delta"][0] > 0
    assert output["conformer_interaction_delta"][1] < 0


def test_balanced_sampler_is_fold_local_and_excludes_unavailable_records() -> None:
    payload = {
        "labels": torch.tensor([0, 0, 0, 1, 1, 1]),
        "sequences": ["A" * 12, "A" * 12, "A" * 18, "A" * 22, "A" * 27, "A" * 35],
        "conformer_available": torch.tensor([True, True, True, True, True, False]),
        "conformer_source_features": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    }
    fold_indices = [0, 1, 2, 3, 4, 5]
    sampler, report = _build_balanced_conformer_sampler(
        payload,
        fold_indices,
        seed=42,
        max_weight_ratio=5.0,
    )
    sampled_positions = list(sampler)
    assert sampler.weights[5] == 0
    assert 5 not in sampled_positions
    assert report["available_records"] == 5
    assert report["num_strata"] == 4


def test_decision_margin_penalty_is_asymmetric_near_threshold() -> None:
    threshold_logit = torch.logit(torch.tensor(0.6))
    logits = torch.tensor(
        [
            float(threshold_logit + 0.05),
            float(threshold_logit - 0.05),
            float(threshold_logit + 0.25),
        ]
    )
    penalty = _decision_margin_penalty(
        logits,
        torch.tensor([1.0, 0.0, 1.0]),
        threshold=0.6,
        margin=0.15,
        positive_multiplier=2.0,
    )
    assert torch.allclose(penalty, torch.tensor([0.2, 0.1, 0.0]), atol=1e-6)


def test_attach_conformer_features_validates_and_writes_adapter_cache(tmp_path) -> None:
    feature_cache = tmp_path / "adapter.pt"
    payload = _payload()
    torch.save(payload, feature_cache)
    processed = tmp_path / "processed.pt"
    records = [
        {
            "sample_id": "peptide_1",
            "sequence": "AC",
            "label": 0,
            "conformer_residue_features": torch.ones(2, 8),
            "conformer_global_features": torch.ones(12),
            "conformer_source_features": torch.tensor([1.0, 0.0, 0.0]),
            "conformer_available": True,
        },
        {
            "sample_id": "peptide_2",
            "sequence": "DEF",
            "label": 1,
            "conformer_residue_features": torch.zeros(3, 8),
            "conformer_global_features": torch.zeros(12),
            "conformer_source_features": torch.zeros(3),
            "conformer_available": False,
        },
    ]
    torch.save({"records": records}, processed)

    summary = attach_conformer_features_to_adapter(feature_cache, processed)
    attached = torch.load(feature_cache, map_location="cpu", weights_only=False)

    assert summary["covered_samples"] == 1
    assert attached["conformer_residue_features"].shape == (5, 8)
    assert attached["conformer_global_features"].shape == (2, 12)
    assert attached["conformer_available"].tolist() == [True, False]


def test_frozen_base_initialization_is_exact_before_conformer_training(
    tmp_path,
) -> None:
    torch.manual_seed(9)
    base = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        chemical_dim=12,
    )
    with torch.no_grad():
        base.delta_head[-1].bias.fill_(0.7)
    checkpoint = tmp_path / "base.pt"
    torch.save({"model_state": base.state_dict()}, checkpoint)

    coupled = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        chemical_dim=12,
        conformer_residue_dim=8,
        conformer_global_dim=12,
        conformer_source_dim=3,
        source_specific_rank=4,
    )
    report = _initialize_adapter_from_checkpoint(
        coupled, checkpoint, freeze_base_adapter=True
    )
    assert report["unexpected_keys"] == []
    assert all(
        name.startswith("conformer_")
        for name, parameter in coupled.named_parameters()
        if parameter.requires_grad
    )

    batch = collate_coupled(
        [CoupledFeatureDataset(_payload())[0], CoupledFeatureDataset(_payload())[1]]
    )
    common = {
        "tokens": batch["tokens"],
        "spatial": batch["spatial"],
        "plddt": batch["plddt"],
        "mask": batch["mask"],
        "base_probability": batch["base_probability"],
        "chemical": torch.randn(2, 3, 12),
    }
    base.eval()
    coupled.eval()
    base_output = base(**common)
    coupled_output = coupled(
        **common,
        conformer_residue=torch.randn(2, 3, 8),
        conformer_global=torch.randn(2, 12),
        conformer_source=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        conformer_available=torch.ones(2, dtype=torch.bool),
    )
    assert torch.equal(coupled_output["logits"], base_output["logits"])
    assert torch.count_nonzero(coupled_output["conformer_interaction_delta"]) == 0

    training_payload = _payload()
    training_payload.update(
        {
            "chemical_features": torch.randn(5, 12),
            "conformer_residue_features": torch.randn(5, 8),
            "conformer_global_features": torch.randn(2, 12),
            "conformer_source_features": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            "conformer_available": torch.ones(2, dtype=torch.bool),
        }
    )
    loader = DataLoader(
        CoupledFeatureDataset(training_payload),
        batch_size=2,
        collate_fn=collate_coupled,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in coupled.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    _run_adapter_epoch(
        coupled,
        loader,
        torch.device("cpu"),
        optimizer,
        threshold=0.6,
        structure_dropout=0.0,
        delta_l1=0.01,
        hard_example_weight=0.0,
        conformer_only_loss=True,
    )
    assert coupled.token_projection.training is False
    assert coupled.cross_attention.training is False
    assert coupled.delta_head.training is False
    assert coupled.conformer_delta_head.training is True


def test_frozen_checkpoint_trains_only_bidirectional_extension(tmp_path) -> None:
    torch.manual_seed(10)
    base = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        chemical_dim=12,
    )
    with torch.no_grad():
        base.delta_head[-1].bias.fill_(0.4)
    checkpoint = tmp_path / "base.pt"
    torch.save({"model_state": base.state_dict()}, checkpoint)

    coupled = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        chemical_dim=12,
        bidirectional_coupling=True,
    )
    report = _initialize_adapter_from_checkpoint(
        coupled, checkpoint, freeze_base_adapter=True
    )
    assert report["unexpected_keys"] == []
    extension_prefixes = ("reverse_", "mutual_", "bidirectional_")
    assert all(
        name.startswith(extension_prefixes)
        for name, parameter in coupled.named_parameters()
        if parameter.requires_grad
    )

    training_payload = _payload()
    training_payload["chemical_features"] = torch.randn(5, 12)
    loader = DataLoader(
        CoupledFeatureDataset(training_payload),
        batch_size=2,
        collate_fn=collate_coupled,
    )
    batch = next(iter(loader))
    common = {
        "tokens": batch["tokens"],
        "spatial": batch["spatial"],
        "plddt": batch["plddt"],
        "mask": batch["mask"],
        "base_probability": batch["base_probability"],
        "chemical": batch["chemical"],
    }
    base.eval()
    coupled.eval()
    assert torch.equal(coupled(**common)["logits"], base(**common)["logits"])

    optimizer = torch.optim.AdamW(
        [parameter for parameter in coupled.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    result = _run_adapter_epoch(
        coupled,
        loader,
        torch.device("cpu"),
        optimizer,
        threshold=0.6,
        structure_dropout=0.0,
        delta_l1=0.01,
        hard_example_weight=0.0,
        bidirectional_only_loss=True,
    )
    assert result.bidirectional_deltas.shape == (2,)
    assert result.mean_mutual_gates.shape == (2,)
    assert coupled.token_projection.training is False
    assert coupled.cross_attention.training is False
    assert coupled.delta_head.training is False
    assert coupled.reverse_cross_attention.training is True
    assert coupled.bidirectional_delta_head.training is True


def test_hierarchical_initialization_resets_incompatible_delta_head(tmp_path) -> None:
    torch.manual_seed(10)
    base = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
    )
    with torch.no_grad():
        base.delta_head[-1].bias.fill_(0.7)
    checkpoint = tmp_path / "base.pt"
    torch.save({"model_state": base.state_dict()}, checkpoint)

    hierarchical = ProtT5StructureAdapter(
        token_dim=6,
        spatial_dim=4,
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        geometry_ssl_dim=8,
        hierarchical_fusion=True,
        num_motifs=2,
    )
    report = _initialize_adapter_from_checkpoint(
        hierarchical, checkpoint, freeze_base_adapter=False
    )
    assert any(
        mismatch.startswith("delta_head.0.weight:")
        for mismatch in report["shape_mismatches"]
    )

    dataset = CoupledFeatureDataset(_payload())
    batch = collate_coupled([dataset[0], dataset[1]])
    output = hierarchical(
        tokens=batch["tokens"],
        spatial=batch["spatial"],
        plddt=batch["plddt"],
        mask=batch["mask"],
        base_probability=batch["base_probability"],
        geometry_ssl=batch["geometry_ssl"],
    )
    assert torch.allclose(
        torch.sigmoid(output["logits"]), batch["base_probability"], atol=1e-7
    )


def test_aggregate_oof_runs_checks_alignment_and_averages_probabilities(
    tmp_path,
) -> None:
    fields = [
        "source_index",
        "fold",
        "sample_id",
        "sequence",
        "label",
        "baseline_probability",
        "comparison_probability",
        "interaction_delta",
        "toxicity_probability",
        "prediction",
    ]
    common = [
        ["0", "0", "a", "AC", "0", "0.2", "0.2", "0", None, "0"],
        ["1", "1", "b", "DE", "1", "0.8", "0.8", "0", None, "1"],
    ]
    inputs = []
    for run_index, probabilities in enumerate(([0.1, 0.7], [0.3, 0.9])):
        path = tmp_path / f"run{run_index}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            for row, probability in zip(common, probabilities):
                value = list(row)
                value[8] = str(probability)
                writer.writerow(value)
        inputs.append(path)

    summary = aggregate_oof_runs(inputs, tmp_path / "ensemble", threshold=0.6)
    with (tmp_path / "ensemble" / "oof_predictions.csv").open(
        encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert torch.allclose(
        torch.tensor([float(row["toxicity_probability"]) for row in rows]),
        torch.tensor([0.2, 0.8]),
    )
    assert summary["protocol"]["test_labels_used"] is False


def test_summarize_oof_ablation_writes_fixed_matrix_outputs(tmp_path) -> None:
    baseline = {"mcc": 0.8, "auprc": 0.9, "brier": 0.1}
    paths = []
    for name, delta, passed in (
        ("control", [0.0, 0.01], False),
        ("candidate", [0.02, 0.03], True),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "baseline": baseline,
                    "adapter": {
                        "accuracy": 0.9,
                        "sensitivity": 0.8,
                        "specificity": 0.9,
                        "mcc": 0.81,
                        "auroc": 0.92,
                        "auprc": 0.91,
                        "brier": 0.09,
                        "ece_10": 0.02,
                    },
                    "fold_reports": [
                        {"fold": fold, "mcc_delta": value}
                        for fold, value in enumerate(delta)
                    ],
                    "protocol": {
                        "decision_margin_weight": 0.1 if name == "candidate" else 0,
                        "source_specific_rank": 8 if name == "candidate" else 0,
                        "balanced_conformer_sampling": name == "candidate",
                    },
                    "retention_gate": {"passed": passed},
                }
            ),
            encoding="utf-8",
        )
        paths.append(f"{name}={path}")

    summary = summarize_oof_ablation(
        paths,
        output_json=tmp_path / "matrix.json",
        output_csv=tmp_path / "matrix.csv",
    )
    assert (tmp_path / "matrix.json").is_file()
    assert (tmp_path / "matrix.csv").is_file()
    assert summary["protocol"]["test1_or_test2_used"] is False
    assert summary["variants"][1]["fold_mcc_wins"] == 2
    assert summary["variants"][1]["retention_passed"] is True
