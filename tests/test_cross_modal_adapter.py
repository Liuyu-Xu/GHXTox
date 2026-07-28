from __future__ import annotations

import torch

from ghxtox.cross_modal_adapter import (
    CoupledFeatureDataset,
    ProtT5StructureAdapter,
    collate_coupled,
)


def _payload() -> dict:
    return {
        "format_version": 1,
        "token_embeddings": torch.arange(30, dtype=torch.float16).reshape(5, 6),
        "spatial_embeddings": torch.arange(20, dtype=torch.float16).reshape(5, 4),
        "plddt": torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]),
        "offsets": torch.tensor([0, 2, 5]),
        "base_probabilities": torch.tensor([0.25, 0.75]),
        "labels": torch.tensor([0, 1]),
        "folds": torch.tensor([0, 1]),
        "sample_ids": ["peptide_1", "peptide_2"],
        "sequences": ["AC", "DEF"],
        "metadata": {"token_dim": 6, "spatial_dim": 4},
    }


def test_coupled_feature_collation_preserves_residue_alignment() -> None:
    dataset = CoupledFeatureDataset(_payload())
    batch = collate_coupled([dataset[0], dataset[1]])
    assert batch["tokens"].shape == (2, 3, 6)
    assert batch["spatial"].shape == (2, 3, 4)
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
