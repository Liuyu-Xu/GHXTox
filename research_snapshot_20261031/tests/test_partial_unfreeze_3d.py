from __future__ import annotations

import math

import torch

from ghxtox.ensemble_domain_ssl_3d import EnsembleTensor
from ghxtox.features import RESIDUE_FEATURE_DIM
from ghxtox.geometry_pretrain import GeometrySSLModel
from ghxtox.partial_unfreeze_3d import (
    PartialUnfreezeEvidenceModel,
    _collate_evidence,
    _combine_held_evidence,
    _configure_partial_unfreeze,
)


def _ensemble(length: int = 7, conformers: int = 3) -> EnsembleTensor:
    return EnsembleTensor(
        residue_features=torch.randn(length, RESIDUE_FEATURE_DIM),
        coords=torch.randn(conformers, length, 3),
        structure_features=torch.randn(conformers, length, 16),
    )


def test_only_last_encoder_layer_and_head_are_trainable() -> None:
    encoder = GeometrySSLModel(
        structure_feature_dim=16,
        hidden_dim=16,
        num_layers=2,
        dropout=0.1,
    )
    model = PartialUnfreezeEvidenceModel(
        encoder,
        head_hidden_dim=8,
        head_dropout=0.1,
    )
    names = _configure_partial_unfreeze(model)
    assert names
    assert all(
        name.startswith("head.") or name.startswith("encoder.encoder.layers.1.")
        for name in names
    )
    assert not any(name.startswith("encoder.encoder.layers.0.") for name in names)
    assert not any(name.startswith("encoder.node_decoder.") for name in names)


def test_packed_conformer_forward_and_bounded_fusion() -> None:
    items = [
        {"source_index": 2, "label": 0, "ensemble": _ensemble(7, 3)},
        {"source_index": 9, "label": 1, "ensemble": _ensemble(5, 4)},
    ]
    batch = _collate_evidence(
        items,
        conformers_per_sample=2,
        stochastic=False,
    )
    assert batch["coords"].shape == (4, 7, 3)
    assert batch["view_owner"].tolist() == [0, 0, 1, 1]
    encoder = GeometrySSLModel(
        structure_feature_dim=16,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    model = PartialUnfreezeEvidenceModel(
        encoder,
        head_hidden_dim=8,
        head_dropout=0.0,
    )
    logits, embedding = model(batch)
    assert logits.shape == (2,)
    assert embedding.shape == (2, 7, 32)
    logits.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    candidate, delta = _combine_held_evidence(0.7, 100.0, 0.15)
    expected = 1.0 / (1.0 + math.exp(-(math.log(0.7 / 0.3) + 0.15)))
    assert math.isclose(delta, 0.15)
    assert math.isclose(candidate, expected)
    _, negative_delta = _combine_held_evidence(0.7, -100.0, 0.15)
    assert math.isclose(negative_delta, -0.15)
