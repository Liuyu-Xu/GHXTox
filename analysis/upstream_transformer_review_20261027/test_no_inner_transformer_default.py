"""Protect native bypass parity and reject silent fallback to legacy weights."""
from pathlib import Path

import pytest
import torch
from torch import nn

from ghxtox.models.layers import SequenceBranch
from ghxtox.multidimensional_default import _validate_structure_checkpoint_architecture
from ghxtox.utils import resolve_inference_checkpoint


class HistoricalBypass(nn.Module):
    def forward(self, src, **kwargs):
        return src


def branch(layers):
    return SequenceBranch(vocab_size=24, group_size=10, residue_feature_dim=5,
                          plm_embedding_dim=8, aa_embedding_dim=4, group_embedding_dim=3,
                          hidden_dim=16, num_layers=layers, num_heads=4, dropout=0).eval()


def test_native_bypass_matches_existing_ablation_and_preserves_legacy_loading():
    legacy = branch(2)
    assert any(key.startswith("encoder.layers.") for key in legacy.state_dict())
    branch(2).load_state_dict(legacy.state_dict(), strict=True)
    legacy.encoder = HistoricalBypass()
    native = branch(0)
    native.load_state_dict(legacy.state_dict(), strict=True)
    inputs = (torch.tensor([[1, 2, 3, 0]]), torch.tensor([[1, 2, 3, 0]]),
              torch.randn(1, 4, 5), torch.tensor([[True, True, True, False]]), torch.randn(1, 4, 8))
    with torch.inference_mode():
        torch.testing.assert_close(native(*inputs), legacy(*inputs), rtol=0, atol=0)
    assert not any(key.startswith("encoder.") for key in native.state_dict())


def test_public_predictor_rejects_legacy_architecture_early(tmp_path: Path):
    checkpoint = tmp_path / "wrong.pt"
    torch.save({"config": {"model": {"num_sequence_layers": 2}}, "model_state": {}}, checkpoint)
    with pytest.raises(ValueError, match="num_sequence_layers=0"):
        _validate_structure_checkpoint_architecture(checkpoint)


def test_renamed_legacy_weights_are_rejected_even_if_config_claims_zero_layers(tmp_path: Path):
    checkpoint = tmp_path / "renamed.pt"
    torch.save({"config": {"model": {"num_sequence_layers": 0}},
                "model_state": {"sequence_branch.encoder.layers.0.weight": torch.ones(1)}}, checkpoint)
    with pytest.raises(ValueError, match="extra Transformer parameters"):
        _validate_structure_checkpoint_architecture(checkpoint)


def test_default_rejects_missing_sites_instead_of_switching_models(tmp_path: Path):
    checkpoint = tmp_path / "new.pt"
    torch.save({"config": {"model": {"chemical_site_branch": True},
                           "inference": {"fallback_checkpoint": None}}}, checkpoint)
    with pytest.raises(ValueError, match="requires chemical-site features"):
        resolve_inference_checkpoint(checkpoint, [{}], torch.device("cpu"))

