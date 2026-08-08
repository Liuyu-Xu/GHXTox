from pathlib import Path

import numpy as np
import pytest
import torch

from ghxtox.features import RESIDUE_FEATURE_DIM
from ghxtox.conformer_features import (
    CONFORMER_GLOBAL_FEATURE_DIM,
    CONFORMER_RESIDUE_FEATURE_DIM,
    CONFORMER_SOURCE_FEATURE_DIM,
    attach_conformer_features,
    ensemble_feature_tensors,
)
from ghxtox.data import collate_peptides
from ghxtox.models import GHXToxModel
from ghxtox.models.conformer_graph import DenseChemicalGraphLayer


def _summary(path: Path, length: int = 3) -> None:
    coords = np.zeros((2, length, 3), dtype=np.float32)
    coords[1, -1, 1] = 2.0
    distances = np.linalg.norm(coords[:, :, None] - coords[:, None, :], axis=-1)
    np.savez_compressed(
        path,
        aligned_coords=coords,
        ensemble_mean_coords=coords.mean(axis=0),
        residue_rmsf=np.asarray([0.0, 0.2, 1.0], dtype=np.float32),
        conformer_rmsd=np.asarray([0.5, 0.5], dtype=np.float32),
        radius_gyration=np.asarray([1.0, 1.5], dtype=np.float32),
        pair_distance_mean=distances.mean(axis=0),
        pair_distance_std=distances.std(axis=0),
        contact_occupancy_6=(distances < 6).mean(axis=0).astype(np.float32),
        contact_occupancy_8=(distances < 8).mean(axis=0).astype(np.float32),
        contact_occupancy_10=(distances < 10).mean(axis=0).astype(np.float32),
    )


def _conformer(path: Path, offset: float, length: int = 3) -> None:
    residue_coords = np.zeros((length, 3), dtype=np.float32)
    residue_coords[:, 0] = np.arange(length, dtype=np.float32)
    site_coords = np.zeros((length, 2, 3), dtype=np.float32)
    site_coords[:, 0] = residue_coords + np.asarray([0.0, offset, 0.0], dtype=np.float32)
    site_types = np.zeros((length, 2, 8), dtype=np.float32)
    site_types[:, 0, 0] = 1.0
    site_mask = np.zeros((length, 2), dtype=np.bool_)
    site_mask[:, 0] = True
    np.savez_compressed(
        path,
        sequence=np.asarray("ACD"),
        coords=residue_coords,
        plddt=np.full(length, 0.8, dtype=np.float32),
        chemical_site_coords=site_coords,
        chemical_site_types=site_types,
        chemical_site_mask=site_mask,
    )


def _record(sample_id: str, available: bool = False) -> dict:
    length = 3
    record = {
        "sample_id": sample_id,
        "sequence": "ACD",
        "label": 1,
        "aa_ids": torch.tensor([1, 2, 3]),
        "group_ids": torch.tensor([0, 0, 0]),
        "residue_features": torch.zeros(length, RESIDUE_FEATURE_DIM),
        "coords": torch.zeros(length, 3),
        "plddt": torch.ones(length),
        "structure_features": torch.zeros(length, 16),
    }
    if available:
        record["conformer_residue_features"] = torch.ones(length, CONFORMER_RESIDUE_FEATURE_DIM)
        record["conformer_global_features"] = torch.ones(CONFORMER_GLOBAL_FEATURE_DIM)
        record["conformer_available"] = True
    return record


def test_ensemble_features_have_stable_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "ensemble_summary.npz"
    _summary(path)
    residue, global_value = ensemble_feature_tensors(path)
    assert residue.shape == (3, CONFORMER_RESIDUE_FEATURE_DIM)
    assert global_value.shape == (CONFORMER_GLOBAL_FEATURE_DIM,)
    assert torch.isfinite(residue).all()


def test_attach_maps_indexed_cache_to_original_id(tmp_path: Path) -> None:
    processed = tmp_path / "processed.pt"
    torch.save({"records": [_record("pep_1"), _record("pep_2")]}, processed)
    cache = tmp_path / "cache" / "pep_1_1"
    cache.mkdir(parents=True)
    _summary(cache / "ensemble_summary.npz")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_index,sample_id,original_id,sequence,label,conformer_index,cache_path\n"
        "0,pep_1_1,pep_1,ACD,1,0,/server/conformer_0000.npz\n",
        encoding="utf-8",
    )
    output = tmp_path / "augmented.pt"
    summary = attach_conformer_features(processed, manifest, tmp_path / "cache", output)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert summary["covered_records"] == 1
    assert payload["records"][0]["conformer_available"] is True
    assert payload["records"][0]["conformer_source_features"].shape == (
        CONFORMER_SOURCE_FEATURE_DIM,
    )
    assert payload["records"][1]["conformer_available"] is False


def test_attach_rejects_manifest_sequence_mismatch_even_when_id_and_length_match(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed.pt"
    torch.save({"records": [_record("pep_1")]}, processed)
    cache = tmp_path / "cache" / "scoped_pep_1"
    cache.mkdir(parents=True)
    _summary(cache / "ensemble_summary.npz")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_index,sample_id,original_id,sequence,label,conformer_index,cache_path\n"
        "0,scoped_pep_1,pep_1,ACE,1,0,/server/conformer_0000.npz\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Sequence mismatch for pep_1"):
        attach_conformer_features(
            processed,
            manifest,
            tmp_path / "cache",
            tmp_path / "augmented.pt",
        )


def test_attach_uses_portable_manifest_paths_across_cache_roots(tmp_path: Path, monkeypatch) -> None:
    processed = tmp_path / "processed.pt"
    torch.save({"records": [_record("pep_1"), _record("pep_2")]}, processed)
    first = tmp_path / "cache_a" / "indexed_1"
    second = tmp_path / "cache_b" / "pep_2"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _summary(first / "ensemble_summary.npz")
    _summary(second / "ensemble_summary.npz")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_index,sample_id,original_id,sequence,label,conformer_index,cache_path\n"
        "0,indexed_1,pep_1,ACD,1,0,cache_a/indexed_1/conformer_0000.npz\n"
        "1,pep_2,pep_2,ACD,1,0,cache_b/pep_2/conformer_0000.npz\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "augmented.pt"
    summary = attach_conformer_features(processed, manifest, tmp_path / "unused", output)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert summary["covered_records"] == 2
    assert all(record["conformer_available"] for record in payload["records"])


def test_attach_combines_disjoint_manifests_and_tracks_generator(tmp_path: Path) -> None:
    processed = tmp_path / "processed.pt"
    torch.save({"records": [_record("pep_1"), _record("pep_2")]}, processed)
    roots = [tmp_path / "pepflow", tmp_path / "af2"]
    manifests = []
    sources = ["pepflow", "colabfold_af2_shallow_msa_multiseed"]
    for index, (sample_id, root, source) in enumerate(zip(("pep_1", "pep_2"), roots, sources)):
        sample_root = root / sample_id
        sample_root.mkdir(parents=True)
        _summary(sample_root / "ensemble_summary.npz")
        manifest = tmp_path / f"manifest_{index}.csv"
        manifest.write_text(
            "source_index,sample_id,original_id,sequence,label,conformer_index,source,cache_path\n"
            f"{index},{sample_id},{sample_id},ACD,1,0,{source},/server/conformer_0000.npz\n",
            encoding="utf-8",
        )
        manifests.append(manifest)
    output = tmp_path / "augmented.pt"
    summary = attach_conformer_features(processed, manifests, roots, output)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert summary["covered_records"] == 2
    assert payload["records"][0]["conformer_source_features"].tolist() == [1.0, 0.0, 0.0]
    assert payload["records"][1]["conformer_source_features"].tolist() == [0.0, 1.0, 0.0]


def test_attach_sequence_subset_reuses_unique_source_without_copying_cache(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed.pt"
    torch.save({"records": [_record("test2_id")]}, processed)
    cache = tmp_path / "cache" / "test1_id"
    cache.mkdir(parents=True)
    _summary(cache / "ensemble_summary.npz")
    unused = tmp_path / "cache" / "unused_test1_id"
    unused.mkdir(parents=True)
    _summary(unused / "ensemble_summary.npz")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_index,sample_id,original_id,sequence,label,conformer_index,source,cache_path\n"
        "0,test1_id,test1_original,ACD,1,0,pepflow,/server/test1_id/conformer_0000.npz\n"
        "1,unused_test1_id,unused_original,AAA,0,0,pepflow,/server/unused_test1_id/conformer_0000.npz\n",
        encoding="utf-8",
    )
    output = tmp_path / "augmented.pt"
    summary = attach_conformer_features(
        processed,
        manifest,
        tmp_path / "cache",
        output,
        match_by_sequence=True,
        allow_unused_manifest_records=True,
    )
    record = torch.load(output, map_location="cpu", weights_only=False)["records"][0]
    assert record["conformer_available"] is True
    assert summary["matching_policy"] == "unique_sequence"
    assert summary["unused_manifest_records"] == 1


def test_attach_sequence_subset_rejects_ambiguous_source_sequences(tmp_path: Path) -> None:
    processed = tmp_path / "processed.pt"
    torch.save({"records": [_record("test2_id")]}, processed)
    for sample_id in ("first", "second"):
        cache = tmp_path / "cache" / sample_id
        cache.mkdir(parents=True)
        _summary(cache / "ensemble_summary.npz")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_index,sample_id,original_id,sequence,label,conformer_index,source,cache_path\n"
        "0,first,first_original,ACD,1,0,pepflow,/server/first/conformer_0000.npz\n"
        "1,second,second_original,ACD,1,0,pepflow,/server/second/conformer_0000.npz\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Sequence matching is ambiguous"):
        attach_conformer_features(
            processed,
            manifest,
            tmp_path / "cache",
            tmp_path / "augmented.pt",
            match_by_sequence=True,
            allow_unused_manifest_records=True,
        )


def test_unavailable_conformer_branch_is_exact_fallback() -> None:
    config = {
        "model": {
            "hidden_dim": 16,
            "num_attention_heads": 4,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "structure_feature_dim": 16,
            "conformer_ensemble_branch": True,
            "dropout": 0.0,
        }
    }
    model = GHXToxModel(config).eval()
    unavailable = collate_peptides([_record("pep_1")], include_atom=False)
    available = collate_peptides([_record("pep_1", available=True)], include_atom=False)
    with torch.no_grad():
        first = model(unavailable)
        second = model(available)
    assert torch.equal(first["logits"], first["base_logits"])
    # Zero initialization makes checkpoint warm-start behavior identical before training.
    assert torch.equal(second["logits"], second["base_logits"])
    with torch.no_grad():
        model.conformer_ensemble_branch[-1].bias.fill_(1.0)
    with torch.no_grad():
        changed = model(available)
        still_fallback = model(unavailable)
    assert not torch.equal(changed["logits"], changed["base_logits"])
    assert torch.equal(still_fallback["logits"], still_fallback["base_logits"])


def test_source_aware_conformer_branch_is_exact_fallback() -> None:
    config = {
        "model": {
            "hidden_dim": 16,
            "num_attention_heads": 4,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "structure_feature_dim": 16,
            "conformer_ensemble_branch": True,
            "conformer_ensemble_source_aware": True,
            "dropout": 0.0,
        }
    }
    model = GHXToxModel(config).eval()
    unavailable = collate_peptides([_record("pep_1")], include_atom=False)
    with torch.no_grad():
        output = model(unavailable)
    assert torch.equal(output["logits"], output["base_logits"])


def test_attach_and_collate_per_conformer_graphs(tmp_path: Path) -> None:
    processed = tmp_path / "processed.pt"
    torch.save({"records": [_record("pep_1"), _record("pep_2")]}, processed)
    cache = tmp_path / "cache" / "pep_1"
    cache.mkdir(parents=True)
    _summary(cache / "ensemble_summary.npz")
    _conformer(cache / "conformer_0000.npz", 0.5)
    _conformer(cache / "conformer_0001.npz", 1.0)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_index,sample_id,original_id,sequence,label,conformer_index,source,cache_path\n"
        "0,pep_1,pep_1,ACD,1,0,pepflow,/server/conformer_0000.npz\n",
        encoding="utf-8",
    )
    output = tmp_path / "augmented.pt"
    attach_conformer_features(
        processed,
        manifest,
        tmp_path / "cache",
        output,
        include_conformer_graphs=True,
    )
    records = torch.load(output, map_location="cpu", weights_only=False)["records"]
    batch = collate_peptides(records, include_atom=False)
    assert batch["conformer_graph_residue_coords"].shape == (2, 8, 3, 3)
    assert batch["conformer_graph_site_coords"].shape == (2, 8, 3, 2, 3)
    assert batch["conformer_graph_site_types"].shape == (2, 8, 3, 2, 8)
    assert batch["conformer_graph_conformer_mask"][0].sum().item() == 2
    assert not batch["conformer_graph_conformer_mask"][1].any()
    assert batch["conformer_graph_site_mask"][0, :2, :, 0].all()
    assert not batch["conformer_graph_site_mask"][1].any()


def test_conformer_graph_heads_are_source_specific_and_missing_is_exact_fallback() -> None:
    config = {
        "model": {
            "hidden_dim": 16,
            "num_attention_heads": 4,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "structure_feature_dim": 16,
            "conformer_graph_set_branch": True,
            "conformer_graph_hidden_dim": 8,
            "conformer_set_attention_heads": 2,
            "dropout": 0.0,
            "conformer_graph_dropout": 0.0,
        }
    }
    model = GHXToxModel(config).eval()
    missing = _record("missing")
    missing["conformer_graph_expected"] = True
    pepflow = _record("pepflow", available=True)
    af2 = _record("af2", available=True)
    for record, source in ((pepflow, 0), (af2, 1)):
        record["conformer_source_features"] = torch.nn.functional.one_hot(
            torch.tensor(source), num_classes=3
        ).float()
        record["conformer_graph_expected"] = True
        record["conformer_graph_residue_coords"] = torch.zeros(2, 3, 3)
        record["conformer_graph_plddt"] = torch.ones(2, 3)
        record["conformer_graph_site_coords"] = torch.zeros(2, 3, 2, 3)
        record["conformer_graph_site_types"] = torch.zeros(2, 3, 2, 8)
        record["conformer_graph_site_types"][:, :, 0, 0] = 1.0
        record["conformer_graph_site_mask"] = torch.zeros(2, 3, 2, dtype=torch.bool)
        record["conformer_graph_site_mask"][:, :, 0] = True
        record["conformer_graph_conformer_mask"] = torch.ones(2, dtype=torch.bool)
    batch = collate_peptides([missing, pepflow, af2], include_atom=False)
    with torch.no_grad():
        initial = model(batch)
    assert torch.equal(initial["logits"], initial["base_logits"])
    with torch.no_grad():
        model.conformer_graph_source_heads[0][-1].bias.fill_(1.0)
    with torch.no_grad():
        changed = model(batch)
    assert torch.equal(changed["logits"][0], changed["base_logits"][0])
    assert not torch.equal(changed["logits"][1], changed["base_logits"][1])
    assert torch.equal(changed["logits"][2], changed["base_logits"][2])


def test_dense_chemical_graph_edge_bias_chunking_is_exact() -> None:
    torch.manual_seed(7)
    layer = DenseChemicalGraphLayer(hidden_dim=8, edge_dim=5, dropout=0.0).eval()
    hidden = torch.randn(2, 4, 8)
    edge_features = torch.randn(2, 4, 4, 5)
    edge_mask = ~torch.eye(4, dtype=torch.bool).unsqueeze(0).expand(2, -1, -1)
    node_mask = torch.ones(2, 4, dtype=torch.bool)

    layer.edge_bias_chunk_size = edge_features.numel()
    unchunked = layer(hidden, edge_features, edge_mask, node_mask)
    layer.edge_bias_chunk_size = 3
    chunked = layer(hidden, edge_features, edge_mask, node_mask)

    torch.testing.assert_close(chunked, unchunked, rtol=0.0, atol=0.0)
