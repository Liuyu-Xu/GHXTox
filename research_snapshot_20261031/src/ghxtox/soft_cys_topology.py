"""Deterministic, label-free candidate Cys geometry (not native bond labels)."""
from __future__ import annotations

import hashlib
import math

import torch

from ghxtox.chemical_sites import CHEMICAL_SITE_TYPE_TO_INDEX, CHEMICAL_SITE_TYPE_DIM
from ghxtox.constants import AA_TO_IDX

FEATURE_VERSION = 1
PAIR_FEATURE_DIM = 10 + CHEMICAL_SITE_TYPE_DIM
GEOMETRY_KEYS = (
    "aa_ids", "coords", "plddt", "chemical_site_coords", "chemical_site_types",
    "chemical_site_orientations", "chemical_site_orientation_mask", "chemical_site_mask",
)


def feature_signature(record: dict, max_distance: float) -> str:
    digest = hashlib.sha256(f"{FEATURE_VERSION}:{max_distance}".encode())
    for key in GEOMETRY_KEYS + (("mask",) if "mask" in record else ()):
        value = record[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_candidate_features(record: dict, max_distance: float = 8.0) -> dict:
    """Build compact Cys tensors; input pLDDT must use the existing [0,1] contract.

    Missing/nonfinite geometry excludes a pair, but retains the Cys free state.
    Signed CB-SG pair alignments are symmetrized; no coordinates enter the MLP.
    """
    if not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be finite and positive")
    missing = [key for key in GEOMETRY_KEYS if key not in record]
    if missing:
        raise ValueError(f"Topology requires chemical-site cache fields: {missing}")
    aa = record["aa_ids"]
    residue_mask = record.get("mask", torch.ones_like(aa, dtype=torch.bool))
    indices = torch.where((aa == AA_TO_IDX["C"]) & residue_mask)[0]
    coords = record["coords"].float()
    confidence = record["plddt"].float()
    valid_conf = torch.isfinite(confidence) & (confidence >= 0) & (confidence <= 1)
    if ((~valid_conf) & residue_mask).any():
        raise ValueError("Topology pLDDT must be finite in [0,1]; normalize upstream")
    sites = record["chemical_site_coords"].float()
    types = record["chemical_site_types"].float()
    site_valid = record["chemical_site_mask"].bool() & torch.isfinite(sites).all(-1)
    site_valid &= torch.isfinite(types).all(-1)
    sulfur = site_valid & (types[..., CHEMICAL_SITE_TYPE_TO_INDEX["sulfur"]] > 0)
    slot = sulfur.long().argmax(-1)
    rows = torch.arange(len(aa), device=aa.device)
    sg = torch.nan_to_num(sites[rows, slot])[indices]
    ca = torch.nan_to_num(coords)[indices]
    valid = sulfur.any(-1)[indices] & torch.isfinite(coords).all(-1)[indices]
    orientation = record["chemical_site_orientations"].float()[rows, slot][indices]
    orient_valid = record["chemical_site_orientation_mask"].bool()[rows, slot][indices]
    orient_valid &= torch.isfinite(orientation).all(-1) & (orientation.norm(dim=-1) > 1e-6)
    orientation = torch.nan_to_num(orientation)
    orientation = orientation / orientation.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    orientation = orientation * orient_valid[:, None]
    delta = sg[None, :, :] - sg[:, None, :]
    distance = delta.norm(dim=-1)
    direction = delta / distance[..., None].clamp_min(1e-6)
    align = (orientation[:, None, :] * direction).sum(-1)
    mutual = orientation @ orientation.T
    both_orient = orient_valid[:, None] & orient_valid[None, :]
    # A 1e-5 A tolerance keeps rigid transforms stable at the floating-point cutoff.
    pair_mask = valid[:, None] & valid[None, :] & (distance > 1e-6) & (distance <= max_distance + 1e-5)
    pair_mask.fill_diagonal_(False)
    conf = confidence[indices]
    # Smooth local chemical-type density within 8A, excluding the same residue.
    flat_sites = torch.nan_to_num(sites).reshape(-1, 3)
    local_dist = torch.cdist(sg, flat_sites)
    local_mask = site_valid.reshape(-1)[None, :] & (
        indices[:, None] != rows.repeat_interleave(sites.shape[1])[None, :]
    )
    local_weight = torch.exp(-local_dist / 8.0) * local_mask
    context = local_weight @ torch.nan_to_num(types).reshape(-1, types.shape[-1])
    context = context / local_weight.sum(-1, keepdim=True).clamp_min(1)
    scalars = [distance / max_distance, torch.cdist(ca, ca) / 24.0,
               (indices[:, None] - indices[None, :]).abs().float() / max(len(aa), 1),
               torch.minimum(conf[:, None], conf[None, :]),
               (conf[:, None] + conf[None, :]) / 2,
               (align + align.T) / 2, (align - align.T).abs(),
               mutual * both_orient, both_orient.float(),
               (orient_valid[:, None].float() + orient_valid[None, :].float()) / 2]
    features = torch.cat([torch.stack(scalars, -1),
                          (context[:, None, :] + context[None, :, :]) / 2], -1)
    features = features.masked_fill(~pair_mask[..., None], 0)
    return {"indices": indices, "pair_features": features, "pair_mask": pair_mask,
            "max_distance": float(max_distance), "version": FEATURE_VERSION}


def cached_candidate_features(record: dict, max_distance: float = 8.0) -> dict:
    """Explicit cache validity includes all geometry, masks and feature protocol."""
    signature = feature_signature(record, max_distance)
    cached = record.get("soft_cys_cache")
    if cached is not None and cached.get("signature") == signature:
        return cached
    return {**build_candidate_features(record, max_distance), "signature": signature}


def candidate_matchings(scores: torch.Tensor, pair_mask: torch.Tensor, free_scores: torch.Tensor,
                        max_candidates: int = 8) -> list[tuple[tuple[int, int], ...]]:
    """Bounded deterministic beam over disjoint pairs, always retaining all-free.

    Selection is discrete/detached; selected candidate weights remain differentiable
    in the encoder. This is a bounded approximation, not exhaustive top-k matching.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    n = len(free_scores)
    values, allowed, free = scores.detach().cpu(), pair_mask.detach().cpu(), free_scores.detach().cpu()
    states = [(0.0, (), frozenset())]
    for i in range(n):
        expanded = []
        for value, pairs, used in states:
            if i in used:
                expanded.append((value, pairs, used))
                continue
            expanded.append((value + float(free[i]), pairs, used | {i}))
            for j in range(i + 1, n):
                if j not in used and allowed[i, j]:
                    expanded.append((value + float(values[i, j]), pairs + ((i, j),), used | {i, j}))
        states = sorted(expanded, key=lambda item: (-item[0], item[1]))[:max_candidates]
    selected = [state[1] for state in states if state[1]]
    return [()] + selected[:max_candidates - 1]
