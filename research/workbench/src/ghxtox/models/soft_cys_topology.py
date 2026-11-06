"""Global soft pairing and bounded compatible-topology message passing."""
from __future__ import annotations

import math
import torch
from torch import nn

from ghxtox.soft_cys_topology import PAIR_FEATURE_DIM, build_candidate_features, candidate_matchings


def soft_matching(scores: torch.Tensor, pair_mask: torch.Tensor,
                  free_scores: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric substochastic projection with explicit residual free capacity.

    Symmetric odds compete against both free states. Dividing an edge by the
    larger endpoint load guarantees degree <= 1 without breaking symmetry.
    This relaxation need not satisfy odd-set constraints; compatible discrete
    candidates supply that constraint in the aggregation variant.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    allowed = pair_mask & pair_mask.T
    allowed = allowed & ~torch.eye(len(scores), dtype=torch.bool, device=scores.device)
    symmetric = (scores + scores.T) / 2
    odds = torch.exp(((symmetric - free_scores[:, None] - free_scores[None, :])
                      / temperature).clamp(-30, 30)) * allowed
    load = 1 + odds.sum(-1)
    pairing = odds / torch.maximum(load[:, None], load[None, :])
    free = (1 - pairing.sum(-1)).clamp_min(0)
    return pairing, free


class SoftCysTopologyBranch(nn.Module):
    def __init__(self, hidden_dim: int, max_distance: float = 8.0,
                 max_candidates: int = 8, mode: str = "topology", temperature: float = 1.0):
        super().__init__()
        if mode not in {"local", "matching", "topology", "null"}:
            raise ValueError(f"Unknown topology mode: {mode}")
        if max_candidates < 2 or not math.isfinite(max_distance) or max_distance <= 0:
            raise ValueError("Need >=2 candidate slots and positive finite distance")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.max_distance, self.max_candidates = max_distance, max_candidates
        self.mode, self.temperature = mode, temperature
        self.pair_score = nn.Sequential(nn.Linear(2 * hidden_dim + PAIR_FEATURE_DIM, hidden_dim),
                                        nn.GELU(), nn.Linear(hidden_dim, 1))
        self.free_score = nn.Linear(hidden_dim, 1)
        self.message = nn.Sequential(nn.Linear(2 * hidden_dim + PAIR_FEATURE_DIM, hidden_dim), nn.GELU())
        self.topology_update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU())
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, spatial_h: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
        residuals, descriptors, evidence = [], [], []
        for b in range(spatial_h.shape[0]):
            length = int(batch["mask"][b].sum())
            cached = batch.get("soft_cys_cache")
            if cached is None:
                record = {key: value[b, :length] for key, value in batch.items()
                          if torch.is_tensor(value) and value.ndim >= 2 and value.shape[:2] == spatial_h.shape[:2]}
                data = build_candidate_features(record, self.max_distance)
            else:
                data = cached[b]
                if data["max_distance"] != self.max_distance or data["version"] != 1:
                    raise ValueError("Topology cache protocol mismatch")
            indices = data["indices"].to(spatial_h.device)
            features = data["pair_features"].to(spatial_h)
            allowed = data["pair_mask"].to(spatial_h.device)
            h = spatial_h[b, indices]
            n = len(indices)
            output = torch.zeros_like(spatial_h[b]) + self.projection.weight.sum() * 0
            entropy = spatial_h.new_zeros(())
            candidates, weights = [()], spatial_h.new_ones(1)
            if n == 0:
                p = spatial_h.new_zeros((0, 0))
                free = spatial_h.new_zeros(0)
            else:
                left, right = h[:, None, :].expand(n, n, -1), h[None, :, :].expand(n, n, -1)
                symmetric_input = torch.cat([left + right, (left - right).abs(), features], -1)
                scores = self.pair_score(symmetric_input).squeeze(-1)
                free_scores = self.free_score(h).squeeze(-1)
                p, free = soft_matching(scores, allowed, free_scores, self.temperature)
                messages = self.message(torch.cat([left, right, features], -1))
                if self.mode == "local":
                    p = torch.sigmoid(scores / self.temperature) * allowed
                    free = (1 - p.sum(-1)).clamp_min(0)
                if self.mode == "topology":
                    log_pair, log_free = p.clamp_min(1e-12).log(), free.clamp_min(1e-12).log()
                    candidates = candidate_matchings(log_pair, allowed, log_free, self.max_candidates)
                    adjacency, candidate_logits = [], []
                    for pairs in candidates:
                        a = torch.zeros_like(p)
                        used = set()
                        score = log_free.sum()
                        for i, j in pairs:
                            a[i, j] = a[j, i] = 1
                            used.update((i, j))
                            score = score + log_pair[i, j] - log_free[i] - log_free[j]
                        adjacency.append(a)
                        candidate_logits.append(score)
                    weights = torch.stack(candidate_logits).softmax(0)
                    adjacency = torch.stack(adjacency)
                    # Nonlinearity is applied per topology before weighted aggregation.
                    context = torch.einsum("kij,ijh->kih", adjacency, messages)
                    state = self.topology_update(torch.cat([h.expand(len(candidates), -1, -1), context], -1))
                    state = state * adjacency.sum(-1)[..., None]
                    update = (state * weights[:, None, None]).sum(0)
                    p = (adjacency * weights[:, None, None]).sum(0)
                    free = 1 - p.sum(-1)
                    entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
                else:
                    context = (p[..., None] * messages).sum(1)
                    update = self.topology_update(torch.cat([h, context], -1)) * p.sum(-1)[:, None]
                if self.mode == "null":
                    update = update * 0
                output = output.index_add(0, indices, self.projection(update))
            residuals.append(output)
            descriptors.append(torch.stack([p.sum(), entropy, free.sum()]))
            evidence.append({"residue_indices": indices, "pairing": p, "free": free,
                             "candidates": candidates, "weights": weights})
        return torch.stack(residuals), {
            "topology_descriptors": torch.stack(descriptors),
            "topology_evidence": evidence,
        }
