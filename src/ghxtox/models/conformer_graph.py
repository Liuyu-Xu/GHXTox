"""Per-conformer chemical-site graphs with set-attention ensemble pooling."""

from __future__ import annotations

import math

import torch
from torch import nn

from ghxtox.chemical_sites import CHEMICAL_SITE_TYPE_DIM, CHEMICAL_SITE_TYPE_TO_INDEX
from ghxtox.models.layers import RadialBasis


class DenseChemicalGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(hidden)
        key = self.key(hidden)
        scores = torch.einsum("bih,bjh->bij", query, key) / math.sqrt(hidden.shape[-1])
        scores = scores + self.edge_bias(edge_features).squeeze(-1)
        scores = scores.masked_fill(~edge_mask, -1e4)
        weights = torch.softmax(scores, dim=-1) * edge_mask.to(dtype=hidden.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aggregate = torch.bmm(weights, self.value(hidden))
        updated = self.norm(hidden + self.update(torch.cat([hidden, aggregate], dim=-1)))
        return updated * node_mask.unsqueeze(-1).to(dtype=hidden.dtype)


class ConformerChemicalSetEncoder(nn.Module):
    """Encode each conformer's functional-group graph, then pool the unordered set."""

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        raw_rbf_bins: int = 8,
        normalized_rbf_bins: int = 8,
        max_distance: float = 16.0,
        normalized_max_distance: float = 4.0,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.max_distance = float(max_distance)
        self.raw_rbf = RadialBasis(raw_rbf_bins, max_distance)
        self.normalized_rbf = RadialBasis(
            normalized_rbf_bins, normalized_max_distance
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(CHEMICAL_SITE_TYPE_DIM + 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        edge_dim = raw_rbf_bins + normalized_rbf_bins + 5
        self.layers = nn.ModuleList(
            DenseChemicalGraphLayer(hidden_dim, edge_dim, dropout)
            for _ in range(num_layers)
        )
        self.conformer_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.set_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.set_query, std=0.02)
        self.set_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _compatibility(types: torch.Tensor) -> torch.Tensor:
        def channel(name: str) -> torch.Tensor:
            return types[..., CHEMICAL_SITE_TYPE_TO_INDEX[name]]

        positive, negative = channel("positive"), channel("negative")
        donor, acceptor = channel("donor"), channel("acceptor")
        aromatic = channel("aromatic")
        sulfur = channel("sulfur")
        hydrophobic = channel("hydrophobic")
        return torch.stack(
            [
                positive[:, :, None] * negative[:, None, :]
                + negative[:, :, None] * positive[:, None, :],
                donor[:, :, None] * acceptor[:, None, :]
                + acceptor[:, :, None] * donor[:, None, :],
                aromatic[:, :, None] * aromatic[:, None, :],
                sulfur[:, :, None] * sulfur[:, None, :],
                hydrophobic[:, :, None] * hydrophobic[:, None, :],
            ],
            dim=-1,
        )

    def forward(
        self,
        site_coords: torch.Tensor,
        site_types: torch.Tensor,
        site_mask: torch.Tensor,
        residue_coords: torch.Tensor,
        residue_mask: torch.Tensor,
        plddt: torch.Tensor,
        conformer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, conformers, length, site_slots, _ = site_coords.shape
        graph_count = batch_size * conformers
        nodes = length * site_slots
        coordinates = site_coords.reshape(graph_count, nodes, 3)
        types = site_types.reshape(graph_count, nodes, site_types.shape[-1])
        node_mask = site_mask.reshape(graph_count, nodes).bool()
        confidence = (
            plddt.unsqueeze(-1)
            .expand(-1, -1, -1, site_slots)
            .reshape(graph_count, nodes)
            .clamp(0.0, 1.0)
        )
        residue_valid = (
            residue_mask[:, None, :]
            .expand(-1, conformers, -1)
            .reshape(graph_count, length)
            .bool()
        )
        residue_coordinates = residue_coords.reshape(graph_count, length, 3)
        lengths = residue_valid.sum(dim=1).clamp_min(1)
        first_anchor = residue_coordinates[:, 0]
        last_indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, 3)
        last_anchor = residue_coordinates.gather(1, last_indices).squeeze(1)
        centroid = (
            residue_coordinates
            * residue_valid.unsqueeze(-1).to(dtype=residue_coordinates.dtype)
        ).sum(dim=1) / lengths.unsqueeze(-1)
        centered = residue_coordinates - centroid.unsqueeze(1)
        radius = (
            (
                centered.square().sum(dim=-1)
                * residue_valid.to(dtype=centered.dtype)
            ).sum(dim=1)
            / lengths
        ).clamp_min(1e-6).sqrt()
        anchors = torch.stack([first_anchor, centroid, last_anchor], dim=1)
        anchor_distances = torch.linalg.vector_norm(
            coordinates[:, :, None, :] - anchors[:, None, :, :], dim=-1
        ) / radius[:, None, None].clamp_min(1e-3)
        hidden = self.node_encoder(
            torch.cat(
                [types, confidence.unsqueeze(-1), anchor_distances.clamp_max(6.0) / 6.0],
                dim=-1,
            )
        )
        hidden = hidden * node_mask.unsqueeze(-1).to(dtype=hidden.dtype)

        distances = torch.cdist(coordinates, coordinates)
        normalized_distances = distances / radius[:, None, None].clamp_min(1e-3)
        edge_features = torch.cat(
            [
                self.raw_rbf(distances),
                self.normalized_rbf(normalized_distances),
                self._compatibility(types),
            ],
            dim=-1,
        )
        eye = torch.eye(nodes, device=distances.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = (
            node_mask[:, :, None]
            & node_mask[:, None, :]
            & ~eye
            & (distances <= self.max_distance)
        )
        for layer in self.layers:
            hidden = layer(hidden, edge_features, edge_mask, node_mask)

        valid = node_mask.unsqueeze(-1)
        mean = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        maximum = hidden.masked_fill(~valid, -1e4).max(dim=1).values
        maximum = torch.where(node_mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        mean_confidence = (
            confidence * node_mask.to(dtype=confidence.dtype)
        ).sum(dim=1) / node_mask.sum(dim=1).clamp_min(1)
        valid_fraction = node_mask.float().mean(dim=1)
        conformer_vectors = self.conformer_projection(
            torch.cat(
                [mean, maximum, mean_confidence.unsqueeze(-1), valid_fraction.unsqueeze(-1)],
                dim=-1,
            )
        ).reshape(batch_size, conformers, -1)

        safe_mask = conformer_mask.bool().clone()
        no_conformers = ~safe_mask.any(dim=1)
        safe_mask[no_conformers, 0] = True
        conformer_vectors = conformer_vectors * conformer_mask.unsqueeze(-1).to(
            dtype=conformer_vectors.dtype
        )
        query = self.set_query.expand(batch_size, -1, -1)
        pooled, weights = self.set_attention(
            query,
            conformer_vectors,
            conformer_vectors,
            key_padding_mask=~safe_mask,
            need_weights=True,
        )
        pooled = self.output_norm(pooled.squeeze(1))
        pooled = pooled * (~no_conformers).unsqueeze(-1).to(dtype=pooled.dtype)
        attention = weights.squeeze(1) * conformer_mask.to(dtype=weights.dtype)
        return pooled, {
            "conformer_set_attention": attention,
            "conformer_graph_count": conformer_mask.sum(dim=1).float(),
            "conformer_site_count": site_mask.sum(dim=(1, 2, 3)).float(),
        }
