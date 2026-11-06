"""Deterministic sequence physicochemical features used by ProtTox-TSK.

The values are model inputs, not mechanistic explanations or toxicity evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


PHYS_CHEM_FEATURE_NAMES = (
    "length",
    "net_charge_ph7",
    "mean_kyte_doolittle_hydropathy",
    "basic_fraction",
    "hydrophobic_fraction",
    "aromatic_fraction",
    "cysteine_count",
    "cysteine_fraction",
    "minimum_cysteine_spacing",
)

_HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3,
}
_BASIC = frozenset("KRH")
_HYDROPHOBIC = frozenset("AVILMFWY")
_AROMATIC = frozenset("FWY")


def _net_charge_ph7(sequence: str) -> float:
    """Henderson--Hasselbalch net charge estimate at pH 7.0."""
    protonated = lambda pka: 1.0 / (1.0 + 10.0 ** (7.0 - pka))
    deprotonated = lambda pka: 1.0 / (1.0 + 10.0 ** (pka - 7.0))
    count = lambda residues: sum(residue in residues for residue in sequence)
    return (
        protonated(9.69)
        + count("K") * protonated(10.5)
        + count("R") * protonated(12.4)
        + count("H") * protonated(6.0)
        - deprotonated(2.34)
        - count("D") * deprotonated(3.86)
        - count("E") * deprotonated(4.25)
        - count("C") * deprotonated(8.33)
        - count("Y") * deprotonated(10.07)
    )


def physchem_feature_vector(sequence: str) -> list[float]:
    """Return the fixed nine-feature ProtTox-TSK descriptor vector."""
    cleaned = "".join(sequence.upper().split())
    if not cleaned:
        raise ValueError("Cannot calculate physicochemical features for an empty sequence")
    unknown = sorted(set(cleaned) - set(_HYDROPATHY))
    if unknown:
        raise ValueError(f"Sequence contains unsupported residues: {', '.join(unknown)}")
    length = len(cleaned)
    count = lambda residues: sum(residue in residues for residue in cleaned)
    cys_positions = [index for index, residue in enumerate(cleaned) if residue == "C"]
    spacing = min((right - left for left, right in zip(cys_positions, cys_positions[1:])), default=0)
    return [
        float(length),
        float(_net_charge_ph7(cleaned)),
        sum(_HYDROPATHY[residue] for residue in cleaned) / length,
        count(_BASIC) / length,
        count(_HYDROPHOBIC) / length,
        count(_AROMATIC) / length,
        float(len(cys_positions)),
        len(cys_positions) / length,
        float(spacing),
    ]


def physchem_features(sequences: Sequence[str], *, device: torch.device | None = None) -> torch.Tensor:
    """Vectorize :func:`physchem_feature_vector` in a stable feature order."""
    return torch.tensor([physchem_feature_vector(sequence) for sequence in sequences], dtype=torch.float32, device=device)
