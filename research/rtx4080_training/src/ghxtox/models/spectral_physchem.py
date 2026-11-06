"""Deterministic multi-channel DCT descriptors for peptide sequences.

This is a lightweight sequence branch: it has no learned residue embedding,
no structure input, and no dependence on the frozen ProtT5 cache values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


SPECTRAL_FEATURE_VERSION = "physchem-dct-v1"
SPECTRAL_CHANNEL_NAMES = ("hydropathy", "charge_ph7", "polarity", "side_chain_volume", "flexibility", "aromaticity")
SPECTRAL_DCT_COEFFICIENTS = 8

_HYDROPATHY = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3}
_CHARGE = {"A": 0.0, "C": -0.1, "D": -1.0, "E": -1.0, "F": 0.0, "G": 0.0, "H": 0.1, "I": 0.0, "K": 1.0, "L": 0.0, "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0, "R": 1.0, "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0, "Y": -0.1}
_POLARITY = {"A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3, "F": 5.2, "G": 9.0, "H": 10.4, "I": 5.2, "K": 11.3, "L": 4.9, "M": 5.7, "N": 11.6, "P": 8.0, "Q": 10.5, "R": 10.5, "S": 9.2, "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2}
_VOLUME = {"A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6}
_FLEXIBILITY = {"A": 0.36, "C": 0.35, "D": 0.51, "E": 0.50, "F": 0.31, "G": 0.54, "H": 0.32, "I": 0.46, "K": 0.47, "L": 0.37, "M": 0.30, "N": 0.46, "P": 0.51, "Q": 0.49, "R": 0.53, "S": 0.51, "T": 0.44, "V": 0.39, "W": 0.31, "Y": 0.42}
_AROMATIC = {residue: float(residue in "FWY") for residue in _HYDROPATHY}
_CHANNELS = (_HYDROPATHY, _CHARGE, _POLARITY, _VOLUME, _FLEXIBILITY, _AROMATIC)

SPECTRAL_FEATURE_NAMES = tuple(name for channel in SPECTRAL_CHANNEL_NAMES for name in (tuple(f"{channel}_dct_{index}" for index in range(SPECTRAL_DCT_COEFFICIENTS)) + tuple(f"{channel}_energy_{band}" for band in ("low", "mid", "high"))))


def _dct2(values: torch.Tensor) -> torch.Tensor:
    """Orthonormal DCT-II for a one-dimensional real-valued sequence."""
    size = int(values.numel())
    if size == 0:
        raise ValueError("DCT input cannot be empty")
    positions = torch.arange(size, dtype=torch.float64)
    frequencies = torch.arange(size, dtype=torch.float64).unsqueeze(1)
    basis = torch.cos(math.pi / size * (positions + 0.5) * frequencies)
    coefficients = basis @ values.to(dtype=torch.float64)
    coefficients[0] *= math.sqrt(1.0 / size)
    if size > 1:
        coefficients[1:] *= math.sqrt(2.0 / size)
    return coefficients


def _channel_features(values: torch.Tensor) -> torch.Tensor:
    centered = values.to(dtype=torch.float64) - values.to(dtype=torch.float64).mean()
    scale = centered.pow(2).mean().sqrt()
    standardized = centered / scale if float(scale) > 1e-12 else torch.zeros_like(centered)
    spectrum = _dct2(standardized)
    leading = torch.zeros(SPECTRAL_DCT_COEFFICIENTS, dtype=torch.float64)
    leading[: min(SPECTRAL_DCT_COEFFICIENTS, len(spectrum))] = spectrum[:SPECTRAL_DCT_COEFFICIENTS]
    energy = spectrum[1:].pow(2)
    if energy.numel() == 0 or float(energy.sum()) <= 1e-12:
        fractions = torch.zeros(3, dtype=torch.float64)
    else:
        frequencies = torch.arange(1, len(spectrum), dtype=torch.float64) / max(len(spectrum), 1)
        total = energy.sum()
        fractions = torch.stack((energy[frequencies <= 1 / 3].sum() / total, energy[(frequencies > 1 / 3) & (frequencies <= 2 / 3)].sum() / total, energy[frequencies > 2 / 3].sum() / total))
    return torch.cat((leading, fractions)).to(dtype=torch.float32)


def spectral_physchem_feature_vector(sequence: str) -> torch.Tensor:
    cleaned = "".join(sequence.upper().split())
    if not cleaned:
        raise ValueError("Cannot calculate spectral descriptors for an empty sequence")
    unknown = sorted(set(cleaned) - set(_HYDROPATHY))
    if unknown:
        raise ValueError(f"Sequence contains unsupported residues: {', '.join(unknown)}")
    return torch.cat([_channel_features(torch.tensor([table[residue] for residue in cleaned])) for table in _CHANNELS])


def spectral_physchem_features(sequences: Sequence[str], *, device: torch.device | None = None) -> torch.Tensor:
    return torch.stack([spectral_physchem_feature_vector(sequence) for sequence in sequences]).to(device=device)
