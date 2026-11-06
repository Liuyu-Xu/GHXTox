"""Training-only feature-space augmentation for processed GHXTox datasets."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import torch


FLOAT_SEQUENCE_KEYS = {
    "residue_features",
    "coords",
    "plddt",
    "backbone_coords",
    "functional_group_coords",
    "structure_features",
    "plm_features",
    "chemical_site_coords",
    "chemical_site_types",
    "chemical_site_orientations",
}


def _descriptor(record: dict[str, Any]) -> torch.Tensor:
    parts = []
    plm = record.get("plm_features")
    if torch.is_tensor(plm) and plm.ndim == 2:
        parts.append(plm.float().mean(dim=0))
    residue = record.get("residue_features")
    if torch.is_tensor(residue) and residue.ndim == 2:
        parts.append(residue.float().mean(dim=0))
    structure = record.get("structure_features")
    if torch.is_tensor(structure) and structure.ndim == 2:
        parts.append(structure.float().mean(dim=0))
    if not parts:
        raise ValueError(f"Cannot build descriptor for {record.get('sample_id')}: no tensor features.")
    return torch.cat(parts)


def _normalize_descriptors(descriptors: torch.Tensor) -> torch.Tensor:
    mean = descriptors.mean(dim=0, keepdim=True)
    std = descriptors.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (descriptors - mean) / std


def _align_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(target)
    slices = tuple(slice(0, min(source.shape[dim], target.shape[dim])) for dim in range(target.ndim))
    out[slices] = source[slices].to(dtype=target.dtype)
    return out


def _mix_tensor(anchor: torch.Tensor, donor: torch.Tensor, lam: float, noise_std: float) -> torch.Tensor:
    donor = _align_like(donor, anchor)
    mixed = lam * anchor.float() + (1.0 - lam) * donor.float()
    if noise_std > 0.0:
        scale = mixed.detach().std().clamp_min(1e-6)
        mixed = mixed + torch.randn_like(mixed) * scale * noise_std
    return mixed.to(dtype=anchor.dtype)


def _synthetic_record(
    anchor: dict[str, Any],
    donor: dict[str, Any],
    sample_id: str,
    lam: float,
    noise_std: float,
) -> dict[str, Any]:
    item = copy.deepcopy(anchor)
    item["sample_id"] = sample_id
    item["original_id"] = f"augmented_from:{anchor.get('sample_id')}|{donor.get('sample_id')}"
    item["header"] = f"{sample_id}|{int(anchor['label'])}"
    item["augmentation"] = {
        "strategy": "boundary_classwise_feature_mixup",
        "anchor": anchor.get("sample_id"),
        "donor": donor.get("sample_id"),
        "lambda": float(lam),
        "feature_noise_std": float(noise_std),
    }
    for key in FLOAT_SEQUENCE_KEYS:
        anchor_value = anchor.get(key)
        donor_value = donor.get(key)
        if torch.is_tensor(anchor_value) and torch.is_tensor(donor_value) and anchor_value.dtype.is_floating_point:
            item[key] = _mix_tensor(anchor_value, donor_value, lam, noise_std)
    return item


def _class_indices(records: list[dict[str, Any]]) -> dict[int, list[int]]:
    classes: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        label = record.get("label")
        if label is None:
            raise ValueError("Augmentation requires labeled training records.")
        classes.setdefault(int(label), []).append(index)
    if len(classes) < 2:
        raise ValueError("Augmentation requires at least two classes.")
    return classes


def augment_records(
    records: list[dict[str, Any]],
    copies_per_record: float = 1.0,
    boundary_fraction: float = 0.6,
    k_neighbors: int = 8,
    lam_min: float = 0.25,
    lam_max: float = 0.75,
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if copies_per_record <= 0:
        raise ValueError("copies_per_record must be positive.")
    if not 0.0 < boundary_fraction <= 1.0:
        raise ValueError("boundary_fraction must be in (0, 1].")
    if not 0.0 <= lam_min <= lam_max <= 1.0:
        raise ValueError("lambda bounds must satisfy 0 <= lam_min <= lam_max <= 1.")
    rng = random.Random(seed)
    torch.manual_seed(seed)

    classes = _class_indices(records)
    descriptors = _normalize_descriptors(torch.stack([_descriptor(record) for record in records]))
    synthetic: list[dict[str, Any]] = []
    per_class_summary = {}

    for label, indices in sorted(classes.items()):
        other_indices = [index for other_label, items in classes.items() if other_label != label for index in items]
        class_desc = descriptors[indices]
        other_desc = descriptors[other_indices]
        opposite_distance = torch.cdist(class_desc, other_desc).min(dim=1).values
        boundary_count = max(1, int(round(len(indices) * boundary_fraction)))
        boundary_order = opposite_distance.argsort()[:boundary_count].tolist()
        boundary_indices = [indices[position] for position in boundary_order]

        same_distance = torch.cdist(class_desc, class_desc)
        same_distance.fill_diagonal_(float("inf"))
        neighbor_count = max(1, min(int(k_neighbors), len(indices) - 1))
        same_neighbors = same_distance.topk(k=neighbor_count, largest=False).indices

        target_count = int(round(len(indices) * copies_per_record))
        for aug_index in range(target_count):
            anchor_index = boundary_indices[aug_index % len(boundary_indices)]
            class_position = indices.index(anchor_index)
            donor_position = int(same_neighbors[class_position, rng.randrange(neighbor_count)])
            donor_index = indices[donor_position]
            lam = rng.uniform(lam_min, lam_max)
            sample_id = f"aug_{label}_{aug_index + 1:06d}_{records[anchor_index]['sample_id']}"
            synthetic.append(
                _synthetic_record(
                    records[anchor_index],
                    records[donor_index],
                    sample_id=sample_id,
                    lam=lam,
                    noise_std=noise_std,
                )
            )

        per_class_summary[str(label)] = {
            "original": len(indices),
            "synthetic": target_count,
            "boundary_pool": len(boundary_indices),
        }

    return records + synthetic, {
        "strategy": "boundary_classwise_feature_mixup",
        "copies_per_record": float(copies_per_record),
        "boundary_fraction": float(boundary_fraction),
        "k_neighbors": int(k_neighbors),
        "lambda_range": [float(lam_min), float(lam_max)],
        "feature_noise_std": float(noise_std),
        "seed": int(seed),
        "classes": per_class_summary,
        "original_records": len(records),
        "synthetic_records": len(synthetic),
        "total_records": len(records) + len(synthetic),
    }


def augment_processed(
    input_path: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    augmented_records, summary = augment_records(records, **kwargs)
    output_payload = dict(payload)
    output_payload["records"] = augmented_records
    output_payload["augmentation"] = summary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input labeled processed .pt file.")
    parser.add_argument("--output", required=True, help="Output augmented processed .pt file.")
    parser.add_argument("--copies-per-record", type=float, default=1.0)
    parser.add_argument("--boundary-fraction", type=float, default=0.6)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--lambda-min", type=float, default=0.25)
    parser.add_argument("--lambda-max", type=float, default=0.75)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = augment_processed(
        args.input,
        args.output,
        copies_per_record=args.copies_per_record,
        boundary_fraction=args.boundary_fraction,
        k_neighbors=args.k_neighbors,
        lam_min=args.lambda_min,
        lam_max=args.lambda_max,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()
