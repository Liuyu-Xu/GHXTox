"""Generate auditable, non-eligible provenance for historical outer-fold-selected bases.

The artifacts this script describes were selected with outer-held-fold metrics.
It deliberately records that selection exposure, rather than manufacturing a
strict-OOF eligibility claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path


DATASET = "D:/dachuang2026/ziliao/train_chemical_sites_final_esm2_multisource_1_30_full2795.pt"
FOLDS = "D:/dachuang2026/ziliao/train_cdhit080_fallback_folds.csv"
PACKAGE = "D:/dachuang2026/ziliao/original_lineage_residual_package_20260809_v1"
CHECKPOINT_ROOT = "D:/dachuang2026/ziliao/fusion_aware_residual_screen_package_20260809_v1/repository/runs"
CHECKPOINT_NAMES = {
    0: "original_fold0_candidate_reproduction_torch231_v1",
    1: "original_lineage_candidate_fold1_torch231_v1",
    2: "original_lineage_base_fold2_zero_chemical_fallback_v1",
    3: "original_lineage_candidate_fold3_torch231_v1",
    4: "original_lineage_candidate_fold4_torch231_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"source_index", "sample_id", "label", "sequence", "group_id", "fold"}
    if not rows or not expected.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(expected)}")
    indexes = [int(row["source_index"]) for row in rows]
    if set(indexes) != set(range(len(rows))):
        raise ValueError("Manifest source_index values must be contiguous and unique")
    group_folds: dict[str, set[str]] = {}
    for row in rows:
        if row["label"] not in {"0", "1"} or row["fold"] not in {"0", "1", "2", "3", "4"}:
            raise ValueError("Manifest label or fold outside expected domain")
        group_folds.setdefault(row["group_id"], set()).add(row["fold"])
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("A group spans outer folds")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="configs/provenance/historical_selected_base_v1")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--fold-manifest", default=FOLDS)
    parser.add_argument("--package", default=PACKAGE)
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing provenance directory: {output}")
    dataset, folds, package, checkpoint_root = map(Path, (args.dataset, args.fold_manifest, args.package, args.checkpoint_root))
    for path in (dataset, folds, package, checkpoint_root):
        if not path.exists():
            raise FileNotFoundError(path)
    rows = read_manifest(folds)
    assignments = {int(row["source_index"]): int(row["fold"]) for row in rows}
    output.mkdir(parents=True, exist_ok=False)
    evidence_root = package / "evidence" / "original_lineage_base_cv_audit_v1" / "summary.json"
    selection = package / "configs" / "original_lineage_base_selection_v1.json"
    if not evidence_root.exists() or not selection.exists():
        raise FileNotFoundError("Missing selected-base audit evidence")
    common = {
        "artifact_type": "ghxtox_historical_selected_base_provenance",
        "version": 1,
        "generated_on": date.today().isoformat(),
        "dataset": {"path": str(dataset), "sha256": sha256(dataset), "record_count_from_manifest": len(rows)},
        "fold_manifest": {"path": str(folds), "sha256": sha256(folds),
                          "schema": ["source_index", "sample_id", "label", "sequence", "group_id", "fold"],
                          "group_disjoint_across_outer_folds": True,
                          "fold_counts": {str(k): v for k, v in sorted(Counter(assignments.values()).items())}},
        "external_test_labels_used": False,
        "external_test_evidence": {"path": str(evidence_root), "sha256": sha256(evidence_root),
                                     "field": "protocol.external_test_sets_used", "value": False},
        "selection_artifact": {"path": str(selection), "sha256": sha256(selection)},
        "eligibility": {
            "eligible_for_strict_outer_fold_topology_oof": False,
            "reason": "Historical selected-base checkpoint selection used its outer held fold; a topology branch initialized from it would inherit held-label selection.",
            "required_remedy": "Train an independently selected base within each outer-training partition, then use only that fold-specific base to fit the topology branch."
        },
        "all_supervised_ancestors_included": False,
        "ancestor_completeness_reason": "The archived records establish selected checkpoint identity and outer-fold selection, but do not supply complete auditable lineage for every upstream warm-start checkpoint."
    }
    entries = []
    for fold, name in CHECKPOINT_NAMES.items():
        checkpoint = checkpoint_root / name / "best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        held = [index for index, assigned in assignments.items() if assigned == fold]
        fitting = [index for index, assigned in assignments.items() if assigned != fold]
        provenance = {
            **common,
            "fold": fold,
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "selection_role": "historical_selected_base"},
            "checkpoint_sha256": sha256(checkpoint),
            "training_data_sha256": common["dataset"]["sha256"],
            "fold_manifest_sha256": common["fold_manifest"]["sha256"],
            "fitting_indices": fitting,
            "selection_indices": held,
            "selection_indices_status": "outer_held_fold_exposure_confirmed",
            "selection_evidence": {
                "path": str(package / "training_records" / "selected_bases" / f"fold{fold}" / "config.json"),
                "observed_contract": "selection_mode=pareto with the current outer fold supplied as validation; historical history reports validation metrics by epoch."
            },
            "runner_compatibility": {
                "expected_topology_oof_preflight": "reject",
                "rejection_fields": ["selection_indices", "all_supervised_ancestors_included"]
            }
        }
        path = output / f"fold{fold}_historical_selected_base.provenance.json"
        path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        entries.append({"fold": fold, "provenance": path.name, "sha256": sha256(path),
                        "checkpoint": str(checkpoint), "checkpoint_sha256": provenance["checkpoint"]["sha256"],
                        "strict_topology_oof_eligible": False})
    index = {"artifact_type": "ghxtox_historical_selected_base_provenance_index", "version": 1,
             "generated_on": date.today().isoformat(), "entries": entries,
             "conclusion": "Five historical base checkpoints are fully identified for audit but are not valid frozen initializations for strict topology OOF."}
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} non-eligible historical provenance files to {output}")


if __name__ == "__main__":
    main()
