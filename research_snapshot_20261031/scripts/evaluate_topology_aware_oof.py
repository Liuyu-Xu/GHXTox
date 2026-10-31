"""Evaluate complete training OOF variants at a fixed threshold, with paired group bootstrap."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, matthews_corrcoef, roc_auc_score)

from ghxtox.topology_oof import sha256, validate_manifest


def metrics(y, p, threshold):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    predicted = p >= threshold
    return {"mcc": float(matthews_corrcoef(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def evaluate(rows, manifest, threshold=0.5, resamples=1000, seed=42, progress_every=0):
    if not 0 < threshold < 1 or resamples < 1:
        raise ValueError("Require fixed threshold in (0,1) and positive bootstrap count")
    ordered = sorted(manifest, key=lambda row: int(row["source_index"]))
    records = [{"sample_id": r["sample_id"], "sequence": r["sequence"], "label": int(r["label"])} for r in ordered]
    validate_manifest(records, manifest)
    n = len(records)
    variants = {"baseline", "local", "matching", "topology", "null"}
    indexed = {}
    for row in rows:
        i, variant = int(row["source_index"]), row["variant"]
        if variant not in variants or not 0 <= i < n or (variant, i) in indexed:
            raise ValueError("Unknown variant, duplicate or invalid OOF index")
        reference = ordered[i]
        if any(str(row[key]) != str(reference[key]) for key in ("sample_id", "sequence", "label", "fold")):
            raise ValueError("OOF identity mismatch")
        probability = float(row["probability"])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid probability")
        indexed[variant, i] = probability
    if len(indexed) != n * len(variants):
        raise ValueError("Incomplete five-fold OOF variant coverage")
    probabilities = {v: np.array([indexed[v, i] for i in range(n)]) for v in variants}
    if not np.allclose(probabilities["baseline"], probabilities["null"], atol=1e-7, rtol=1e-6):
        raise ValueError("Null variant must reproduce retained baseline within float32 tolerance")
    y = np.array([int(r["label"]) for r in ordered])
    if len(np.unique(y)) != 2:
        raise ValueError("OOF evaluation requires both labels")
    counts = np.array([r["sequence"].count("C") for r in ordered])
    if not np.allclose(probabilities["baseline"][counts < 2], probabilities["topology"][counts < 2], atol=1e-7, rtol=1e-6):
        raise ValueError("No-pair peptides must reproduce retained baseline within float32 tolerance")
    label_weights = [(y == label).mean() for label in (0, 1)]
    def subgroup(mask, p):
        errors = (p >= threshold) != y
        rates = [float(errors[mask & (y == label)].mean()) if (mask & (y == label)).any() else None for label in (0, 1)]
        return {"n": int(mask.sum()), "label_error_rates": rates,
                "label_standardized_error": sum(w * r for w, r in zip(label_weights, rates)) if all(r is not None for r in rates) else None}
    result = {"threshold": threshold, "metrics": {}, "subgroups": {}, "fold_metrics": {},
              "paired_group_bootstrap_delta_ci95": {}, "promotion": "not_evaluated_no_prespecified_promotion_rule"}
    result["null_max_probability_difference"] = float(np.max(np.abs(probabilities["baseline"] - probabilities["null"])))
    for variant, p in probabilities.items():
        result["metrics"][variant] = metrics(y, p, threshold)
        result["fold_metrics"][variant] = {str(fold): metrics(y[[int(r["fold"]) == fold for r in ordered]], p[[int(r["fold"]) == fold for r in ordered]], threshold) for fold in range(5)}
        result["subgroups"][variant] = {
            "cys_ge4": subgroup(counts >= 4, p), "cys_lt4": subgroup(counts < 4, p),
            "by_exact_cys_count": {str(c): subgroup(counts == c, p) for c in sorted(set(counts.tolist()))}}
        common_counts = [c for c in sorted(set(counts.tolist()))
                         if all(((counts == c) & (y == label)).any() for label in (0, 1))]
        stratum_sizes = np.array([(counts == c).sum() for c in common_counts], dtype=float)
        rates = [result["subgroups"][variant]["by_exact_cys_count"][str(c)]["label_standardized_error"]
                 for c in common_counts]
        result["subgroups"][variant]["count_and_label_standardized_error"] = {
            "value": float(np.average(rates, weights=stratum_sizes)) if rates else None,
            "common_support_cys_counts": common_counts,
            "covered_samples": int(stratum_sizes.sum()),
            "weighting": "shared empirical exact-Cys-count weights; shared full-cohort label weights"}
        # Same peptides and exact-Cys strata for every variant; no resampling imbalance.
    group_column = "group" if "group" in ordered[0] else "group_id"
    groups_by_fold = {}
    for fold in range(5):
        group_names = sorted({r[group_column] for r in ordered if int(r["fold"]) == fold})
        groups_by_fold[fold] = [np.array([i for i, r in enumerate(ordered) if r[group_column] == group]) for group in group_names]
    rng = np.random.default_rng(seed)
    delta = {v: {key: [] for key in result["metrics"]["baseline"]} for v in variants - {"baseline"}}
    standardized_delta = {v: {"cys_ge4": [], "cys_lt4": []} for v in delta}
    for iteration in range(resamples):
        selection = np.concatenate([group_list[k] for group_list in groups_by_fold.values()
                                    for k in rng.integers(0, len(group_list), len(group_list))])
        base = metrics(y[selection], probabilities["baseline"][selection], threshold)
        if base is None:
            continue
        for variant in delta:
            current = metrics(y[selection], probabilities[variant][selection], threshold)
            for key in base:
                delta[variant][key].append(current[key] - base[key])
            error_delta = ((probabilities[variant][selection] >= threshold) != y[selection]).astype(float)
            error_delta -= ((probabilities["baseline"][selection] >= threshold) != y[selection])
            for name, mask in (("cys_ge4", counts[selection] >= 4), ("cys_lt4", counts[selection] < 4)):
                strata = [mask & (y[selection] == label) for label in (0, 1)]
                if all(stratum.any() for stratum in strata):
                    standardized_delta[variant][name].append(sum(w * error_delta[s].mean() for w, s in zip(label_weights, strata)))
        if progress_every and ((iteration + 1) % progress_every == 0 or iteration + 1 == resamples):
            print(f"paired_group_bootstrap={iteration + 1}/{resamples}", flush=True)
    for variant, values in delta.items():
        result["paired_group_bootstrap_delta_ci95"][variant] = {
            key: {"interval": np.quantile(samples, [.025, .975]).tolist() if samples else None,
                  "valid_resamples": len(samples)} for key, samples in values.items()}
    result["paired_label_standardized_error_delta_ci95"] = {
        variant: {name: {"interval": np.quantile(samples, [.025, .975]).tolist() if samples else None,
                         "valid_resamples": len(samples)} for name, samples in groups.items()}
        for variant, groups in standardized_delta.items()}
    stability = [r for r in rows if r["variant"] == "topology"]
    result["stability"] = {key: float(np.mean([float(r[key]) for r in stability]))
                           if all(r.get(key) not in (None, "") for r in stability) else None
                           for key in ("probability_perturbation_mae", "pairing_perturbation_mae")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8-sig"))
    if protocol["scope"] != "training_oof_only" or sha256(protocol["fold_manifest"]) != protocol["fold_manifest_sha256"]:
        raise ValueError("Require frozen training-only manifest")
    with open(args.oof, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with open(protocol["fold_manifest"], encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    print(f"Evaluating {len(rows)} OOF rows with {protocol['bootstrap_resamples']} paired group bootstraps...", flush=True)
    result = evaluate(
        rows, manifest, protocol["threshold"], protocol["bootstrap_resamples"], protocol["seed"],
        progress_every=500,
    )
    result["input_sha256"] = {"oof": sha256(args.oof), "protocol": sha256(args.protocol)}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Evaluation saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
