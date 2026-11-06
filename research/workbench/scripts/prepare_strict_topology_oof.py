"""Validate a frozen protocol and create base provenance/final topology protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghxtox.nested_folds import load_nested_indices
from ghxtox.topology_oof import sha256, validate_checkpoint_provenance


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def assert_hash(path: str | Path, expected: str, label: str) -> None:
    actual = sha256(path).upper()
    if actual != expected.upper():
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def assert_frozen_values(expected: object, actual: object, path: str = "config") -> None:
    """Require every user-frozen JSON value to survive runtime config expansion."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{path} was not preserved as an object")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{path}.{key} is absent from the runtime config")
            assert_frozen_values(value, actual[key], f"{path}.{key}")
    elif expected != actual:
        raise ValueError(f"{path} changed between the frozen and runtime configs: {actual!r} != {expected!r}")


def verify_inputs(protocol: dict, processed: Path, outer_manifest: Path, nested: Path) -> None:
    inputs = protocol["inputs"]
    assert_hash(processed, inputs["processed_train_sha256"], "processed training cache")
    assert_hash(outer_manifest, inputs["outer_fold_manifest_sha256"], "outer-fold manifest")
    assert_hash(nested, inputs["nested_manifest_sha256"], "nested manifest")
    summary = nested.with_suffix(".json")
    assert_hash(summary, inputs["nested_summary_sha256"], "nested summary")
    for fold in inputs["outer_folds"]:
        roles = load_nested_indices(nested, fold, int(inputs["sample_count"]))
        if set(roles["train"]) & (set(roles["validation"]) | set(roles["calibration"]) | set(roles["test"])):
            raise ValueError(f"Nested training leakage in outer fold {fold}")


def write_base_provenance(args: argparse.Namespace) -> None:
    protocol, processed, outer, nested = map(Path, (args.protocol, args.processed, args.outer_manifest, args.nested_manifest))
    frozen = load_json(protocol)
    if frozen.get("status") != "frozen_before_base_training":
        raise ValueError("Protocol is not frozen for base training")
    verify_inputs(frozen, processed, outer, nested)
    base_dir = Path(args.base_dir)
    checkpoint, runtime_config_path, history = base_dir / "best_model.pt", base_dir / "config.json", base_dir / "history.json"
    frozen_config_path = Path(args.frozen_base_config)
    for path in (checkpoint, runtime_config_path, history, frozen_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    assert_hash(frozen_config_path, frozen["base_training"]["config_sha256"], "frozen base training config")
    frozen_config, config = load_json(frozen_config_path), load_json(runtime_config_path)
    assert_frozen_values(frozen_config, config)
    history_doc = load_json(history)
    if config["train"].get("selection_mode") != "fixed_epoch" or config["train"].get("epochs") != frozen["base_training"]["epochs"]:
        raise ValueError("Base checkpoint was not trained with frozen fixed-epoch protocol")
    if "initial_checkpoint" in config["train"] or "trainable_prefixes" in config["train"]:
        raise ValueError("Strict base must be scratch-trained with every base parameter trainable")
    if len(history_doc.get("history", [])) != frozen["base_training"]["epochs"]:
        raise ValueError("Base history does not contain the frozen number of epochs")
    roles = load_nested_indices(nested, args.fold, int(frozen["inputs"]["sample_count"]))
    provenance = {
        "artifact_type": "ghxtox_strict_nested_scratch_base_provenance", "version": 1, "fold": args.fold,
        "checkpoint_sha256": sha256(checkpoint), "training_data_sha256": sha256(processed),
        "fold_manifest_sha256": sha256(outer), "nested_manifest_sha256": sha256(nested),
        "checkpoint": str(checkpoint.resolve()), "processed_train": str(processed.resolve()),
        "outer_fold_manifest": str(outer.resolve()), "nested_manifest": str(nested.resolve()),
        "base_config": str(frozen_config_path.resolve()), "base_config_sha256": sha256(frozen_config_path),
        "runtime_config": str(runtime_config_path.resolve()), "runtime_config_sha256": sha256(runtime_config_path),
        "history": str(history.resolve()), "history_sha256": sha256(history),
        "initialization": "scratch", "supervised_ancestor_lineage": [],
        "all_supervised_ancestors_included": True, "external_test_labels_used": False,
        "fitting_indices": roles["train"], "selection_indices": [],
        "diagnostic_validation_indices": roles["validation"], "unused_calibration_indices": roles["calibration"],
        "outer_test_indices": roles["test"], "selection_policy": "fixed_epoch; validation diagnostic only",
        "eligible_for_strict_outer_fold_topology_oof": True
    }
    validate_checkpoint_provenance(provenance, checkpoint, sha256(processed), sha256(outer), roles["test"], int(frozen["inputs"]["sample_count"]))
    target = base_dir / "provenance.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(target)


def finalize(args: argparse.Namespace) -> None:
    frozen = load_json(args.protocol)
    if frozen.get("status") != "frozen_before_base_training":
        raise ValueError("Protocol is not frozen for base training")
    processed, outer, nested = map(Path, (args.processed, args.outer_manifest, args.nested_manifest))
    verify_inputs(frozen, processed, outer, nested)
    base_root, output = Path(args.base_root), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    checkpoints = []
    for fold in frozen["inputs"]["outer_folds"]:
        directory = base_root / f"fold{fold}"
        checkpoint, provenance_path = directory / "best_model.pt", directory / "provenance.json"
        provenance = load_json(provenance_path)
        roles = load_nested_indices(nested, fold, int(frozen["inputs"]["sample_count"]))
        validate_checkpoint_provenance(provenance, checkpoint, sha256(processed), sha256(outer), roles["test"], int(frozen["inputs"]["sample_count"]))
        checkpoints.append({"fold": fold, "path": str(checkpoint.resolve()), "provenance": str(provenance_path.resolve())})
    topology = frozen["topology_training"]
    final = {
        "artifact_type": "ghxtox_soft_cys_topology_oof_execution_protocol", "version": 1,
        "status": "frozen_before_candidate_oof", "scope": "training_oof_only", "checkpoint_selection": "fixed_epochs",
        "processed_train": str(processed.resolve()), "training_data_sha256": sha256(processed),
        "fold_manifest": str(outer.resolve()), "fold_manifest_sha256": sha256(outer), "checkpoints": checkpoints,
        "seed": topology["seed"], "epochs": topology["epochs"], "batch_size": topology["batch_size"],
        "learning_rate": topology["learning_rate"], "weight_decay": topology["weight_decay"], "threshold": frozen["evaluation"]["threshold"],
        "variants": topology["variants"], "model_overrides": {"topology_max_distance": topology["max_distance_angstrom"], "topology_max_candidates": topology["max_candidates"], "topology_temperature": topology["temperature"]},
        "stability_repeats": 3, "stability_sigma": 0.5, "bootstrap_resamples": frozen["evaluation"]["bootstrap_resamples"],
        "promotion_rule": frozen["promotion_rule"], "parent_frozen_protocol": str(Path(args.protocol).resolve()), "parent_protocol_sha256": sha256(args.protocol),
        "external_test_labels_used": False
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--protocol", required=True); common.add_argument("--processed", required=True)
    common.add_argument("--outer-manifest", required=True); common.add_argument("--nested-manifest", required=True)
    base = sub.add_parser("base-provenance", parents=[common])
    base.add_argument("--fold", type=int, required=True); base.add_argument("--base-dir", required=True)
    base.add_argument("--frozen-base-config", required=True)
    final = sub.add_parser("finalize", parents=[common])
    final.add_argument("--base-root", required=True); final.add_argument("--output", required=True)
    args = parser.parse_args()
    (write_base_provenance if args.command == "base-provenance" else finalize)(args)


if __name__ == "__main__":
    main()
