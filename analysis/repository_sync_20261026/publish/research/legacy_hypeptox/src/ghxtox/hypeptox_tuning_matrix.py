"""Generate reproducible HyPepTox-Fuse parameter tuning configs and commands."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


TRAIN_PATH = "data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt"
VAL_PATH = "data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt"
TEST_PATH = "data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt"

COMMON_ALGORITHM_UPDATES = {
    "train": {
        "scheduler": "cosine",
        "min_learning_rate": 0.00003,
        "ema_decay": 0.995,
        "ema_start_epoch": 2,
        "selection_score": {"mcc": 1.0, "auroc": 0.2, "auprc": 0.2},
        "selection_constraints": {"sn": {"min": 0.875}, "sp": {"min": 0.91}},
        "selection_penalty_weight": 8.0,
    }
}


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "bilstm_focal_mcc",
        "base": "configs/sequence_1d_bilstm.json",
        "seeds": [42, 123, 2025],
        "updates": {
            "model": {"dropout": 0.24, "hidden_dim": 160},
            "train": {
                "epochs": 45,
                "batch_size": 40,
                "learning_rate": 0.0004,
                "weight_decay": 0.00015,
                "early_stop_patience": 10,
                "loss": "focal_bce",
                "focal_gamma": 1.5,
                "monitor": "mcc",
                "save_auxiliary_checkpoints": True,
            },
        },
        "reason": "Increase sequence expert capacity and use focal BCE to improve hard-sample MCC/SN.",
    },
    {
        "name": "bilstm_smooth_auprc",
        "base": "configs/sequence_1d_bilstm.json",
        "seeds": [42, 123, 2025],
        "updates": {
            "model": {"dropout": 0.18, "hidden_dim": 128},
            "train": {
                "epochs": 45,
                "batch_size": 48,
                "learning_rate": 0.00035,
                "weight_decay": 0.0002,
                "early_stop_patience": 10,
                "label_smoothing": 0.03,
                "monitor": "auprc",
                "save_auxiliary_checkpoints": True,
            },
        },
        "reason": "Use mild label smoothing and AUPRC checkpointing to improve ranking stability.",
    },
    {
        "name": "geometry_focal_mcc",
        "base": "configs/plm_fusion_esm2_geometry_confidence.json",
        "seeds": [42, 123, 2025],
        "updates": {
            "model": {"dropout": 0.24, "hidden_dim": 160, "spatial_top_k": 12},
            "train": {
                "epochs": 45,
                "batch_size": 32,
                "learning_rate": 0.00035,
                "weight_decay": 0.00015,
                "early_stop_patience": 10,
                "coord_noise_std": 0.10,
                "coord_noise_plddt_scale": 1.5,
                "loss": "focal_bce",
                "focal_gamma": 1.3,
                "monitor": "mcc",
                "save_auxiliary_checkpoints": True,
            },
        },
        "reason": "Tune geometry regularization and graph neighborhood for MCC while retaining 3D signal.",
    },
    {
        "name": "geometry_low_noise_auprc",
        "base": "configs/plm_fusion_esm2_geometry_confidence.json",
        "seeds": [42, 123, 2025],
        "updates": {
            "model": {"dropout": 0.18, "spatial_top_k": 8, "plddt_gate_temperature": 0.16},
            "train": {
                "epochs": 45,
                "batch_size": 40,
                "learning_rate": 0.0003,
                "weight_decay": 0.0002,
                "early_stop_patience": 10,
                "coord_noise_std": 0.06,
                "coord_noise_plddt_scale": 1.0,
                "label_smoothing": 0.02,
                "monitor": "auprc",
                "save_auxiliary_checkpoints": True,
            },
        },
        "reason": "Reduce coordinate noise and select by AUPRC to target the remaining AUROC/AUPRC gap.",
    },
    {
        "name": "chemical_branch_focal_mcc",
        "base": "configs/hypeptox_fuse_3d_v2_augmix_noleak.json",
        "seeds": [42, 123, 2025],
        "updates": {
            "model": {"chemical_site_dropout": 0.16, "chemical_site_hidden_dim": 96},
            "train": {
                "epochs": 45,
                "batch_size": 32,
                "learning_rate": 0.00045,
                "weight_decay": 0.00015,
                "early_stop_patience": 10,
                "loss": "focal_bce",
                "focal_gamma": 1.2,
                "monitor": "mcc",
                "save_auxiliary_checkpoints": True,
            },
        },
        "reason": "Tune only the chemical-site expert capacity around the current 3D-v2 branch.",
    },
    {
        "name": "chemical_full_finetune_auprc",
        "base": "configs/hypeptox_fuse_3d_v2_augmix_noleak.json",
        "seeds": [42],
        "updates": {
            "model": {"dropout": 0.18, "chemical_site_dropout": 0.12},
            "train": {
                "epochs": 35,
                "batch_size": 32,
                "learning_rate": 0.00012,
                "weight_decay": 0.00025,
                "early_stop_patience": 8,
                "label_smoothing": 0.02,
                "monitor": "auprc",
                "save_auxiliary_checkpoints": True,
                "trainable_prefixes": [],
            },
        },
        "reason": "Low-LR full fine-tune from the 3D-v2 initialization to test whether frozen-base training is limiting ranking metrics.",
    },
]


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _powershell_command(lines: list[str]) -> str:
    return " `\n  ".join(lines)


def _bash_command(lines: list[str]) -> str:
    return " \\\n  ".join(lines)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    config_dir = root / args.config_dir
    run_root = Path(args.run_root)
    powershell_command_file = root / args.powershell_command_file
    bash_command_file = root / args.bash_command_file
    manifest_rows: list[dict[str, Any]] = []
    ps_commands: list[str] = [
        "# Auto-generated HyPepTox-Fuse tuning commands.",
        "# Run from the project root after activating the Python/PyTorch environment.",
        "$env:PYTHONPATH='src'",
        "",
    ]
    sh_commands: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Auto-generated HyPepTox-Fuse tuning commands.",
        "# Run from the project root after activating the Python/PyTorch environment.",
        "export PYTHONPATH=src",
        "",
    ]

    for variant in VARIANTS:
        base_config = _load_json(root / variant["base"])
        for seed in variant["seeds"]:
            config = _deep_update(base_config, COMMON_ALGORITHM_UPDATES)
            config = _deep_update(config, variant["updates"])
            config["seed"] = int(seed)
            config.setdefault("experiment", {})
            config["experiment"].update(
                {
                    "name": f"hypeptox_tuning_{variant['name']}_seed{seed}",
                    "tuning_variant": variant["name"],
                    "reason": variant["reason"],
                    "selection_data": "HyPepTox-Fuse train-fit/train-val split; independent test untouched.",
                    "external_sets_used_for_selection": False,
                }
            )
            config_path = config_dir / f"{variant['name']}_seed{seed}.json"
            config_path_rel = config_path.relative_to(root).as_posix()
            run_dir = run_root / f"{variant['name']}_seed{seed}"
            _save_json(config_path, config)

            train_lines = [
                "python -m ghxtox.train",
                f"--train {TRAIN_PATH}",
                f"--val {VAL_PATH}",
                f"--config {config_path_rel}",
                f"--output-dir {run_dir.as_posix()}",
                f"--device {args.device}",
            ]
            val_eval_lines = [
                "python -m ghxtox.evaluate",
                f"--checkpoint {(run_dir / 'best_model.pt').as_posix()}",
                f"--processed {VAL_PATH}",
                f"--output {(run_dir / 'val_metrics.json').as_posix()}",
                f"--predictions {(run_dir / 'val_predictions.csv').as_posix()}",
                f"--device {args.device}",
                "--threshold 0.5",
            ]
            test_eval_lines = [
                "python -m ghxtox.evaluate",
                f"--checkpoint {(run_dir / 'best_model.pt').as_posix()}",
                f"--processed {TEST_PATH}",
                f"--output {(run_dir / 'test_metrics.json').as_posix()}",
                f"--predictions {(run_dir / 'test_predictions.csv').as_posix()}",
                f"--device {args.device}",
                "--threshold 0.5",
            ]
            threshold_lines = [
                "python -m ghxtox.threshold_predictions",
                f"--validation {(run_dir / 'val_predictions.csv').as_posix()}",
                f"--apply {(run_dir / 'test_predictions.csv').as_posix()}",
                f"--output-dir {(run_dir / 'thresholded').as_posix()}",
                f"--output-json {(run_dir / 'thresholded' / 'threshold_summary.json').as_posix()}",
                "--objective mcc",
                "--min-sn 0.875",
                "--min-sp 0.91",
            ]
            ps_commands.extend(
                [
                    f"# {variant['name']} seed {seed}: {variant['reason']}",
                    _powershell_command(train_lines),
                    _powershell_command(val_eval_lines),
                    _powershell_command(test_eval_lines),
                    _powershell_command(threshold_lines),
                    "",
                ]
            )
            sh_commands.extend(
                [
                    f"# {variant['name']} seed {seed}: {variant['reason']}",
                    _bash_command(train_lines),
                    _bash_command(val_eval_lines),
                    _bash_command(test_eval_lines),
                    _bash_command(threshold_lines),
                    "",
                ]
            )
            manifest_rows.append(
                {
                    "variant": variant["name"],
                    "seed": int(seed),
                    "config": config_path_rel,
                    "run_dir": run_dir.as_posix(),
                    "base": variant["base"],
                    "reason": variant["reason"],
                }
            )

    manifest = {
        "train_path": TRAIN_PATH,
        "val_path": VAL_PATH,
        "test_path": TEST_PATH,
        "device": args.device,
        "variants": manifest_rows,
        "next_step": (
            "After all runs finish, rank validation MCC/AUPRC, evaluate the best validation "
            "candidates once on test, then add useful checkpoints to ghxtox.ensemble_evaluate."
        ),
    }
    _save_json(root / args.manifest, manifest)
    powershell_command_file.parent.mkdir(parents=True, exist_ok=True)
    powershell_command_file.write_text("\n".join(ps_commands).rstrip() + "\n", encoding="utf-8")
    bash_command_file.parent.mkdir(parents=True, exist_ok=True)
    bash_command_file.write_text("\n".join(sh_commands).rstrip() + "\n", encoding="utf-8")
    print(f"Generated {len(manifest_rows)} tuning configs in {config_dir}")
    print(f"PowerShell command file: {powershell_command_file}")
    print(f"Bash command file: {bash_command_file}")
    print(f"Manifest: {root / args.manifest}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate HyPepTox-Fuse tuning configs and PowerShell commands.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config-dir", default="configs/hypeptox_tuning")
    parser.add_argument("--run-root", default="runs/hypeptox_tuning")
    parser.add_argument("--powershell-command-file", default="docs/hypeptox_fuse_tuning_commands.ps1")
    parser.add_argument("--bash-command-file", default="docs/hypeptox_fuse_tuning_commands.sh")
    parser.add_argument("--manifest", default="configs/hypeptox_tuning/manifest.json")
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    generate(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
