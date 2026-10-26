"""Command-line entry points for the frozen Quick-to-Full routing study."""
from __future__ import annotations

import argparse
from pathlib import Path

from ghxtox.quick_full_routing import build_quick_full_baseline, run_quick_full_router

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-OOF-only Quick-to-Full routing analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--contract", type=Path, default=ROOT / "runs/unified_inference_oof_v1/contract.csv")
    baseline.add_argument("--quick-policy", type=Path, default=ROOT / "configs/web_quick_sequence_policy_v1.json")
    baseline.add_argument("--seed-oof", type=Path, nargs=3, default=[ROOT / f"runs/lora_mamba_full/oof_seed{seed}.json" for seed in (42, 123, 2025)])
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--full-threshold", type=float, default=0.47022256531284556)
    router = subparsers.add_parser("router")
    router.add_argument("--protocol", type=Path, required=True)
    router.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "baseline":
        build_quick_full_baseline(args.contract, args.quick_policy, args.seed_oof, args.output, full_threshold=args.full_threshold)
    else:
        run_quick_full_router(args.protocol, args.output)


if __name__ == "__main__":
    main()
