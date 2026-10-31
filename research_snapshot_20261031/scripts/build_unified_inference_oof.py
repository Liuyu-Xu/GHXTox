"""Create the stable training-OOF inference contract required by the utility router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghxtox.unified_inference import build_unified_oof_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--processed-train", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-oof", action="append", default=[])
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite contract: {output}")
    summary = build_unified_oof_contract(args.predictions, args.processed_train, args.fold_manifest, output, args.seed_oof)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
