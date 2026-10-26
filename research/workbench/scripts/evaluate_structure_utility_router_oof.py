"""Fit and evaluate the frozen cross-fitted pre-ESMFold utility router."""
from __future__ import annotations

import argparse
import json

from ghxtox.structure_utility_router import run_cross_fitted_router


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_cross_fitted_router(args.protocol, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
