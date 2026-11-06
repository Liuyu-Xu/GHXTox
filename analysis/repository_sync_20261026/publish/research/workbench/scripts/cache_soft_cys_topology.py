"""Attach deterministic candidate feature caches to a new processed tensor file."""
import argparse
from pathlib import Path
import torch
from ghxtox.soft_cys_topology import cached_candidate_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-distance", type=float, default=8.0)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("Output must be a new path; retain original structure cache")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    for record in payload["records"]:
        record["soft_cys_cache"] = cached_candidate_features(record, args.max_distance)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Cached {len(payload['records'])} peptides")


if __name__ == "__main__":
    main()
