#!/usr/bin/env python
"""Build derived multi-conformer inputs for the frozen GHXTox interpretation run.

The source processed tensors and audited conformer caches are never modified.
The derived files live under ``paper_figures/data/current_interpretability`` and
therefore remain separate from canonical training and benchmark artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghxtox.conformer_features import attach_conformer_features


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "paper_figures" / "data" / "current_interpretability" / "inputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing derived inputs. Canonical source artifacts are never overwritten.",
    )
    return parser.parse_args()


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = [
        _require(root / "data" / "test_conformers" / "pepflow" / "test1" / "cache" / "conformer_manifest.csv"),
        _require(root / "data" / "test_conformers" / "af2" / "test1" / "cache_16_30" / "conformer_manifest.csv"),
    ]
    cache_roots = [path.parent for path in manifests]
    jobs = [
        {
            "split": "Test1",
            "processed": _require(root / "data" / "processed" / "test1_chemical_sites_final_esm2.pt"),
            "output": output_dir / "test1_current_multiconformer.pt",
            "match_by_sequence": False,
            "allow_unused_manifest_records": False,
        },
        {
            "split": "Test2",
            "processed": _require(root / "data" / "processed" / "test2_chemical_sites_final_esm2.pt"),
            "output": output_dir / "test2_current_multiconformer.pt",
            "match_by_sequence": True,
            "allow_unused_manifest_records": True,
        },
    ]

    summaries: list[dict[str, object]] = []
    for job in jobs:
        output = Path(job["output"])
        if output.exists() and not args.overwrite:
            summaries.append(
                {
                    "split": job["split"],
                    "output": str(output),
                    "status": "reused_existing_derived_input",
                }
            )
            continue
        summary = attach_conformer_features(
            processed=Path(job["processed"]),
            conformer_manifest=manifests,
            cache_root=cache_roots,
            output=output,
            include_conformer_graphs=False,
            match_by_sequence=bool(job["match_by_sequence"]),
            allow_unused_manifest_records=bool(job["allow_unused_manifest_records"]),
        )
        summaries.append({"split": job["split"], "status": "created", **summary})

    report = {
        "analysis": "current frozen GHXTox interpretability",
        "derived_input_policy": (
            "Audited conformer summaries are joined without changing source processed tensors; "
            "uncovered records receive exact zero descriptors with a hard availability mask."
        ),
        "conformer_manifests": [str(path) for path in manifests],
        "conformer_cache_roots": [str(path) for path in cache_roots],
        "jobs": summaries,
    }
    report_path = output_dir / "preparation_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
