from __future__ import annotations

import numpy as np

from ghxtox.mentor_report import _bootstrap_intervals, _paper_delta_rows
from ghxtox.publication_analysis import PAPER_BENCHMARKS, PredictionData


def test_bootstrap_intervals_cover_point_estimates() -> None:
    data = PredictionData(
        labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        probabilities=np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
        sequences=["A", "B", "C", "D"],
        threshold=0.5,
    )
    intervals = _bootstrap_intervals(data, iterations=50, seed=11)
    for values in intervals.values():
        assert values["lower"] <= values["estimate"] <= values["upper"]


def test_paper_deltas_never_impute_unreported_metrics() -> None:
    results = {
        dataset: {
            **papers,
            "ProtT5-3Seed": {
                metric: 0.95
                for metric in next(iter(papers.values()))
            },
        }
        for dataset, papers in PAPER_BENCHMARKS.items()
    }
    rows = _paper_delta_rows(results)
    toxmsrc_auprc = next(
        row
        for row in rows
        if row["dataset"] == "Test1"
        and row["reference"] == "ToxMSRC"
        and row["metric"] == "auprc"
    )
    assert toxmsrc_auprc["reference_value"] == "NR"
    assert toxmsrc_auprc["delta"] == "NR"
