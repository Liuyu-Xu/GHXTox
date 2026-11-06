from ghxtox.hypeptox_paper_analysis import HEADLINE_METRICS, _headline_rows


def test_balanced_accuracy_is_not_a_headline_metric() -> None:
    assert "balanced_accuracy" not in HEADLINE_METRICS


def test_headline_table_marks_unreported_hypeptox_metrics() -> None:
    final_row = {
        "accuracy": 0.91,
        "sn": 0.90,
        "sp": 0.92,
        "f1": 0.91,
        "mcc": 0.82,
        "auroc": 0.96,
        "auprc": 0.97,
        "brier": 0.07,
        "ece_10": 0.03,
    }
    rows = _headline_rows(final_row)
    by_key = {row["metric_key"]: row for row in rows}

    assert by_key["auprc"]["direction"] == "not_reported_by_hypeptox_fuse"
    assert by_key["brier"]["direction"] == "not_reported_by_hypeptox_fuse"
    assert by_key["ece_10"]["direction"] == "not_reported_by_hypeptox_fuse"
