import torch

from ghxtox.prediction_ensemble_search import search_prediction_ensembles
from ghxtox.threshold_predictions import find_threshold
from ghxtox.train import ExponentialMovingAverage


def test_threshold_selection_respects_sensitivity_constraint() -> None:
    probabilities = torch.tensor([0.10, 0.35, 0.45, 0.80])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])

    result = find_threshold(probabilities, labels, objective="mcc", min_sn=1.0)

    assert result["metrics"]["sn"] == 1.0
    assert result["threshold"] <= 0.45


def test_ema_swaps_and_restores_parameters() -> None:
    model = torch.nn.Linear(2, 1)
    ema = ExponentialMovingAverage(model, decay=0.5)
    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2.0)
    updated = model.weight.detach().clone()
    ema.update(model)

    with ema.average_parameters(model):
        assert not torch.allclose(model.weight, updated)

    assert torch.allclose(model.weight, updated)
    assert not torch.allclose(model.weight, original)


def test_prediction_ensemble_search_finds_weighted_combo(tmp_path) -> None:
    manifest_rows = []
    labels = [0, 0, 1, 1]
    candidate_probs = {
        "weak_a": [0.2, 0.55, 0.7, 0.8],
        "weak_b": [0.4, 0.2, 0.45, 0.9],
        "strong_c": [0.1, 0.2, 0.8, 0.9],
    }
    for name, probs in candidate_probs.items():
        run_dir = tmp_path / "runs" / name
        run_dir.mkdir(parents=True)
        manifest_rows.append({"run_dir": run_dir.as_posix()})
        for split in ("val", "test"):
            path = run_dir / f"{split}_predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write("sample_id,sequence,label,toxicity_probability\n")
                for index, (label, prob) in enumerate(zip(labels, probs, strict=True)):
                    handle.write(f"id{index},AAAA,{label},{prob}\n")
    manifest = {"variants": manifest_rows}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "manifest": manifest_path,
            "run_dirs": [],
            "val_predictions_name": "val_predictions.csv",
            "test_predictions_name": "test_predictions.csv",
            "output_dir": tmp_path / "out",
            "objective": "mcc",
            "min_sn": None,
            "min_sp": None,
            "min_k": 1,
            "max_k": 3,
            "refine_top": 3,
            "weight_trials": 4,
            "weight_seed": 7,
            "weight_alpha": 1.0,
            "keep_uniform": 8,
            "output_top": 8,
        },
    )()

    summary = search_prediction_ensembles(args)

    assert summary["best"]["test_metrics"]["mcc"] == 1.0
    assert (tmp_path / "out" / "ensemble_search_summary.json").exists()
    assert (tmp_path / "out" / "best_test_predictions.csv").exists()
