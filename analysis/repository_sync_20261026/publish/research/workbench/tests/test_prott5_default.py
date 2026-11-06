from __future__ import annotations

import json

import torch

from ghxtox._legacy.plm_textcnn import PooledPLMTextCNN
from ghxtox._legacy.prott5_default import predict_default


def test_prott5_default_averages_fifteen_members(tmp_path) -> None:
    features_path = tmp_path / "features.pt"
    torch.save(
        {
            "features": torch.randn(4, 8),
            "labels": torch.tensor([0, 1, 0, 1]),
            "sample_ids": ["a", "b", "c", "d"],
            "sequences": ["AA", "CC", "DD", "EE"],
        },
        features_path,
    )
    checkpoint_dirs = []
    for seed in range(3):
        directory = tmp_path / f"seed{seed}"
        directory.mkdir()
        checkpoint_dirs.append(str(directory))
        for fold in range(5):
            model = PooledPLMTextCNN(
                input_dim=8,
                projection_dim=8,
                window_sizes=(2, 3),
                num_filters=4,
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": {
                        "input_dim": 8,
                        "projection_dim": 8,
                        "window_sizes": [2, 3],
                        "num_filters": 4,
                        "dropout": 0.2,
                    },
                },
                directory / f"fold{fold}_best_model.pt",
            )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "public_model_name": "test",
                "default_threshold": 0.45,
                "ensemble": "unweighted arithmetic mean",
                "checkpoint_dirs": checkpoint_dirs,
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "predictions.csv"
    summary = predict_default(
        config_path,
        features_path,
        output_path,
        metrics_path=None,
        device_name="cpu",
        batch_size=4,
    )
    assert output_path.exists()
    assert len(summary["protocol"]["checkpoints"]) == 15
    assert summary["metrics"]["tp"] + summary["metrics"]["fn"] == 2
