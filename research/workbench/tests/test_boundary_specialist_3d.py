from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from ghxtox.boundary_specialist_3d import _load_resumed_fold_result


def _payload() -> dict:
    return {
        "schema_version": 1,
        "phase": 33,
        "fold": 2,
        "seed": 20260805,
        "protocol_sha256": "frozen-protocol-hash",
        "fine_tuning": {"epochs": 8, "maximum_absolute_logit_delta": 0.15},
        "held_source_indices": [11, 17],
        "predictions": [
            {
                "source_index": 11,
                "standalone_logit": 0.0,
                "standalone_probability": 0.5,
            },
            {
                "source_index": 17,
                "standalone_logit": 1.0,
                "standalone_probability": 0.7310585786300049,
            },
        ],
        "history": {
            "fold": 2,
            "train_rows": 480,
            "held_rows": 2,
            "epochs": [],
        },
    }


def _load(path):
    return _load_resumed_fold_result(
        path,
        fold=2,
        fold_seed=20260805,
        expected_held_indices={11, 17},
        expected_train_rows=480,
        protocol_sha256="frozen-protocol-hash",
        fine_tuning={"epochs": 8, "maximum_absolute_logit_delta": 0.15},
    )


def test_resume_accepts_identical_frozen_fold_result(tmp_path) -> None:
    path = tmp_path / "result.pt"
    torch.save(_payload(), path)
    result = _load(path)
    assert result["held_source_indices"] == [11, 17]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_sha256", "stale", "protocol_sha256"),
        ("fold", 1, "fold"),
        ("seed", 99, "seed"),
    ],
)
def test_resume_rejects_stale_metadata(tmp_path, field, value, message) -> None:
    payload = _payload()
    payload[field] = value
    path = tmp_path / "result.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        _load(path)


def test_resume_rejects_wrong_held_rows_or_probability(tmp_path) -> None:
    wrong_rows = deepcopy(_payload())
    wrong_rows["held_source_indices"] = [11, 18]
    path = tmp_path / "wrong_rows.pt"
    torch.save(wrong_rows, path)
    with pytest.raises(ValueError, match="held cohort"):
        _load(path)

    wrong_probability = deepcopy(_payload())
    wrong_probability["predictions"][1]["standalone_probability"] = 0.2
    path = tmp_path / "wrong_probability.pt"
    torch.save(wrong_probability, path)
    with pytest.raises(ValueError, match="probability/logit mismatch"):
        _load(path)


def test_resume_rejects_legacy_unversioned_result(tmp_path) -> None:
    legacy = {
        "predictions": _payload()["predictions"],
        "history": _payload()["history"],
    }
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)
    with pytest.raises(ValueError, match="schema_version"):
        _load(path)
