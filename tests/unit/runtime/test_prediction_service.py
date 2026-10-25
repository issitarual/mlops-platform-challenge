import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from mlship_runtime.serving.services.prediction_service import PredictionService


class FixedPredictor:
    model_name = "houses"
    model_version = "3"

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> list[Any]:
        return [1.0 for _ in rows]


def test_sampled_requests_are_logged_without_excluded_columns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = PredictionService(
        FixedPredictor(), sample_rate=0.5, exclude_columns=["email"], draw=lambda: 0.1
    )
    with caplog.at_level(logging.INFO, logger="mlship.predictions"):
        service.predict([{"rooms": 3.0, "email": "a@b.c"}], "req-1")
    (record,) = caplog.records
    assert record.rows == [{"rooms": 3.0}]  # type: ignore[attr-defined]
    assert record.model_version == "3" and record.request_id == "req-1"  # type: ignore[attr-defined]


def test_unsampled_requests_are_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    service = PredictionService(FixedPredictor(), sample_rate=0.5, draw=lambda: 0.9)
    with caplog.at_level(logging.INFO, logger="mlship.predictions"):
        service.predict([{"rooms": 3.0}], "req-1")
    assert caplog.records == []
