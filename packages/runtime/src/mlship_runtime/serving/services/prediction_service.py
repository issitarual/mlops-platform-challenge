"""Prediction use cases: batch limits and model version in every response."""

from collections.abc import Mapping, Sequence
from typing import Any

from mlship_runtime.serving.services.ports import Predictor


class PredictionService:
    def __init__(self, predictor: Predictor, max_batch_rows: int) -> None:
        self._predictor = predictor
        self._max_batch_rows = max_batch_rows

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> list[Any]:
        """Enforce the batch limit and delegate to the predictor (E4)."""
        raise NotImplementedError("E4")
