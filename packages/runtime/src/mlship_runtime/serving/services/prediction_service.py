"""Prediction use case: delegate to the predictor and keep a sampled record for drift."""

import logging
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mlship_runtime.serving.services.ports import Predictor

logger = logging.getLogger("mlship.predictions")


class PredictionService:
    def __init__(
        self,
        predictor: Predictor,
        *,
        sample_rate: float = 0.0,
        exclude_columns: Sequence[str] = (),
        draw: Callable[[], float] = random.random,
    ) -> None:
        self._predictor = predictor
        self._sample_rate = sample_rate
        self._exclude = frozenset(exclude_columns)
        self._draw = draw

    @property
    def model_name(self) -> str:
        return self._predictor.model_name

    @property
    def model_version(self) -> str:
        return self._predictor.model_version

    def predict(self, rows: Sequence[Mapping[str, Any]], request_id: str) -> list[Any]:
        predictions = self._predictor.predict(rows)
        if self._sample_rate and self._draw() < self._sample_rate:
            logger.info(
                "prediction sample",
                extra={
                    "request_id": request_id,
                    "model_version": self.model_version,
                    "rows": [
                        {k: v for k, v in row.items() if k not in self._exclude} for row in rows
                    ],
                    "predictions": predictions,
                },
            )
        return predictions
