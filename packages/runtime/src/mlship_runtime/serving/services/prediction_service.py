"""Prediction use case: delegate to the predictor, feed drift, keep a sampled record."""

import logging
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mlship_runtime.serving.services.drift import DriftMonitor
from mlship_runtime.serving.services.ports import Predictor

logger = logging.getLogger("mlship.predictions")


class PredictionService:
    def __init__(
        self,
        predictor: Predictor,
        *,
        sample_rate: float = 0.0,
        exclude_columns: Sequence[str] = (),
        drift: DriftMonitor | None = None,
        draw: Callable[[], float] = random.random,
    ) -> None:
        self._predictor = predictor
        self._drift = drift
        self._sample_rate = sample_rate
        self._exclude = frozenset(exclude_columns)
        self._draw = draw

    @property
    def model_name(self) -> str:
        return self._predictor.model_name

    @property
    def model_version(self) -> str:
        return self._predictor.model_version

    @property
    def drift_features(self) -> tuple[str, ...]:
        return self._drift.features if self._drift is not None else ()

    def drift_scores(self) -> dict[str, float]:
        return self._drift.scores() if self._drift is not None else {}

    def predict(self, rows: Sequence[Mapping[str, Any]], request_id: str) -> list[Any]:
        predictions = self._predictor.predict(rows)
        if self._drift is not None:
            self._drift.observe(rows)
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
