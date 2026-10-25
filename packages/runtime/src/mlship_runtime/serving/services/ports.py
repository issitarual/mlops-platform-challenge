"""Interfaces the serving services depend on (implemented in ``engines/``)."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class Predictor(Protocol):
    """Anything that can predict on validated rows and report which model it is."""

    model_name: str
    model_version: str

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> list[Any]: ...
