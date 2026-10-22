"""Model loading and inference (the adapter around MLflow pyfunc)."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ModelEngine:
    """Implements the ``Predictor`` port. Loads only after the hash check passes."""

    def __init__(self, model_dir: Path, expected_sha256: str, model_version: str) -> None:
        self._model_dir = model_dir
        self._expected_sha256 = expected_sha256
        self.model_version = model_version

    def load(self) -> None:
        """Verify ``expected_sha256`` against the files, then load the pyfunc model."""
        raise NotImplementedError("E4")

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> list[Any]:
        """Vectorized prediction for already validated rows."""
        raise NotImplementedError("E4")
