"""Model loading and inference: the adapter around MLflow pyfunc (``Predictor`` port).

Order matters: verify the files, check the load policy, and only then deserialize.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mlship_runtime.serving.core.config import ServingSettings
from mlship_runtime.serving.engines import load_policy, model_files


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    type: str
    required: bool


class ModelEngine:
    def __init__(self, settings: ServingSettings) -> None:
        self._settings = settings
        self.model_name = settings.model_name
        self.model_version = settings.model_version
        self._model: Any = None
        self.columns: tuple[Column, ...] = ()

    def inspect(self) -> None:
        """Verify integrity and policy, read the input schema. Never runs model code."""
        settings = self._settings
        model_files.verify(settings.model_dir, settings.model_sha256)
        descriptor = load_policy.read_descriptor(settings.model_dir)
        load_policy.check(
            descriptor,
            allow_pickle=settings.allow_pickle,
            trusted_types=settings.skops_trusted_types,
        )
        self.columns = _columns(descriptor.signature)

    def load(self) -> None:
        import mlflow.pyfunc

        self._model = mlflow.pyfunc.load_model(str(self._settings.model_dir))

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> list[Any]:
        frame = pd.DataFrame(list(rows), columns=[c.name for c in self.columns] or None)
        return _to_list(self._model.predict(frame))


def _columns(signature: dict[str, Any] | None) -> tuple[Column, ...]:
    """Columns of a column-based signature; empty when there is none (generic rows)."""
    if not signature or not signature.get("inputs"):
        return ()
    from mlflow.models import ModelSignature

    parsed = ModelSignature.from_dict(signature)
    specs = getattr(parsed.inputs, "inputs", [])
    columns = []
    for spec in specs:
        name = getattr(spec, "name", None)
        dtype = getattr(spec, "type", None)
        if name is None or dtype is None or not hasattr(dtype, "name"):
            return ()
        columns.append(Column(str(name), str(dtype.name), bool(getattr(spec, "required", True))))
    return tuple(columns)


def _to_list(predictions: Any) -> list[Any]:
    if isinstance(predictions, pd.DataFrame):
        return list(predictions.to_dict(orient="records"))
    if isinstance(predictions, pd.Series | np.ndarray):
        return list(np.asarray(predictions).tolist())
    return list(predictions)
