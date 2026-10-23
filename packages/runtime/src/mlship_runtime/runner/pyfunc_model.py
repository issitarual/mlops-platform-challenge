"""Level-2 wrapper (ADR-017): any object with ``.predict()``, saved with cloudpickle.

Lives in the runtime package, which is installed in the serving image too, so the
pickle references a class that exists when the model is loaded.
"""

from typing import Any

from mlflow.pyfunc.model import PythonModel


class PredictOnlyModel(PythonModel):
    def __init__(self, model: Any) -> None:
        self.model = model

    # No type hints on purpose: MLflow would read them as a schema and transform inputs.
    def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
        return self.model.predict(model_input)
