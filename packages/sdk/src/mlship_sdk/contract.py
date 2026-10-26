"""The only type a data scientist needs from mlship."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainOutput:
    """Value that ``train(data_dir, params)`` must return.

    Validation of the fields happens in the training runner, not here, so the
    contract stays a plain data holder with no logic to version.

    Attributes:
        model: Fitted object exposing ``.predict(X)``. Preprocessing must live inside
            it (e.g. a scikit-learn ``Pipeline``) so training and serving apply the
            same transformations.
        metrics: Evaluation metrics, name -> finite number (e.g. ``{"rmse": 0.51}``).
        input_example: A few rows of model input. Used to infer the API schema and
            to smoke-test the model right after training.
        reference_data: Optional. The training features (e.g. ``x_train``), as a
            pandas DataFrame with the same columns as ``input_example``. Only summary
            statistics are kept: they are the baseline serving compares requests with
            to detect drift. Without it, drift monitoring is off for the model.
    """

    model: Any
    metrics: Mapping[str, float]
    input_example: Any
    reference_data: Any = None
