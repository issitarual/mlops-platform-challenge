import math
from typing import Any, cast

import pandas as pd
from sklearn.linear_model import Ridge

from mlship_runtime.runner.output_validation import validate_train_output
from mlship_sdk import TrainOutput

X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 1.0, 0.0]})
MODEL = Ridge().fit(X, [1.0, 2.0, 3.0])


def codes(output: object) -> set[str]:
    return {issue.code for issue in validate_train_output(output)[0]}


def test_valid_output_has_no_issues_and_returns_predictions() -> None:
    issues, predictions = validate_train_output(
        TrainOutput(model=MODEL, metrics={"rmse": 0.1, "n": 3}, input_example=X)
    )
    assert issues == []
    assert len(predictions) == 3


def test_wrong_return_type_is_reported() -> None:
    assert codes({"model": MODEL}) == {"TRAIN_OUTPUT_TYPE"}


def test_every_problem_is_reported_at_once() -> None:
    output = TrainOutput(
        model=object(),
        metrics={"rmse": math.nan, "flag": True, "name": "x", "inf": math.inf},  # type: ignore[dict-item]
        input_example=None,
    )
    assert codes(output) == {
        "MODEL_NO_PREDICT",
        "METRIC_NOT_FINITE",
        "METRIC_NOT_NUMBER",
        "INPUT_EXAMPLE_MISSING",
    }
    assert len(validate_train_output(output)[0]) == 6


def test_smoke_prediction_failure_is_reported() -> None:
    wrong_columns = pd.DataFrame({"z": [1.0]})
    output = TrainOutput(model=MODEL, metrics={}, input_example=wrong_columns)
    assert codes(output) == {"SMOKE_PREDICT_FAILED"}


def test_prediction_count_must_match_rows() -> None:
    class OnePrediction:
        def predict(self, x: object) -> list[float]:
            return [1.0]

    output = TrainOutput(model=OnePrediction(), metrics={}, input_example=X)
    assert codes(output) == {"SMOKE_PREDICT_SHAPE"}


def test_metrics_must_be_a_mapping() -> None:
    output = TrainOutput(model=MODEL, metrics=cast(Any, [0.1]), input_example=X)
    assert codes(output) == {"METRICS_TYPE"}


def test_metric_names_mlflow_would_refuse_are_reported() -> None:
    output = TrainOutput(model=MODEL, metrics={"rmse (test)": 0.5, "ok": 0.1}, input_example=X)
    assert codes(output) == {"METRIC_NAME_INVALID"}


def test_too_many_metrics_are_reported() -> None:
    metrics = {f"m{i}": 0.1 for i in range(101)}
    assert codes(TrainOutput(model=MODEL, metrics=metrics, input_example=X)) == {"METRICS_TOO_MANY"}
