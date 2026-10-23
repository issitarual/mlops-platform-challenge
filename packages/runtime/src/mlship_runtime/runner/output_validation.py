"""Validate what ``train()`` returned, collecting every problem at once."""

import math
from collections.abc import Mapping, Sized
from numbers import Real
from typing import Any

from mlship_runtime.runner.issues import Issue
from mlship_sdk import TrainOutput


def validate_train_output(output: object) -> tuple[list[Issue], Any]:
    """Return (issues, predictions on the input example). No issues = valid.

    Checks: is a ``TrainOutput``; ``model`` has ``.predict``; metrics are finite
    numbers; ``model.predict(input_example)`` runs and returns one value per row.
    """
    if not isinstance(output, TrainOutput):
        return [
            Issue(
                "TRAIN_OUTPUT_TYPE",
                f"train() returned {type(output).__name__}, not TrainOutput.",
                "Return TrainOutput(model=..., metrics=..., input_example=...) from mlship_sdk.",
            )
        ], None

    issues: list[Issue] = []
    predict = getattr(output.model, "predict", None)
    if not callable(predict):
        issues.append(
            Issue(
                "MODEL_NO_PREDICT",
                f"The model ({type(output.model).__name__}) has no predict() method.",
                "Return a fitted model; keep preprocessing inside it (e.g. a Pipeline).",
            )
        )
    issues.extend(_metric_issues(output.metrics))
    if output.input_example is None:
        issues.append(
            Issue(
                "INPUT_EXAMPLE_MISSING",
                "input_example is None.",
                "Return a few rows of model input, e.g. x_test.head(5).",
            )
        )

    predictions = None
    if callable(predict) and output.input_example is not None:
        try:
            predictions = predict(output.input_example)
        except Exception as exc:
            issues.append(
                Issue(
                    "SMOKE_PREDICT_FAILED",
                    f"model.predict(input_example) raised {type(exc).__name__}: {exc}",
                    "input_example must have the same shape and columns the model expects.",
                )
            )
        else:
            expected, got = _length(output.input_example), _length(predictions)
            if expected is not None and got is not None and expected != got:
                issues.append(
                    Issue(
                        "SMOKE_PREDICT_SHAPE",
                        f"predict() returned {got} values for {expected} input rows.",
                        "predict() must return one prediction per input row.",
                    )
                )
    return issues, predictions


def _metric_issues(metrics: object) -> list[Issue]:
    if not isinstance(metrics, Mapping):
        return [
            Issue(
                "METRICS_TYPE",
                f"metrics is {type(metrics).__name__}, not a dict.",
                'Return metrics as a dict, e.g. {"rmse": 0.51}.',
            )
        ]
    issues: list[Issue] = []
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            issues.append(
                Issue("METRIC_NAME", f"Metric name {name!r} is not a non-empty string.", "")
            )
        if isinstance(value, bool) or not isinstance(value, Real):
            issues.append(
                Issue(
                    "METRIC_NOT_NUMBER",
                    f"Metric {name!r} is {type(value).__name__}, not a number.",
                    "Convert it with float(...).",
                )
            )
        elif not math.isfinite(float(value)):
            issues.append(
                Issue(
                    "METRIC_NOT_FINITE",
                    f"Metric {name!r} is {value}.",
                    "NaN or infinite metrics usually mean an empty or degenerate test set.",
                )
            )
    return issues


def _length(value: object) -> int | None:
    return len(value) if isinstance(value, Sized) else None
