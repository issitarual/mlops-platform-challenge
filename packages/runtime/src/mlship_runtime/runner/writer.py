"""Persist the model in the format its trust level allows (ADR-017)."""

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlship_runtime.runner.config import SerializationPolicy
from mlship_runtime.runner.issues import Issue


@dataclass(frozen=True, slots=True)
class SavedModel:
    serialization: str
    trusted_types: tuple[str, ...]


def _is_sklearn_estimator(model: object) -> bool:
    try:
        from sklearn.base import BaseEstimator
    except ImportError:
        return False
    return isinstance(model, BaseEstimator)


def save_model(
    model: Any,
    input_example: Any,
    predictions: Any,
    model_dir: Path,
    policy: SerializationPolicy,
) -> tuple[SavedModel | None, list[Issue]]:
    from mlflow.models import infer_signature

    signature = infer_signature(input_example, predictions)
    # The environment is the image itself (ADR-005), so MLflow's environment files are
    # informational only. An explicit conda_env also stops MLflow from looking for pip,
    # which the image does not have (it would log a warning on every run).
    common: dict[str, Any] = {
        "signature": signature,
        "input_example": input_example,
        "conda_env": {
            "name": "mlship",
            "channels": [],
            "dependencies": [f"python={platform.python_version()}"],
        },
    }

    if _is_sklearn_estimator(model):
        import mlflow.sklearn
        from skops.io import dumps, get_untrusted_types

        used = sorted(get_untrusted_types(data=dumps(model)))
        not_allowed = [t for t in used if t not in policy.skops_trusted_types]
        if not_allowed:
            return None, [
                Issue(
                    "MODEL_UNTRUSTED_TYPES",
                    "The model uses types outside the platform allowlist: "
                    + ", ".join(not_allowed),
                    "If these types were reviewed, the platform can allow them with "
                    "MLSHIP_SKOPS_TRUSTED_TYPES. Otherwise, train with --allow-pickle "
                    "(loading the model can then run arbitrary code).",
                )
            ]
        mlflow.sklearn.save_model(
            model,
            str(model_dir),
            serialization_format="skops",
            skops_trusted_types=used,
            **common,
        )
        return SavedModel("skops", tuple(used)), []

    if not policy.allow_pickle:
        return None, [
            Issue(
                "MODEL_REQUIRES_PICKLE",
                f"{type(model).__name__} is not a scikit-learn estimator, so it can only "
                "be saved with cloudpickle.",
                "Train with --allow-pickle to accept that risk explicitly; the choice is "
                "recorded with the run.",
            )
        ]
    import mlflow.pyfunc

    from mlship_runtime.runner.pyfunc_model import PredictOnlyModel

    mlflow.pyfunc.save_model(str(model_dir), python_model=PredictOnlyModel(model), **common)
    return SavedModel("cloudpickle", ()), []


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
