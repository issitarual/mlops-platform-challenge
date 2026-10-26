"""A real, small model saved exactly as the runner saves it, plus serving settings."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlship_runtime import reference
from mlship_runtime.runner.config import SerializationPolicy
from mlship_runtime.runner.writer import save_model
from mlship_runtime.serving.core.config import ServingSettings
from mlship_runtime.serving.engines.model_files import model_files_digest

X = pd.DataFrame({"rooms": np.arange(1.0, 21.0), "age": np.arange(20) % 7})  # double, long


def saved_model(
    root: Path, *, allow_pickle: bool = False, model: Any = None, with_reference: bool = False
) -> Path:
    fitted = model if model is not None else Ridge().fit(X, X["rooms"] * 2 + X["age"])
    model_dir = root / "model"
    saved, issues = save_model(
        fitted,
        X.head(3),
        fitted.predict(X.head(3)),
        model_dir,
        SerializationPolicy(allow_pickle, ()),
    )
    assert saved is not None and not issues, issues
    if with_reference:
        training = pd.DataFrame({"rooms": np.arange(500.0) % 20, "age": np.arange(500) % 7})
        reference.save(reference.compute(training, ["rooms", "age"]), model_dir)
    return model_dir


def settings_for(model_dir: Path, **overrides: object) -> ServingSettings:
    values: dict[str, object] = {
        "model_dir": model_dir,
        "model_name": "houses",
        "model_version": "3",
        "model_sha256": model_files_digest(model_dir),
        "max_batch_rows": 5,
        "max_payload_bytes": 4_000,
    }
    values.update(overrides)
    return ServingSettings(**values)  # type: ignore[arg-type]


class NotSklearn:
    """A model that can only be saved with cloudpickle."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(x))
