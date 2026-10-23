from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from mlship_runtime.runner.config import SerializationPolicy
from mlship_runtime.runner.issues import Issue
from mlship_runtime.runner.writer import SavedModel, save_model

X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0) % 3})
Y = X["a"] * 2 + X["b"]
TREE_PREDICTOR = "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"


def _save(
    model: Any, tmp_path: Path, policy: SerializationPolicy
) -> tuple[SavedModel | None, list[Issue]]:
    predictions = model.predict(X.head(3))
    return save_model(model, X.head(3), predictions, tmp_path / "model", policy)


def test_sklearn_model_is_saved_with_skops(tmp_path: Path) -> None:
    saved, issues = _save(Ridge().fit(X, Y), tmp_path, SerializationPolicy(False, ()))
    assert issues == []
    assert saved is not None and saved.serialization == "skops"
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))
    assert np.allclose(loaded.predict(X.head(3)), Ridge().fit(X, Y).predict(X.head(3)))


def test_allowlisted_tree_types_are_recorded(tmp_path: Path) -> None:
    model = HistGradientBoostingRegressor(max_iter=5, random_state=0).fit(X, Y)
    saved, issues = _save(model, tmp_path, SerializationPolicy(False, (TREE_PREDICTOR,)))
    assert issues == []
    assert saved is not None and saved.trusted_types == (TREE_PREDICTOR,)


def test_types_outside_the_allowlist_are_refused(tmp_path: Path) -> None:
    model = HistGradientBoostingRegressor(max_iter=5, random_state=0).fit(X, Y)
    saved, issues = _save(model, tmp_path, SerializationPolicy(False, ()))
    assert saved is None
    assert [i.code for i in issues] == ["MODEL_UNTRUSTED_TYPES"]
    assert TREE_PREDICTOR in issues[0].message
    assert not (tmp_path / "model").exists()


class CustomModel:
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(x["a"]) * 2


def test_non_sklearn_model_needs_explicit_pickle_opt_in(tmp_path: Path) -> None:
    saved, issues = _save(CustomModel(), tmp_path, SerializationPolicy(False, ()))
    assert saved is None
    assert [i.code for i in issues] == ["MODEL_REQUIRES_PICKLE"]


def test_non_sklearn_model_is_saved_with_cloudpickle_when_allowed(tmp_path: Path) -> None:
    saved, issues = _save(CustomModel(), tmp_path, SerializationPolicy(True, ()))
    assert issues == []
    assert saved is not None and saved.serialization == "cloudpickle"
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))
    assert list(loaded.predict(X.head(2))) == [0.0, 2.0]
