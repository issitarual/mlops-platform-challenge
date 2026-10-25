"""An instance only serves the registered model, loaded with the trust the platform grants."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from mlflow.models import Model
from serving_fixtures import NotSklearn, saved_model, settings_for

from mlship_runtime.serving.engines.load_policy import LoadPolicyError
from mlship_runtime.serving.engines.model_files import ModelIntegrityError
from mlship_runtime.serving.main import create_app


def test_tampered_model_file_never_loads(tmp_path: Path) -> None:
    model_dir = saved_model(tmp_path)
    settings = settings_for(model_dir)
    with (model_dir / "model.skops").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ModelIntegrityError, match="do not match"):
        create_app(settings)


def test_missing_expected_hash_refuses_to_load(tmp_path: Path) -> None:
    with pytest.raises(ModelIntegrityError, match="No expected model hash"):
        create_app(settings_for(saved_model(tmp_path), model_sha256=""))


def test_pickled_model_needs_the_run_to_have_allowed_pickle(tmp_path: Path) -> None:
    model_dir = saved_model(tmp_path, allow_pickle=True, model=NotSklearn())
    with pytest.raises(LoadPolicyError, match="did not allow pickle"):
        create_app(settings_for(model_dir, allow_pickle=False))
    create_app(settings_for(model_dir, allow_pickle=True))  # explicit opt-in loads


def _rewrite_mlmodel(model_dir: Path, change: Callable[[dict[str, Any]], None]) -> None:
    """Edit MLmodel as an attacker with write access to the files would."""
    descriptor = Model.load(str(model_dir))
    change({"flavors": descriptor.flavors})
    descriptor.save(str(model_dir / "MLmodel"))


def test_types_outside_the_platform_allowlist_are_refused(tmp_path: Path) -> None:
    model_dir = saved_model(tmp_path)
    _rewrite_mlmodel(
        model_dir,
        lambda d: d["flavors"]["sklearn"].__setitem__("skops_trusted_types", ["os.system"]),
    )
    with pytest.raises(LoadPolicyError, match=r"os\.system"):
        create_app(settings_for(model_dir))  # the hash is recomputed: policy still refuses


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["flavors"]["python_function"].__setitem__(
            "loader_module", "mlflow.pyfunc.model"
        ),
        lambda d: d["flavors"]["python_function"].__setitem__("code", "code"),
    ],
    ids=["pyfunc-loader-swapped", "bundled-code"],
)
def test_a_crafted_mlmodel_cannot_claim_skops_and_load_something_else(
    tmp_path: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    model_dir = saved_model(tmp_path)
    _rewrite_mlmodel(model_dir, change)
    with pytest.raises(LoadPolicyError, match="arbitrary code"):
        create_app(settings_for(model_dir))
