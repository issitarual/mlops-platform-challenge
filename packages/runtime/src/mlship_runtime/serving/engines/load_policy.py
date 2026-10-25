"""Which serialized models may be loaded (ADR-017), checked from ``MLmodel`` alone.

``MLmodel`` was written in the same process as the scientist's code, so it is data,
not a decision: the allowlist comes from the platform (settings). Reading the
descriptor (YAML, parsed safely by MLflow) never runs code; loading the model may,
which is why this runs first.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SAFE_SKLEARN = "skops"


class LoadPolicyError(Exception):
    """The model needs a trust level the platform did not grant."""


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    flavors: dict[str, Any]
    signature: dict[str, Any] | None


def read_descriptor(model_dir: Path) -> ModelDescriptor:
    from mlflow.models import Model

    try:
        descriptor = Model.load(str(model_dir))
    except Exception as exc:  # malformed YAML or missing file: not a loadable model
        raise LoadPolicyError(f"MLmodel cannot be read: {exc}") from exc
    signature = descriptor.signature.to_dict() if descriptor.signature else None
    return ModelDescriptor(dict(descriptor.flavors or {}), signature)


def check(descriptor: ModelDescriptor, *, allow_pickle: bool, trusted_types: list[str]) -> None:
    """Raise unless the model loads with skops inside the allowlist, or pickle is allowed.

    ``mlflow.pyfunc.load_model`` follows the ``python_function`` flavor, not ``sklearn``:
    the safe path also requires that flavor to delegate to ``mlflow.sklearn`` and to
    bring no code of its own. Otherwise a crafted MLmodel could claim skops while
    loading a pickled object or importing arbitrary modules.
    """
    flavors = descriptor.flavors
    pyfunc = flavors.get("python_function")
    sklearn = flavors.get("sklearn")
    if not isinstance(pyfunc, dict):
        raise LoadPolicyError("MLmodel has no python_function flavor to serve.")
    if not isinstance(sklearn, dict):
        sklearn = {}
    safe_path = (
        sklearn.get("serialization_format") == _SAFE_SKLEARN
        and pyfunc.get("loader_module") == "mlflow.sklearn"
        and not pyfunc.get("code")
        and not sklearn.get("code")
    )
    if safe_path:
        requested = sklearn.get("skops_trusted_types") or []
        if not isinstance(requested, list):
            raise LoadPolicyError("skops_trusted_types in MLmodel is malformed.")
        outside = sorted(set(map(str, requested)) - set(trusted_types))
        if outside:
            raise LoadPolicyError(
                "The model asks to trust types outside the platform allowlist: "
                + ", ".join(outside)
            )
        return
    if not allow_pickle:
        raise LoadPolicyError(
            "The model is not a plain skops scikit-learn model, so loading it can run "
            "arbitrary code (pickle or bundled code); the run that produced it did not "
            "allow pickle."
        )
