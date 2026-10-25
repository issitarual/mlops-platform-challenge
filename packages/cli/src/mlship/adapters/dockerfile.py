"""Renders the Dockerfiles from the packaged templates."""

import json
import re
from importlib import resources
from string import Template

from mlship.domain.serving import ServingImageSpec

_SAFE_VALUE = re.compile(r"^[A-Za-z0-9._-]+$")


def _template(name: str) -> Template:
    return Template(
        resources.files("mlship.adapters.templates").joinpath(name).read_text(encoding="utf-8")
    )


class EnvironmentDockerfile:
    def __init__(self, base_image: str) -> None:
        self._base_image = base_image

    def render(self) -> str:
        return _template("environment.Dockerfile").substitute(base_image=self._base_image)


def render_serving_dockerfile(spec: ServingImageSpec) -> str:
    """Every value is validated before it is written into the Dockerfile."""
    model = spec.model
    plain = {
        "model_name": model.name,
        "model_version": str(model.version),
        "model_sha256": model.model_sha256,
    }
    for key, value in plain.items():
        if not _SAFE_VALUE.match(value):
            raise ValueError(f"unsafe {key} for a Dockerfile: {value!r}")
    trusted = json.dumps(list(spec.skops_trusted_types), separators=(",", ":"))
    if "'" in trusted or "\n" in trusted:
        raise ValueError("unsafe skops trusted types for a Dockerfile")
    return _template("serving.Dockerfile").substitute(
        base_image=spec.base.tag,
        allow_pickle="true" if spec.allow_pickle else "false",
        skops_trusted_types=trusted,
        **plain,
    )
