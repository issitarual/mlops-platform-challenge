"""Renders the environment Dockerfile from the packaged template."""

from importlib import resources
from string import Template


class EnvironmentDockerfile:
    def __init__(self, base_image: str) -> None:
        self._base_image = base_image

    def render(self) -> str:
        template = (
            resources.files("mlship.adapters.templates")
            .joinpath("environment.Dockerfile")
            .read_text(encoding="utf-8")
        )
        return Template(template).substitute(base_image=self._base_image)
