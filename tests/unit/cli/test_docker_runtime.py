"""Error mapping of the Docker adapter, with a fake client (no Docker needed)."""

from pathlib import Path
from typing import Any

import docker.errors
import pytest

from mlship.adapters.docker_runtime import DockerRuntime
from mlship.domain.environment import ImageRef
from mlship.domain.errors import EnvironmentBuildError, ToolNotFoundError


class FakeImage:
    id = "sha256:feed"


class FakeImages:
    def __init__(self, build_error: Exception | None = None, present: bool = False) -> None:
        self.build_error = build_error
        self.present = present
        self.build_kwargs: dict[str, Any] = {}

    def get(self, tag: str) -> FakeImage:
        if not self.present:
            raise docker.errors.ImageNotFound(f"No such image: {tag}")
        return FakeImage()

    def build(self, **kwargs: Any) -> tuple[FakeImage, list[dict[str, str]]]:
        self.build_kwargs = kwargs
        if self.build_error is not None:
            raise self.build_error
        return FakeImage(), []


class FakeClient:
    def __init__(self, images: FakeImages) -> None:
        self.images = images

    def ping(self) -> bool:
        return True


def _runtime(images: FakeImages) -> DockerRuntime:
    return DockerRuntime("linux/amd64", client_factory=lambda: FakeClient(images))


def test_missing_image_is_none_not_an_error() -> None:
    assert _runtime(FakeImages()).find_image("mlship-env:abc") is None


def test_present_image_is_returned_with_its_id() -> None:
    found = _runtime(FakeImages(present=True)).find_image("mlship-env:abc")
    assert found == ImageRef(tag="mlship-env:abc", digest="sha256:feed")


def test_build_passes_tag_labels_and_platform(tmp_path: Path) -> None:
    images = FakeImages()
    image = _runtime(images).build_image(tmp_path, "mlship-env:abc", {"org.mlship.k": "v"})
    assert image == ImageRef(tag="mlship-env:abc", digest="sha256:feed")
    assert images.build_kwargs["tag"] == "mlship-env:abc"
    assert images.build_kwargs["labels"] == {"org.mlship.k": "v"}
    assert images.build_kwargs["platform"] == "linux/amd64"


def test_failed_build_carries_the_log_and_dockers_reason(tmp_path: Path) -> None:
    error = docker.errors.BuildError(
        "COPY failed: file not found", [{"stream": "Step 2/9 : COPY requirements.lock\n"}]
    )
    with pytest.raises(EnvironmentBuildError) as info:
        _runtime(FakeImages(build_error=error)).build_image(tmp_path, "t:1", {})
    assert "Step 2/9" in info.value.details
    assert "COPY failed" in info.value.details


def test_unreachable_daemon_becomes_a_clear_error() -> None:
    def unreachable() -> FakeClient:
        raise docker.errors.DockerException("Error while fetching server API version")

    runtime = DockerRuntime("linux/amd64", client_factory=unreachable)
    with pytest.raises(ToolNotFoundError) as info:
        runtime.find_image("t:1")
    assert "Docker Desktop" in info.value.hint
