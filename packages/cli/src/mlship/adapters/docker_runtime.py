"""``ContainerRuntime`` implemented with the Docker SDK for Python.

The client is created on first use, so building the object never needs a Docker daemon.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import docker
import docker.errors

from mlship.domain.environment import ImageRef
from mlship.domain.errors import EnvironmentBuildError, ToolNotFoundError
from mlship.domain.runs import ModelVersion
from mlship.domain.training import TrainingJobResult, TrainingJobSpec

_LOG_TAIL_CHARS = 4_000


class DockerRuntime:
    def __init__(self, platform: str, client_factory: Callable[[], Any] = docker.from_env) -> None:
        self._platform = platform
        self._client_factory = client_factory  # injectable, so error mapping is unit-tested
        self._client: Any = None

    def _docker(self) -> Any:
        if self._client is None:
            try:
                self._client = self._client_factory()
                self._client.ping()
            except docker.errors.DockerException as exc:
                self._client = None
                raise ToolNotFoundError(
                    "Could not connect to Docker.",
                    hint="Start Docker Desktop (or the Docker daemon) and try again.",
                    details=str(exc),
                ) from exc
        return self._client

    def find_image(self, tag: str) -> ImageRef | None:
        try:
            image = self._docker().images.get(tag)
        except docker.errors.ImageNotFound:
            return None
        return ImageRef(tag=tag, digest=image.id)

    def build_image(self, context_dir: Path, tag: str, labels: Mapping[str, str]) -> ImageRef:
        try:
            image, _ = self._docker().images.build(
                path=str(context_dir),
                tag=tag,
                labels=dict(labels),
                platform=self._platform,
                rm=True,
                forcerm=True,
            )
        except docker.errors.BuildError as exc:
            log = "".join(str(chunk.get("stream", "")) for chunk in exc.build_log)
            raise EnvironmentBuildError(
                "Building the environment image failed.",
                hint="See the build log below; dependency installation is the usual cause.",
                details=f"{log}\n{exc.msg}"[-_LOG_TAIL_CHARS:],
            ) from exc
        except docker.errors.APIError as exc:
            raise EnvironmentBuildError("Docker rejected the build.", details=str(exc)) from exc
        return ImageRef(tag=tag, digest=image.id)

    def run_training(self, spec: TrainingJobSpec) -> TrainingJobResult:
        """E2: --network none, non-root, read-only rootfs, cap-drop ALL, limits."""
        raise NotImplementedError("E2")

    def ensure_serving_image(
        self, base: ImageRef, model: ModelVersion, model_dir: Path
    ) -> ImageRef:
        raise NotImplementedError("E4")

    def run_serving(self, image: ImageRef, port: int, workers: int) -> None:
        raise NotImplementedError("E4")
