"""Builds (or reuses) the image for a project: one lock -> one image."""

from pathlib import Path

from mlship.domain.environment import ImageRef
from mlship.services.ports import ContainerRuntime, LockResolver


class EnvironmentService:
    def __init__(self, lock_resolver: LockResolver, container_runtime: ContainerRuntime) -> None:
        self._lock_resolver = lock_resolver
        self._container_runtime = container_runtime

    def prepare(self, project_dir: Path) -> ImageRef:
        raise NotImplementedError("E1")
