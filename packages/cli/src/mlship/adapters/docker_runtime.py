"""``ContainerRuntime`` implemented with the Docker SDK for Python.

The client is created lazily so building the object never needs a Docker daemon.
"""

from pathlib import Path

from mlship.domain.environment import ImageRef, ResolvedLock
from mlship.domain.runs import ModelVersion
from mlship.domain.training import TrainingJobResult, TrainingJobSpec


class DockerRuntime:
    def ensure_image(self, lock: ResolvedLock) -> ImageRef:
        raise NotImplementedError("E1")

    def run_training(self, spec: TrainingJobSpec) -> TrainingJobResult:
        """E2: --network none, non-root, read-only rootfs, cap-drop ALL, limits."""
        raise NotImplementedError("E2")

    def ensure_serving_image(
        self, base: ImageRef, model: ModelVersion, model_dir: Path
    ) -> ImageRef:
        raise NotImplementedError("E4")

    def run_serving(self, image: ImageRef, port: int, workers: int) -> None:
        raise NotImplementedError("E4")
