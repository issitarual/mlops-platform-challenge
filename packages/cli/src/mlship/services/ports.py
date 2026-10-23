"""Interfaces implemented by ``repositories/`` and ``adapters/``."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from mlship.domain.environment import ImageRef
from mlship.domain.runs import Manifest, ModelVersion, RunRecord
from mlship.domain.training import TrainingJobResult, TrainingJobSpec


class LockResolver(Protocol):
    def resolve(self, requirement_files: Sequence[Path], output: Path) -> None:
        """Resolve all files together into ``output`` (pinned, with hashes).

        All input files and ``output`` live in the same directory, so the file names
        written into the lock do not depend on where it was generated.
        """
        ...


class RuntimePackager(Protocol):
    """The tool's own code that goes into the image (runtime + sdk)."""

    def requirements(self) -> list[str]: ...

    def source_sha256(self) -> str: ...

    def build_wheels(self, output_dir: Path) -> list[Path]: ...


class DockerfileProvider(Protocol):
    def render(self) -> str: ...


class ContainerRuntime(Protocol):
    def find_image(self, tag: str) -> ImageRef | None: ...

    def build_image(self, context_dir: Path, tag: str, labels: Mapping[str, str]) -> ImageRef: ...

    def verify_image(self, tag: str) -> str | None:
        """Run the image self-check, locked down; ``None`` if ok, else its output."""
        ...

    def remove_image(self, tag: str) -> None: ...

    def run_training(
        self, spec: TrainingJobSpec, on_log: Callable[[str], None]
    ) -> TrainingJobResult:
        """Run the job with the isolation of ADR-018; stream logs to ``on_log``."""
        ...

    def ensure_serving_image(
        self, base: ImageRef, model: ModelVersion, model_dir: Path
    ) -> ImageRef: ...

    def run_serving(self, image: ImageRef, port: int, workers: int) -> None: ...


class RunRepository(Protocol):
    def save(self, manifest: Manifest, output_dir: Path) -> RunRecord: ...

    def list_recent(self, limit: int) -> Sequence[RunRecord]: ...


class ModelRegistry(Protocol):
    def register(self, run_id: str) -> ModelVersion: ...

    def set_alias(self, version: int, alias: str) -> None: ...

    def resolve_alias(self, alias: str) -> ModelVersion: ...

    def download(self, model: ModelVersion, target_dir: Path) -> Path: ...
