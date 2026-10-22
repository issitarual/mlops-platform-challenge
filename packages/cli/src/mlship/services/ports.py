"""Interfaces implemented by ``repositories/`` and ``adapters/``.

Signatures are provisional and get refined in E1-E4.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from mlship.domain.environment import ImageRef, ResolvedLock
from mlship.domain.runs import Manifest, ModelVersion, RunRecord
from mlship.domain.training import TrainingJobResult, TrainingJobSpec


class LockResolver(Protocol):
    def resolve(self, requirements_file: Path, output_dir: Path) -> ResolvedLock: ...


class ContainerRuntime(Protocol):
    def ensure_image(self, lock: ResolvedLock) -> ImageRef: ...

    def run_training(self, spec: TrainingJobSpec) -> TrainingJobResult: ...

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
