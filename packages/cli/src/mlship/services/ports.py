"""Interfaces implemented by ``repositories/`` and ``adapters/``."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from mlship.domain.environment import ImageRef
from mlship.domain.runs import ModelVersion, RunRecord, RunStatus, ToolInfo
from mlship.domain.serving import ServingImageSpec, ServingJobSpec
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

    def ensure_serving_image(self, spec: ServingImageSpec, model_dir: Path) -> ImageRef:
        """``FROM`` the environment image + the verified model; reused when it exists."""
        ...

    def run_serving(
        self,
        spec: ServingJobSpec,
        on_log: Callable[[str], None],
        on_ready: Callable[[str], None],
    ) -> None:
        """Run in the foreground until Ctrl+C; an exit on its own is an error."""
        ...


class ContentHasher(Protocol):
    def file(self, path: Path) -> str: ...

    def tree(
        self,
        root: Path,
        exclude_dirs: frozenset[str] = frozenset(),
        include: Callable[[str], bool] | None = None,
    ) -> dict[str, str]: ...

    def untrusted_tree(self, root: Path, max_bytes: int, max_files: int) -> dict[str, str]:
        """Hash output of untrusted code; ``UntrustedArtifactError`` on links or excess."""
        ...


class ToolInfoProvider(Protocol):
    def info(self) -> ToolInfo: ...


class RunRepository(Protocol):
    """Experiment tracking: one experiment per model name, one run per ``train``."""

    def check_available(self) -> None:
        """Fail fast (``TrackingUnavailableError``) before any expensive work."""
        ...

    def start(self, experiment: str, tags: Mapping[str, str]) -> str:
        """Create a RUNNING run and return its id."""
        ...

    def log(
        self,
        run_id: str,
        *,
        params: Mapping[str, str],
        metrics: Mapping[str, float],
        tags: Mapping[str, str],
        artifacts: Mapping[str, Path],
    ) -> None:
        """Log values and upload ``artifacts`` (name -> file or directory)."""
        ...

    def end(self, run_id: str, status: RunStatus, tags: Mapping[str, str]) -> None: ...

    def list_recent(self, experiment: str, limit: int) -> Sequence[RunRecord]: ...

    def get_tags(self, run_id: str) -> Mapping[str, str]:
        """Tags live in the tracking database, apart from the artifact store."""
        ...

    def download_artifact(self, run_id: str, path: str, target_dir: Path) -> Path | None:
        """Download one artifact of a run; ``None`` when the run does not have it."""
        ...


class ModelRegistry(Protocol):
    def register(self, name: str, run_id: str, model_sha256: str) -> ModelVersion:
        """New immutable version from the run's ``model`` artifact."""
        ...

    def get_version(self, name: str, version: int) -> ModelVersion: ...

    def get_alias(self, name: str, alias: str) -> ModelVersion | None: ...

    def set_alias(self, name: str, alias: str, version: int, tags: Mapping[str, str]) -> None:
        """Point ``alias`` at ``version`` and record ``tags`` on that version."""
        ...

    def download(self, model: ModelVersion, target_dir: Path) -> Path:
        """Download the model files of a version into ``target_dir``; return their folder."""
        ...
