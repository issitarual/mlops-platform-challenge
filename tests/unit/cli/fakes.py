"""In-memory implementations of the ports, for service tests without Docker or uv."""

import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlship.domain.environment import ImageRef
from mlship.domain.errors import ModelNotFoundError, TrackingUnavailableError
from mlship.domain.runs import ModelVersion, RunRecord, RunStatus, ToolInfo
from mlship.domain.serving import ServingImageSpec, ServingJobSpec
from mlship.domain.training import TrainingJobResult, TrainingJobSpec


class FakeLockResolver:
    def __init__(self, content: str = "numpy==2.0.0 \\\n    --hash=sha256:abc\n") -> None:
        self.content = content
        self.calls: list[list[str]] = []

    def resolve(self, requirement_files: Sequence[Path], output: Path) -> None:
        self.calls.append([f.read_text() for f in requirement_files])
        output.write_text(self.content)


class FakeRuntimePackager:
    def __init__(self, source: str = "s" * 64) -> None:
        self.source = source
        self.built: list[Path] = []

    def requirements(self) -> list[str]:
        return ["pydantic>=2", "fastapi>=0.1"]

    def source_sha256(self) -> str:
        return self.source

    def build_wheels(self, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True)
        wheel = output_dir / "mlship_runtime-0.1.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        self.built.append(wheel)
        return [wheel]


class FakeDockerfile:
    def __init__(self, text: str = "FROM python:3.12-slim@sha256:abc\n") -> None:
        self.text = text

    def render(self) -> str:
        return self.text


class FakeContainerRuntime:
    def __init__(self, selfcheck_failure: str | None = None) -> None:
        self.images: dict[str, ImageRef] = {}
        self.builds: list[tuple[str, dict[str, str], list[str]]] = []
        self.selfcheck_failure = selfcheck_failure
        self.verified: list[str] = []
        self.serving_specs: list[tuple[ServingImageSpec, list[str]]] = []
        self.serving_jobs: list[ServingJobSpec] = []

    def find_image(self, tag: str) -> ImageRef | None:
        return self.images.get(tag)

    def build_image(self, context_dir: Path, tag: str, labels: Mapping[str, str]) -> ImageRef:
        files = sorted(p.relative_to(context_dir).as_posix() for p in context_dir.rglob("*"))
        self.builds.append((tag, dict(labels), files))
        image = ImageRef(tag=tag, digest=f"sha256:{len(self.builds)}")
        self.images[tag] = image
        return image

    def verify_image(self, tag: str) -> str | None:
        self.verified.append(tag)
        return self.selfcheck_failure

    def remove_image(self, tag: str) -> None:
        self.images.pop(tag, None)

    def run_training(
        self, spec: TrainingJobSpec, on_log: Callable[[str], None]
    ) -> TrainingJobResult:
        raise NotImplementedError

    def ensure_serving_image(self, spec: ServingImageSpec, model_dir: Path) -> ImageRef:
        self.serving_specs.append((spec, sorted(p.name for p in model_dir.rglob("*"))))
        tag = f"mlship-serve-{spec.model.name}:v{spec.model.version}"
        return self.images.setdefault(tag, ImageRef(tag, f"sha256:serve{spec.model.version}"))

    def run_serving(
        self,
        spec: ServingJobSpec,
        on_log: Callable[[str], None],
        on_ready: Callable[[str], None],
    ) -> None:
        self.serving_jobs.append(spec)
        on_ready(f"http://127.0.0.1:{spec.host_port}")


class FakeRunRepository:
    """In-memory tracking: records every call, like an MLflow server would keep it."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.runs: dict[str, dict[str, Any]] = {}
        self.checks = 0
        # Uploaded artifacts are copied, like an artifact store: later local changes
        # do not affect them, and tests can tamper with the "store" itself.
        self.store = Path(tempfile.mkdtemp(prefix="fake-artifacts-"))

    def check_available(self) -> None:
        self.checks += 1
        if not self.available:
            raise TrackingUnavailableError("MLflow is not reachable.", hint="start it")

    def start(self, experiment: str, tags: Mapping[str, str]) -> str:
        run_id = f"{len(self.runs) + 1:032x}"
        self.runs[run_id] = {
            "experiment": experiment,
            "status": RunStatus.RUNNING,
            "tags": dict(tags),
            "params": {},
            "metrics": {},
            "artifacts": {},
        }
        return run_id

    def log(
        self,
        run_id: str,
        *,
        params: Mapping[str, str],
        metrics: Mapping[str, float],
        tags: Mapping[str, str],
        artifacts: Mapping[str, Path],
    ) -> None:
        run = self.runs[run_id]
        run["params"].update(params)
        run["metrics"].update(metrics)
        run["tags"].update(tags)
        run["artifacts"].update(artifacts)
        for name, source in artifacts.items():
            target = self.store / run_id / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copyfile(source, target)

    def end(self, run_id: str, status: RunStatus, tags: Mapping[str, str]) -> None:
        self.runs[run_id]["status"] = status
        self.runs[run_id]["tags"].update(tags)

    def get_tags(self, run_id: str) -> Mapping[str, str]:
        return dict(self.runs[run_id]["tags"])

    def download_artifact(self, run_id: str, path: str, target_dir: Path) -> Path | None:
        source = self.store / run_id / path
        if not source.exists():
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target_dir / path)
        return target_dir / path

    def list_recent(self, experiment: str, limit: int) -> Sequence[RunRecord]:
        return [
            RunRecord(
                run_id, datetime(2026, 10, 23, tzinfo=UTC), run["status"], run["metrics"], None
            )
            for run_id, run in reversed(self.runs.items())
            if run["experiment"] == experiment
        ][:limit]


class FakeModelRegistry:
    def __init__(self, runs: "FakeRunRepository | None" = None) -> None:
        self.runs = runs  # where download() finds the model files
        self.versions: dict[str, list[ModelVersion]] = {}
        self.aliases: dict[tuple[str, str], int] = {}
        self.version_tags: dict[tuple[str, int], dict[str, str]] = {}

    def register(self, name: str, run_id: str, model_sha256: str) -> ModelVersion:
        versions = self.versions.setdefault(name, [])
        created = ModelVersion(name, len(versions) + 1, run_id, model_sha256)
        versions.append(created)
        return created

    def get_version(self, name: str, version: int) -> ModelVersion:
        for candidate in self.versions.get(name, []):
            if candidate.version == version:
                return candidate
        raise ModelNotFoundError(f"{name} has no version {version}.")

    def get_alias(self, name: str, alias: str) -> ModelVersion | None:
        version = self.aliases.get((name, alias))
        return None if version is None else self.get_version(name, version)

    def set_alias(self, name: str, alias: str, version: int, tags: Mapping[str, str]) -> None:
        self.aliases[(name, alias)] = version
        self.version_tags.setdefault((name, version), {}).update(tags)

    def download(self, model: ModelVersion, target_dir: Path) -> Path:
        assert self.runs is not None, "give the registry the run repository to download"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.runs.store / model.run_id / "model", target_dir / "model")
        return target_dir / "model"


class FakeToolInfo:
    def info(self) -> ToolInfo:
        return ToolInfo("0.1.0", "abc123", False)
