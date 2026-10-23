"""Helpers for TrainingService tests: a project on disk and a recording runtime."""

import json
from collections.abc import Callable
from pathlib import Path

from fakes import FakeContainerRuntime, FakeDockerfile, FakeLockResolver, FakeRuntimePackager

from mlship.domain.training import ResourceLimits, TrainingJobResult, TrainingJobSpec
from mlship.services.environment_service import EnvironmentService
from mlship.services.training_service import TrainingPolicy, TrainingService
from mlship.services.validation.validator import ProjectValidator

LIMITS = ResourceLimits(cpus=2.0, memory="4g", pids=256, timeout_seconds=60, tmpfs_size="1g")
GOOD_RESULT = {
    "schema_version": 1,
    "status": "succeeded",
    "errors": [],
    "metrics": {"rmse": 0.5},
    "model": {"path": "model", "serialization": "skops", "trusted_types": []},
}


def make_project(root: Path) -> tuple[Path, Path]:
    project, data = root / "project", root / "data"
    project.mkdir()
    data.mkdir()
    (project / "train.py").write_text("def train(data_dir, params):\n    return None\n")
    (project / "requirements.txt").write_text("scikit-learn==1.9.1\n")
    (data / "data.csv").write_text("a\n1\n")
    return project, data


class RecordingTrainingRuntime(FakeContainerRuntime):
    """Pretends to run the container: writes the given result.json and a model dir."""

    def __init__(self, result: dict[str, object] | None = None, exit_code: int = 0) -> None:
        super().__init__()
        self.result = GOOD_RESULT if result is None else result
        self.exit_code = exit_code
        self.specs: list[TrainingJobSpec] = []

    def run_training(
        self, spec: TrainingJobSpec, on_log: Callable[[str], None]
    ) -> TrainingJobResult:
        self.specs.append(spec)
        on_log("training...\n")
        (spec.output_dir / "result.json").write_text(json.dumps(self.result))
        (spec.output_dir / "model").mkdir()
        (spec.output_dir / "model" / "MLmodel").write_text("flavors: {}\n")
        return TrainingJobResult(self.exit_code, False, False, "training...\n")


def make_service(
    runtime: RecordingTrainingRuntime, *, allow_pickle_enabled: bool = True
) -> TrainingService:
    environment = EnvironmentService(
        FakeLockResolver(),
        FakeRuntimePackager(),
        FakeDockerfile(),
        runtime,
        python_version="3.12",
        python_platform="x86_64-manylinux_2_28",
        docker_platform="linux/amd64",
    )
    policy = TrainingPolicy(LIMITS, ("a.B",), allow_pickle_enabled)
    return TrainingService(ProjectValidator(), environment, runtime, None, policy)  # type: ignore[arg-type]
