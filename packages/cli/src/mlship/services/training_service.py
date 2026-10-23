"""Validate -> prepare environment -> run isolated job -> read the result (E3: register)."""

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from mlship.domain.errors import MlshipError, ProjectInvalidError
from mlship.domain.training import (
    CONFIG_MOUNT,
    CONTAINER_TMP,
    DATA_MOUNT,
    OUTPUT_MOUNT,
    PROJECT_MOUNT,
    RUN_CONFIG_SCHEMA_VERSION,
    ResourceLimits,
    TrainingJobSpec,
    TrainingOutcome,
)
from mlship.services.environment_service import EnvironmentService
from mlship.services.ports import ContainerRuntime, RunRepository
from mlship.services.training_result import parse_result
from mlship.services.validation.validator import ProjectValidator

DEFAULT_SEED = 42


@dataclass(frozen=True, slots=True)
class TrainingPolicy:
    """What the platform allows (set by whoever operates mlship, never by train.py)."""

    limits: ResourceLimits
    skops_trusted_types: tuple[str, ...]
    allow_pickle_enabled: bool
    runs_dirname: str = ".mlship/runs"


class TrainingService:
    def __init__(
        self,
        validator: ProjectValidator,
        environment: EnvironmentService,
        container_runtime: ContainerRuntime,
        run_repository: RunRepository,
        policy: TrainingPolicy,
    ) -> None:
        self._validator = validator
        self._environment = environment
        self._container_runtime = container_runtime
        self._run_repository = run_repository
        self._policy = policy

    def train(
        self,
        project_dir: Path,
        data_dir: Path,
        params: Mapping[str, str],
        *,
        allow_pickle: bool = False,
        on_log: Callable[[str], None] = lambda _: None,
    ) -> TrainingOutcome:
        report = self._validator.validate(project_dir, data_dir)
        if report.blocks(strict=False):
            raise ProjectInvalidError(report)
        if allow_pickle and not self._policy.allow_pickle_enabled:
            raise MlshipError(
                "--allow-pickle is disabled by the platform configuration.",
                hint="Use a scikit-learn model (saved safely with skops), or ask the "
                "platform team (MLSHIP_ALLOW_PICKLE_ENABLED).",
            )
        params = dict(params)
        seed = self._seed(params)
        params["seed"] = str(seed)  # train() always sees the seed that was actually used

        environment = self._environment.prepare(project_dir)

        run_id = uuid.uuid4().hex
        run_dir = project_dir / self._policy.runs_dirname / run_id
        output_dir = run_dir / "output"
        output_dir.mkdir(parents=True)
        config_file = run_dir / "run.json"
        config_file.write_text(
            json.dumps(
                {
                    "schema_version": RUN_CONFIG_SCHEMA_VERSION,
                    "project_dir": PROJECT_MOUNT,
                    "data_dir": DATA_MOUNT,
                    "output_dir": OUTPUT_MOUNT,
                    "params": params,
                    "seed": seed,
                    "serialization": {
                        "allow_pickle": allow_pickle,
                        "skops_trusted_types": list(self._policy.skops_trusted_types),
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        threads = str(max(1, int(self._policy.limits.cpus)))
        spec = TrainingJobSpec(
            image_tag=environment.image.tag,
            project_dir=project_dir,
            data_dir=data_dir,
            config_file=config_file,
            output_dir=output_dir,
            environment={
                "MLSHIP_RUN_CONFIG": CONFIG_MOUNT,
                "PYTHONHASHSEED": str(seed),
                "HOME": CONTAINER_TMP,
                "OMP_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
            },
            limits=self._policy.limits,
            run_id=run_id,
        )
        job = self._container_runtime.run_training(spec, on_log)
        limits = self._policy.limits
        parsed = parse_result(
            output_dir / "result.json",
            job,
            f"timeout {limits.timeout_seconds}s, memory {limits.memory}: "
            "MLSHIP_TRAIN_TIMEOUT_SECONDS / MLSHIP_TRAIN_MEMORY_LIMIT",
        )
        return TrainingOutcome(
            run_id=run_id,
            run_dir=run_dir,
            succeeded=parsed.succeeded,
            issues=parsed.issues,
            warnings=report.warnings,
            metrics=parsed.metrics,
            serialization=parsed.serialization,
            environment=environment,
            job=job,
            params=params,
        )

    @staticmethod
    def _seed(params: Mapping[str, str]) -> int:
        raw = params.get("seed")
        if raw is None:
            return DEFAULT_SEED
        try:
            return int(raw)
        except ValueError as exc:
            raise MlshipError(
                f"--param seed={raw} is not an integer.", hint="Example: -p seed=42"
            ) from exc
