"""Validate -> open the run -> prepare environment -> isolated job -> manifest -> register.

The MLflow run is opened before any expensive step, so its id names the run on disk
and failures are tracked too (ADR-019).
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mlship.domain.errors import (
    ArtifactTooLargeError,
    MlshipError,
    ProjectInvalidError,
    TrackingError,
    UntrustedArtifactError,
)
from mlship.domain.params import param_issues
from mlship.domain.runs import Manifest, ToolInfo
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
from mlship.domain.validation import Severity, ValidationIssue, ValidationReport
from mlship.services.environment_service import EnvironmentService
from mlship.services.lineage import (
    CODE_EXCLUDED_DIRS,
    CodeSelector,
    data_location,
    issue_dicts,
    write_manifest,
)
from mlship.services.ports import ContainerRuntime, ContentHasher, ToolInfoProvider
from mlship.services.run_recorder import RunRecorder
from mlship.services.training_result import parse_result
from mlship.services.validation.validator import ProjectValidator

DEFAULT_SEED = 42
_SAFE_SERIALIZATION = "skops"


@dataclass(frozen=True, slots=True)
class TrainingPolicy:
    """What the platform allows (set by whoever operates mlship, never by train.py)."""

    limits: ResourceLimits
    skops_trusted_types: tuple[str, ...]
    allow_pickle_enabled: bool
    model_max_bytes: int = 2 * 1024**3
    model_max_files: int = 10_000
    runs_dirname: str = ".mlship/runs"


@dataclass(frozen=True, slots=True)
class _RunInputs:
    project_dir: Path
    data_dir: Path
    model_name: str
    params: dict[str, str]
    seed: int
    allow_pickle: bool
    warnings: tuple[ValidationIssue, ...]
    tool: ToolInfo


class TrainingService:
    def __init__(
        self,
        validator: ProjectValidator,
        environment: EnvironmentService,
        container_runtime: ContainerRuntime,
        recorder: RunRecorder,
        hasher: ContentHasher,
        tool_info: ToolInfoProvider,
        policy: TrainingPolicy,
    ) -> None:
        self._validator = validator
        self._environment = environment
        self._container_runtime = container_runtime
        self._recorder = recorder
        self._hasher = hasher
        self._tool_info = tool_info
        self._policy = policy

    def train(
        self,
        project_dir: Path,
        data_dir: Path,
        params: Mapping[str, str],
        *,
        model_name: str,
        allow_pickle: bool = False,
        strict: bool = False,
        on_log: Callable[[str], None] = lambda _: None,
    ) -> TrainingOutcome:
        report = self._validator.validate(project_dir, data_dir)
        report = ValidationReport((*report.issues, *param_issues(params)))
        if report.blocks(strict=strict):
            raise ProjectInvalidError(report)
        if allow_pickle and not self._policy.allow_pickle_enabled:
            raise MlshipError(
                "--allow-pickle is disabled by the platform configuration.",
                hint="Use a scikit-learn model (saved safely with skops), or ask the "
                "platform team (MLSHIP_ALLOW_PICKLE_ENABLED).",
            )
        all_params = dict(params)
        seed = self._seed(all_params)
        all_params["seed"] = str(seed)  # train() always sees the seed that was actually used

        self._recorder.check_available()  # before the lock, the build and hours of training
        tool = self._tool_info.info()
        run_id = self._recorder.open(model_name, tool)
        inputs = _RunInputs(
            project_dir=project_dir,
            data_dir=data_dir,
            model_name=model_name,
            params=all_params,
            seed=seed,
            allow_pickle=allow_pickle,
            warnings=report.warnings,
            tool=tool,
        )
        try:
            outcome, manifest = self._execute(run_id, inputs, on_log)
        except BaseException as exc:
            self._recorder.close(run_id, exc)
            raise
        try:
            return self._recorder.record(outcome, manifest)
        except TrackingError as exc:
            self._recorder.close(run_id, exc)
            raise MlshipError(
                "Training finished, but recording it in MLflow failed.",
                hint=f"The model and the manifest are kept in {outcome.run_dir}.",
                details=f"{exc.message}\n{exc.details}",
            ) from exc
        except BaseException as exc:  # e.g. Ctrl+C during the upload: never leave it RUNNING
            self._recorder.close(run_id, exc)
            raise

    def _execute(
        self, run_id: str, inputs: _RunInputs, on_log: Callable[[str], None]
    ) -> tuple[TrainingOutcome, Manifest]:
        environment = self._environment.prepare(inputs.project_dir)
        run_dir = inputs.project_dir / self._policy.runs_dirname / run_id
        output_dir = run_dir / "output"
        output_dir.mkdir(parents=True)
        config_file = run_dir / "run.json"
        config_file.write_text(self._run_config(inputs), encoding="utf-8")

        code = CodeSelector(inputs.project_dir, inputs.data_dir)
        code_files = self._hasher.tree(inputs.project_dir, CODE_EXCLUDED_DIRS, code)
        data_files = self._hasher.tree(inputs.data_dir)

        limits = self._policy.limits
        threads = str(max(1, int(limits.cpus)))
        spec = TrainingJobSpec(
            image_tag=environment.image.tag,
            project_dir=inputs.project_dir,
            data_dir=inputs.data_dir,
            config_file=config_file,
            output_dir=output_dir,
            environment={
                "MLSHIP_RUN_CONFIG": CONFIG_MOUNT,
                "PYTHONHASHSEED": str(inputs.seed),
                "HOME": CONTAINER_TMP,
                "OMP_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
            },
            limits=limits,
            run_id=run_id,
        )
        job = self._container_runtime.run_training(spec, on_log)
        parsed = parse_result(
            output_dir / "result.json",
            job,
            f"timeout {limits.timeout_seconds}s, memory {limits.memory}: "
            "MLSHIP_TRAIN_TIMEOUT_SECONDS / MLSHIP_TRAIN_MEMORY_LIMIT",
        )
        succeeded, issues = parsed.succeeded, parsed.issues
        model_files: dict[str, str] = {}
        if succeeded and parsed.serialization != _SAFE_SERIALIZATION and not inputs.allow_pickle:
            succeeded = False
            issues = (
                ValidationIssue(
                    "MODEL_SERIALIZATION_NOT_ALLOWED",
                    Severity.ERROR,
                    f"The model reports {parsed.serialization!r} serialization, "
                    "but --allow-pickle was not given.",
                    "result.json",
                    "Use a scikit-learn model (saved with skops), or train with --allow-pickle "
                    "if you trust this code.",
                ),
            )
        if succeeded:
            try:
                model_files = self._hasher.untrusted_tree(
                    output_dir / "model", self._policy.model_max_bytes, self._policy.model_max_files
                )
            except UntrustedArtifactError as exc:
                code_name = (
                    "MODEL_TOO_LARGE"
                    if isinstance(exc, ArtifactTooLargeError)
                    else "MODEL_UNTRUSTED_FILES"
                )
                succeeded = False
                issues = (
                    ValidationIssue(code_name, Severity.ERROR, exc.message, "model/", exc.hint),
                )

        manifest = Manifest(
            run_id=run_id,
            model_name=inputs.model_name,
            status="succeeded" if succeeded else "failed",
            created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
            tool=inputs.tool,
            code_files=code_files,
            code_skipped=code.skipped,
            lock_sha256=environment.lock.sha256,
            lock_inputs_sha256=environment.lock.inputs_sha256,
            environment={
                "image_tag": environment.image.tag,
                "image_id": environment.image.digest,
                "environment_key": environment.inputs.key(),
                "runtime_source_sha256": environment.inputs.runtime_source_sha256,
                "dockerfile_sha256": environment.inputs.dockerfile_sha256,
                "platform": environment.inputs.platform,
            },
            data_location=data_location(inputs.project_dir, inputs.data_dir),
            data_files=data_files,
            params=inputs.params,
            seed=inputs.seed,
            limits={
                "cpus": limits.cpus,
                "memory": limits.memory,
                "pids": limits.pids,
                "timeout_seconds": limits.timeout_seconds,
            },
            metrics=parsed.metrics,
            model_serialization=parsed.serialization if succeeded else None,
            allow_pickle=inputs.allow_pickle,
            model_files=model_files,
            errors=issue_dicts(issues),
            warnings=issue_dicts(inputs.warnings),
        )
        write_manifest(manifest, run_dir / "manifest.json")
        outcome = TrainingOutcome(
            run_id=run_id,
            run_dir=run_dir,
            succeeded=succeeded,
            issues=issues,
            warnings=inputs.warnings,
            metrics=parsed.metrics,
            serialization=manifest.model_serialization,
            environment=environment,
            job=job,
            params=inputs.params,
            model_name=inputs.model_name,
        )
        return outcome, manifest

    def _run_config(self, inputs: _RunInputs) -> str:
        return json.dumps(
            {
                "schema_version": RUN_CONFIG_SCHEMA_VERSION,
                "project_dir": PROJECT_MOUNT,
                "data_dir": DATA_MOUNT,
                "output_dir": OUTPUT_MOUNT,
                "params": inputs.params,
                "seed": inputs.seed,
                "serialization": {
                    "allow_pickle": inputs.allow_pickle,
                    "skops_trusted_types": list(self._policy.skops_trusted_types),
                },
            },
            indent=2,
            sort_keys=True,
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
