"""The tracking side of a training run: open, record, register, close (ADR-019)."""

import contextlib
from collections.abc import Callable
from dataclasses import replace

from mlship.domain.errors import MlshipError
from mlship.domain.runs import MODEL_VERSION_TAG, Manifest, ModelVersion, RunStatus, ToolInfo
from mlship.domain.training import TrainingOutcome
from mlship.services.lineage import manifest_tags
from mlship.services.ports import ModelRegistry, RunRepository

ERROR_TAG = "mlship.error"
LOCK_ARTIFACT = "mlship.lock"


class RunRecorder:
    def __init__(
        self,
        run_repository: RunRepository,
        model_registry: ModelRegistry,
        actor: Callable[[], str] = lambda: "unknown",
    ) -> None:
        self._runs = run_repository
        self._registry = model_registry
        self._actor = actor

    def check_available(self) -> None:
        self._runs.check_available()

    def open(self, model_name: str, tool: ToolInfo) -> str:
        return self._runs.start(
            model_name,
            tags={
                "mlflow.user": self._actor(),
                "mlship.model_name": model_name,
                "mlship.tool_version": tool.version,
                "mlship.tool_git_sha": tool.git_sha,
            },
        )

    def record(self, outcome: TrainingOutcome, manifest: Manifest) -> TrainingOutcome:
        """Log, upload, register a version on success, then close the run."""
        run_dir = outcome.run_dir
        artifacts = {
            "manifest.json": run_dir / "manifest.json",
            "run.json": run_dir / "run.json",
            LOCK_ARTIFACT: outcome.environment.lock.path,
        }
        result_file = run_dir / "output" / "result.json"
        if result_file.is_file() and not result_file.is_symlink():
            artifacts["result.json"] = result_file
        if outcome.succeeded:
            artifacts["model"] = run_dir / "output" / "model"
        self._runs.log(
            outcome.run_id,
            params=outcome.params,
            metrics=outcome.metrics,
            tags=manifest_tags(manifest),
            artifacts=artifacts,
        )
        version: ModelVersion | None = None
        end_tags: dict[str, str] = {}
        if outcome.succeeded and manifest.model_sha256 is not None:
            version = self._registry.register(
                outcome.model_name, outcome.run_id, manifest.model_sha256
            )
            end_tags[MODEL_VERSION_TAG] = str(version.version)
        status = RunStatus.FINISHED if outcome.succeeded else RunStatus.FAILED
        self._runs.end(outcome.run_id, status, end_tags)
        return replace(outcome, model_version=version)

    def close(self, run_id: str, exc: BaseException) -> None:
        """Best effort: never hide the original error behind a tracking error."""
        status = RunStatus.KILLED if isinstance(exc, KeyboardInterrupt) else RunStatus.FAILED
        reason = exc.message if isinstance(exc, MlshipError) else type(exc).__name__
        with contextlib.suppress(Exception):
            self._runs.end(run_id, status, {ERROR_TAG: reason[:500]})
