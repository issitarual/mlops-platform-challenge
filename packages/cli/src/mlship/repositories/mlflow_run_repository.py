"""``RunRepository`` backed by MLflow tracking (ADR-008, ADR-019)."""

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlship.domain.errors import TrackingError, TrackingUnavailableError
from mlship.domain.redaction import redact
from mlship.domain.runs import MODEL_VERSION_TAG, RunRecord, RunStatus

_START_HINT = (
    "Start the tracking server with `docker compose up -d` at the repository root, "
    "or point MLSHIP_TRACKING_URI at a running MLflow."
)


@contextmanager
def _fail_fast() -> Iterator[None]:
    """No retries and a short timeout, only for the availability check.

    The MLflow client retries for minutes by default: right for uploads, wrong for
    telling the user that the server is down. MLflow reads these variables on every
    request, so they are set around one call and restored. Process-wide, which is safe
    here: the CLI runs nothing else concurrently at that point.
    """
    overrides = {"MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0", "MLFLOW_HTTP_REQUEST_TIMEOUT": "5"}
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class MlflowRunRepository:
    def __init__(self, tracking_uri: str, artifact_root: str | None = None) -> None:
        self._tracking_uri = tracking_uri
        self._artifact_root = artifact_root  # only for stores without a server (tests)
        self._client: MlflowClient | None = None

    def _mlflow(self) -> MlflowClient:
        if self._client is None:
            self._client = MlflowClient(tracking_uri=self._tracking_uri)
        return self._client

    def check_available(self) -> None:
        try:
            with _fail_fast():
                self._mlflow().search_experiments(max_results=1)
        except Exception as exc:  # connection, auth or missing driver: all mean "not usable"
            raise TrackingUnavailableError(
                f"MLflow is not reachable at {redact(self._tracking_uri)}.",
                hint=_START_HINT,
                details=redact(str(exc))[:2000],
            ) from exc

    def _experiment_id(self, name: str) -> str:
        client = self._mlflow()
        existing = client.get_experiment_by_name(name)
        if existing is not None:
            return str(existing.experiment_id)
        location = f"{self._artifact_root.rstrip('/')}/{name}" if self._artifact_root else None
        try:
            return str(client.create_experiment(name, artifact_location=location))
        except MlflowException:  # created concurrently by another run
            created = client.get_experiment_by_name(name)
            if created is None:
                raise
            return str(created.experiment_id)

    def start(self, experiment: str, tags: Mapping[str, str]) -> str:
        try:
            run = self._mlflow().create_run(self._experiment_id(experiment), tags=dict(tags))
        except MlflowException as exc:
            raise TrackingError("MLflow could not create the run.", details=str(exc)) from exc
        return str(run.info.run_id)

    def log(
        self,
        run_id: str,
        *,
        params: Mapping[str, str],
        metrics: Mapping[str, float],
        tags: Mapping[str, str],
        artifacts: Mapping[str, Path],
    ) -> None:
        client = self._mlflow()
        now = int(datetime.now(tz=UTC).timestamp() * 1000)
        try:
            client.log_batch(
                run_id,
                metrics=[Metric(k, v, now, 0) for k, v in sorted(metrics.items())],
                params=[Param(k, v) for k, v in sorted(params.items())],
                tags=[RunTag(k, v) for k, v in sorted(tags.items())],
            )
            for name, path in artifacts.items():
                if path.is_dir():
                    client.log_artifacts(run_id, str(path), artifact_path=name)
                else:
                    client.log_artifact(run_id, str(path))
        except MlflowException as exc:
            raise TrackingError(
                "MLflow rejected the run data.", hint="See the details below.", details=str(exc)
            ) from exc

    def end(self, run_id: str, status: RunStatus, tags: Mapping[str, str]) -> None:
        client = self._mlflow()
        try:
            for key, value in tags.items():
                client.set_tag(run_id, key, value)
            client.set_terminated(run_id, status.value)
        except MlflowException as exc:
            raise TrackingError("MLflow could not close the run.", details=str(exc)) from exc

    def list_recent(self, experiment: str, limit: int) -> Sequence[RunRecord]:
        client = self._mlflow()
        found = client.get_experiment_by_name(experiment)
        if found is None:
            return []
        runs = client.search_runs(
            [found.experiment_id], order_by=["attributes.start_time DESC"], max_results=limit
        )
        return [
            RunRecord(
                run_id=run.info.run_id,
                created_at=datetime.fromtimestamp(run.info.start_time / 1000, tz=UTC),
                status=RunStatus(run.info.status),
                metrics=dict(run.data.metrics),
                model_version=int(v) if (v := run.data.tags.get(MODEL_VERSION_TAG)) else None,
            )
            for run in runs
        ]
