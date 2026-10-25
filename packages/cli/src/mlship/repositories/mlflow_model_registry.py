"""``ModelRegistry`` backed by the MLflow Model Registry (ADR-015, ADR-019).

Versions are immutable and consumed through explicit aliases, never "latest".
The SHA-256 the host computed for the model files is stored as a version tag and
checked again before any load (E4).
"""

from collections.abc import Mapping
from pathlib import Path

from mlflow.entities.model_registry import ModelVersion as MlflowModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlship.domain.errors import ModelNotFoundError, TrackingError
from mlship.domain.runs import ModelVersion

MODEL_ARTIFACT = "model"
SHA256_TAG = "mlship.model_sha256"
_NOT_FOUND = "RESOURCE_DOES_NOT_EXIST"
_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
_READY = "READY"


def _is_not_found(exc: MlflowException) -> bool:
    return str(exc.error_code) == _NOT_FOUND


def _is_missing_alias(exc: MlflowException) -> bool:
    """MLflow 3.17 reports an unknown alias as INVALID_PARAMETER_VALUE, not as missing."""
    message = str(exc).lower()
    return _is_not_found(exc) or (
        str(exc.error_code) == "INVALID_PARAMETER_VALUE"
        and "alias" in message
        and "not found" in message
    )


class MlflowModelRegistry:
    def __init__(self, tracking_uri: str) -> None:
        self._tracking_uri = tracking_uri
        self._client: MlflowClient | None = None

    def _mlflow(self) -> MlflowClient:
        if self._client is None:
            self._client = MlflowClient(
                tracking_uri=self._tracking_uri, registry_uri=self._tracking_uri
            )
        return self._client

    def _to_domain(self, found: MlflowModelVersion) -> ModelVersion:
        return ModelVersion(
            name=found.name,
            version=int(found.version),
            run_id=found.run_id or "",
            model_sha256=(found.tags or {}).get(SHA256_TAG, ""),
            aliases=tuple(sorted(found.aliases or ())),
        )

    def register(self, name: str, run_id: str, model_sha256: str) -> ModelVersion:
        client = self._mlflow()
        try:
            self._ensure_registered_model(name)
            created = client.create_model_version(
                name,
                source=f"runs:/{run_id}/{MODEL_ARTIFACT}",
                run_id=run_id,
                tags={SHA256_TAG: model_sha256},
            )
        except MlflowException as exc:
            raise TrackingError(
                f"Registering a new version of {name} failed.", details=str(exc)
            ) from exc
        if created.status != _READY:  # the client already waited for asynchronous stores
            raise TrackingError(
                f"Version {created.version} of {name} ended as {created.status}.",
                details=created.status_message or "",
            )
        return self._to_domain(created)

    def _ensure_registered_model(self, name: str) -> None:
        """Create the registered model once; a concurrent creation is not an error."""
        client = self._mlflow()
        try:
            client.get_registered_model(name)
            return
        except MlflowException as exc:
            if not _is_not_found(exc):
                raise
        try:
            client.create_registered_model(name)
        except MlflowException as exc:
            if str(exc.error_code) != _ALREADY_EXISTS:
                raise

    def get_version(self, name: str, version: int) -> ModelVersion:
        try:
            return self._to_domain(self._mlflow().get_model_version(name, str(version)))
        except MlflowException as exc:
            if _is_not_found(exc):
                raise ModelNotFoundError(
                    f"{name} has no version {version}.",
                    hint=f"List the versions with `mlship runs` (model {name}).",
                ) from exc
            raise TrackingError("Reading the model registry failed.", details=str(exc)) from exc

    def get_alias(self, name: str, alias: str) -> ModelVersion | None:
        try:
            return self._to_domain(self._mlflow().get_model_version_by_alias(name, alias))
        except MlflowException as exc:
            if _is_missing_alias(exc):
                return None
            raise TrackingError("Reading the model registry failed.", details=str(exc)) from exc

    def set_alias(self, name: str, alias: str, version: int, tags: Mapping[str, str]) -> None:
        client = self._mlflow()
        try:
            client.set_registered_model_alias(name, alias, str(version))
            for key, value in tags.items():
                client.set_model_version_tag(name, str(version), key, value)
        except MlflowException as exc:
            raise TrackingError(f"Moving the alias {alias} failed.", details=str(exc)) from exc

    def download(self, model: ModelVersion, target_dir: Path) -> Path:
        """The files of the version's run; the caller verifies them before any use."""
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            local = self._mlflow().download_artifacts(model.run_id, MODEL_ARTIFACT, str(target_dir))
        except MlflowException as exc:
            raise TrackingError(
                f"Downloading {model.name} version {model.version} failed.", details=str(exc)
            ) from exc
        return Path(local)
