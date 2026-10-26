"""Resolve alias -> download -> verify -> environment image -> serving image -> run (ADR-020).

The alias is resolved once, here; the running instance only knows its version number.
Trust anchor: the registry version and the run's tags, both in the tracking database.
The artifact store is not trusted: nothing downloaded from it is used before its hash
matches what the database recorded at training.
"""

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from mlship.domain.environment import ImageRef
from mlship.domain.errors import MlshipError, ModelIntegrityError, ModelNotFoundError
from mlship.domain.runs import ModelVersion, tree_digest
from mlship.domain.serving import (
    SINGLE_THREAD_ENVIRONMENT,
    PreparedServing,
    ServingImageSpec,
    ServingJobSpec,
    ServingLimits,
)
from mlship.services.environment_service import EnvironmentService
from mlship.services.ports import (
    ContainerRuntime,
    ContentHasher,
    HttpProbe,
    ModelRegistry,
    RunRepository,
)
from mlship.services.run_recorder import LOCK_ARTIFACT

LOCK_SHA256_TAG = "mlship.lock_sha256"
ENVIRONMENT_KEY_TAG = "mlship.environment_key"
SERIALIZATION_TAG = "mlship.serialization"
DASHBOARD_UID = "mlship-serving"
_PICKLE_SERIALIZATION = "cloudpickle"


@dataclass(frozen=True, slots=True)
class ServingPolicy:
    """What the platform allows when loading a model (same allowlist as training)."""

    skops_trusted_types: tuple[str, ...]
    limits: ServingLimits
    model_max_bytes: int
    model_max_files: int
    cache_dirname: str = ".mlship/serving"
    dashboard_url: str | None = None


class ServingService:
    def __init__(
        self,
        run_repository: RunRepository,
        registry: ModelRegistry,
        environment: EnvironmentService,
        container_runtime: ContainerRuntime,
        hasher: ContentHasher,
        policy: ServingPolicy,
        probe: HttpProbe | None = None,
    ) -> None:
        self._probe = probe
        self._runs = run_repository
        self._registry = registry
        self._environment = environment
        self._containers = container_runtime
        self._hasher = hasher
        self._policy = policy

    def dashboard(self) -> str | None:
        """The Grafana dashboard, when the monitoring stack is up (ADR-016)."""
        base = self._policy.dashboard_url
        if not base or self._probe is None or not self._probe.answers(f"{base}/api/health"):
            return None
        return f"{base}/d/{DASHBOARD_UID}"

    def prepare(self, project_dir: Path, model_name: str, alias: str) -> PreparedServing:
        self._runs.check_available()
        model = self._registry.get_alias(model_name, alias)
        if model is None:
            raise ModelNotFoundError(
                f"No version of {model_name} has the alias {alias!r}.",
                hint=f"Promote one first: mlship promote <version> --alias {alias}",
            )
        workdir = project_dir / self._policy.cache_dirname / model.name / f"v{model.version}"
        if workdir.exists():
            shutil.rmtree(workdir)  # always verify a fresh download, never a cached copy
        model_dir = self._registry.download(model, workdir)
        self._verify(model, model_dir)

        tags = self._runs.get_tags(model.run_id)
        environment = self._training_environment(model, tags, workdir)
        image = self._containers.ensure_serving_image(
            ServingImageSpec(
                base=environment[0],
                model=model,
                allow_pickle=tags.get(SERIALIZATION_TAG) == _PICKLE_SERIALIZATION,
                skops_trusted_types=self._policy.skops_trusted_types,
            ),
            model_dir,
        )
        return PreparedServing(model, alias, environment[0], environment[1], image)

    def run(
        self,
        prepared: PreparedServing,
        *,
        port: int,
        workers: int,
        environment: Mapping[str, str],
        on_log: Callable[[str], None],
        on_ready: Callable[[str], None],
    ) -> None:
        self._containers.run_serving(
            ServingJobSpec(
                image=prepared.serving_image,
                model=prepared.model,
                host_port=port,
                workers=workers,
                environment={**SINGLE_THREAD_ENVIRONMENT, **environment},
                limits=self._policy.limits,
            ),
            on_log,
            on_ready,
        )

    def _verify(self, model: ModelVersion, model_dir: Path) -> None:
        files = self._hasher.untrusted_tree(
            model_dir, self._policy.model_max_bytes, self._policy.model_max_files
        )
        actual = tree_digest(files)
        if not model.model_sha256 or actual != model.model_sha256:
            raise ModelIntegrityError(
                f"{model.name} version {model.version} does not match its registered hash; "
                "it was not loaded.",
                hint="The artifact store was changed after registration. Train and register "
                "the model again, and check who has write access to the artifact store.",
                details=f"registered {model.model_sha256[:12] or '(none)'}…, "
                f"downloaded {actual[:12]}…",
            )

    def _training_environment(
        self, model: ModelVersion, tags: Mapping[str, str], workdir: Path
    ) -> tuple[ImageRef, bool]:
        """The image the model was trained in: reused, or rebuilt only if identical."""
        lock_sha256 = tags.get(LOCK_SHA256_TAG, "")
        key = tags.get(ENVIRONMENT_KEY_TAG, "")
        if not lock_sha256 or not key:
            raise MlshipError(
                "The run that produced this model does not record its environment.",
                hint="Train and register the model again with this version of mlship.",
            )
        lock_file = self._runs.download_artifact(model.run_id, LOCK_ARTIFACT, workdir)
        if lock_file is None:
            raise MlshipError(
                "The run that produced this model did not record its lock file, so its "
                "environment cannot be verified.",
                hint="Train and register the model again with this version of mlship.",
            )
        prepared = self._environment.ensure_image_for_lock(lock_file, lock_sha256, key)
        return prepared.image, prepared.cache_hit
