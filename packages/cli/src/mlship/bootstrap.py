"""Composition root: the only module that instantiates adapters and repositories.

Controllers receive ready services from here; services only know the ports.
"""

import getpass
from dataclasses import dataclass

from mlship.adapters.docker_runtime import DockerRuntime
from mlship.adapters.dockerfile import EnvironmentDockerfile
from mlship.adapters.hashing import Sha256Hasher
from mlship.adapters.http_probe import UrllibProbe
from mlship.adapters.project_template import PackagedProjectTemplate
from mlship.adapters.runtime_packager import WorkspaceRuntimePackager
from mlship.adapters.tool_info import GitToolInfo
from mlship.adapters.uv_lock_resolver import UvLockResolver
from mlship.config import Settings
from mlship.domain.serving import ServingLimits
from mlship.domain.training import ResourceLimits
from mlship.repositories.mlflow_model_registry import MlflowModelRegistry
from mlship.repositories.mlflow_run_repository import MlflowRunRepository
from mlship.services.environment_service import EnvironmentService
from mlship.services.registry_service import RegistryService
from mlship.services.run_recorder import RunRecorder
from mlship.services.scaffold_service import ScaffoldService
from mlship.services.serving_service import ServingPolicy, ServingService
from mlship.services.training_service import TrainingPolicy, TrainingService
from mlship.services.validation.validator import ProjectValidator
from mlship.services.validation_service import ValidationService


@dataclass(frozen=True, slots=True)
class Services:
    validation: ValidationService
    environment: EnvironmentService
    training: TrainingService
    registry: RegistryService
    serving: ServingService
    scaffold: ScaffoldService


def build_services(settings: Settings) -> Services:
    container_runtime = DockerRuntime(settings.platform)
    run_repository = MlflowRunRepository(settings.tracking_uri, settings.artifact_root)
    model_registry = MlflowModelRegistry(settings.tracking_uri)
    runtime_packager = WorkspaceRuntimePackager()
    validator = ProjectValidator(runtime_packager.requirements())
    environment = EnvironmentService(
        UvLockResolver(settings.python_version, settings.python_platform),
        runtime_packager,
        EnvironmentDockerfile(settings.base_image),
        container_runtime,
        python_version=settings.python_version,
        python_platform=settings.python_platform,
        docker_platform=settings.platform,
        lock_filename=settings.lock_filename,
        image_repository=settings.image_repository,
    )
    return Services(
        validation=ValidationService(validator, environment),
        environment=environment,
        training=TrainingService(
            validator,
            environment,
            container_runtime,
            RunRecorder(run_repository, model_registry, actor=_current_user),
            Sha256Hasher(),
            GitToolInfo(),
            TrainingPolicy(
                limits=ResourceLimits(
                    cpus=settings.train_cpu_limit,
                    memory=settings.train_memory_limit,
                    pids=settings.train_pids_limit,
                    timeout_seconds=settings.train_timeout_seconds,
                    tmpfs_size=settings.train_tmpfs_size,
                ),
                skops_trusted_types=tuple(settings.skops_trusted_types),
                allow_pickle_enabled=settings.allow_pickle_enabled,
                model_max_bytes=settings.model_max_bytes,
                model_max_files=settings.model_max_files,
            ),
        ),
        registry=RegistryService(run_repository, model_registry, actor=_current_user),
        scaffold=ScaffoldService(PackagedProjectTemplate()),
        serving=ServingService(
            run_repository,
            model_registry,
            environment,
            container_runtime,
            Sha256Hasher(),
            ServingPolicy(
                skops_trusted_types=tuple(settings.skops_trusted_types),
                limits=ServingLimits(
                    cpus=settings.serve_cpu_limit,
                    memory=settings.serve_memory_limit,
                    pids=settings.serve_pids_limit,
                ),
                model_max_bytes=settings.model_max_bytes,
                model_max_files=settings.model_max_files,
                dashboard_url=settings.dashboard_url,
            ),
            UrllibProbe(),
        ),
    )


def _current_user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):  # no user name in the environment or the passwd database
        return "unknown"
