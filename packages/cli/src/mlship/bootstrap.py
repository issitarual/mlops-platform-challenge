"""Composition root: the only module that instantiates adapters and repositories.

Controllers receive ready services from here; services only know the ports.
"""

from dataclasses import dataclass

from mlship.adapters.docker_runtime import DockerRuntime
from mlship.adapters.dockerfile import EnvironmentDockerfile
from mlship.adapters.runtime_packager import WorkspaceRuntimePackager
from mlship.adapters.uv_lock_resolver import UvLockResolver
from mlship.config import Settings
from mlship.domain.training import ResourceLimits
from mlship.repositories.mlflow_model_registry import MlflowModelRegistry
from mlship.repositories.mlflow_run_repository import MlflowRunRepository
from mlship.services.environment_service import EnvironmentService
from mlship.services.registry_service import RegistryService
from mlship.services.serving_service import ServingService
from mlship.services.training_service import TrainingPolicy, TrainingService
from mlship.services.validation.validator import ProjectValidator


@dataclass(frozen=True, slots=True)
class Services:
    validator: ProjectValidator
    environment: EnvironmentService
    training: TrainingService
    registry: RegistryService
    serving: ServingService


def build_services(settings: Settings) -> Services:
    container_runtime = DockerRuntime(settings.platform)
    run_repository = MlflowRunRepository(settings.tracking_uri, settings.experiment_name)
    model_registry = MlflowModelRegistry(settings.tracking_uri, settings.model_name)
    validator = ProjectValidator()
    environment = EnvironmentService(
        UvLockResolver(settings.python_version, settings.python_platform),
        WorkspaceRuntimePackager(),
        EnvironmentDockerfile(settings.base_image),
        container_runtime,
        python_version=settings.python_version,
        python_platform=settings.python_platform,
        docker_platform=settings.platform,
        lock_filename=settings.lock_filename,
        image_repository=settings.image_repository,
    )
    return Services(
        validator=validator,
        environment=environment,
        training=TrainingService(
            validator,
            environment,
            container_runtime,
            run_repository,
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
            ),
        ),
        registry=RegistryService(run_repository, model_registry),
        serving=ServingService(model_registry, container_runtime),
    )
