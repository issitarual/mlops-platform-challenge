"""Composition root: the only module that instantiates adapters and repositories.

Controllers receive ready services from here; services only know the ports.
"""

from dataclasses import dataclass

from mlship.adapters.docker_runtime import DockerRuntime
from mlship.adapters.uv_lock_resolver import UvLockResolver
from mlship.config import Settings
from mlship.repositories.mlflow_model_registry import MlflowModelRegistry
from mlship.repositories.mlflow_run_repository import MlflowRunRepository
from mlship.services.environment_service import EnvironmentService
from mlship.services.registry_service import RegistryService
from mlship.services.serving_service import ServingService
from mlship.services.training_service import TrainingService
from mlship.services.validation.validator import ProjectValidator


@dataclass(frozen=True, slots=True)
class Services:
    validator: ProjectValidator
    training: TrainingService
    registry: RegistryService
    serving: ServingService


def build_services(settings: Settings) -> Services:
    container_runtime = DockerRuntime()
    run_repository = MlflowRunRepository(settings.tracking_uri, settings.experiment_name)
    model_registry = MlflowModelRegistry(settings.tracking_uri, settings.model_name)
    validator = ProjectValidator()
    environment = EnvironmentService(
        UvLockResolver(settings.python_version, settings.platform), container_runtime
    )
    return Services(
        validator=validator,
        training=TrainingService(validator, environment, container_runtime, run_repository),
        registry=RegistryService(run_repository, model_registry),
        serving=ServingService(model_registry, container_runtime),
    )
