"""Validate -> prepare environment -> run isolated job -> verify -> register."""

from collections.abc import Mapping
from pathlib import Path

from mlship.domain.runs import RunRecord
from mlship.services.environment_service import EnvironmentService
from mlship.services.ports import ContainerRuntime, RunRepository
from mlship.services.validation.validator import ProjectValidator


class TrainingService:
    def __init__(
        self,
        validator: ProjectValidator,
        environment: EnvironmentService,
        container_runtime: ContainerRuntime,
        run_repository: RunRepository,
    ) -> None:
        self._validator = validator
        self._environment = environment
        self._container_runtime = container_runtime
        self._run_repository = run_repository

    def train(self, project_dir: Path, data_dir: Path, params: Mapping[str, str]) -> RunRecord:
        raise NotImplementedError("E2/E3")
