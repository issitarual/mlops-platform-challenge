"""Run listing and promotion by explicit alias (never "latest")."""

from collections.abc import Sequence

from mlship.domain.runs import ModelVersion, RunRecord
from mlship.services.ports import ModelRegistry, RunRepository


class RegistryService:
    def __init__(self, run_repository: RunRepository, registry: ModelRegistry) -> None:
        self._run_repository = run_repository
        self._registry = registry

    def list_runs(self, limit: int) -> Sequence[RunRecord]:
        raise NotImplementedError("E3")

    def promote(self, version: int, alias: str) -> ModelVersion:
        raise NotImplementedError("E3")
