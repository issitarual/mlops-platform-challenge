"""Resolve alias -> verify hash -> build serving image (cached) -> run."""

from mlship.services.ports import ContainerRuntime, ModelRegistry


class ServingService:
    def __init__(self, registry: ModelRegistry, container_runtime: ContainerRuntime) -> None:
        self._registry = registry
        self._container_runtime = container_runtime

    def serve(self, alias: str, port: int, workers: int) -> None:
        raise NotImplementedError("E4")
