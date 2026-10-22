"""``ModelRegistry`` backed by the MLflow Model Registry."""

from pathlib import Path

from mlship.domain.runs import ModelVersion


class MlflowModelRegistry:
    def __init__(self, tracking_uri: str, model_name: str) -> None:
        self._tracking_uri = tracking_uri
        self._model_name = model_name

    def register(self, run_id: str) -> ModelVersion:
        raise NotImplementedError("E3")

    def set_alias(self, version: int, alias: str) -> None:
        raise NotImplementedError("E3")

    def resolve_alias(self, alias: str) -> ModelVersion:
        raise NotImplementedError("E3")

    def download(self, model: ModelVersion, target_dir: Path) -> Path:
        raise NotImplementedError("E4")
