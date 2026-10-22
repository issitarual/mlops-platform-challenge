"""``RunRepository`` backed by MLflow tracking."""

from collections.abc import Sequence
from pathlib import Path

from mlship.domain.runs import Manifest, RunRecord


class MlflowRunRepository:
    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name

    def save(self, manifest: Manifest, output_dir: Path) -> RunRecord:
        raise NotImplementedError("E3")

    def list_recent(self, limit: int) -> Sequence[RunRecord]:
        raise NotImplementedError("E3")
