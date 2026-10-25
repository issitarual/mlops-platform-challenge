"""The MLflow repositories against a real MLflow store (SQLite, no server, no network)."""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from mlship.domain.errors import ModelNotFoundError, TrackingUnavailableError
from mlship.domain.runs import RunStatus
from mlship.repositories.mlflow_model_registry import MlflowModelRegistry
from mlship.repositories.mlflow_run_repository import MlflowRunRepository

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # tests never phone home


@pytest.fixture(scope="session")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, str | None]]:
    """One database for the whole session: creating MLflow's schema takes seconds.

    With MLSHIP_TEST_TRACKING_URI set (e.g. a local ``mlflow server``), the same tests
    run over HTTP, the path the CLI uses by default; the server then stores artifacts.
    """
    server = os.environ.get("MLSHIP_TEST_TRACKING_URI")
    if server:
        yield server, None
        return
    root = tmp_path_factory.mktemp("mlflow")
    yield f"sqlite:///{(root / 'mlflow.db').as_posix()}", (root / "artifacts").as_uri()


@pytest.fixture
def runs(store: tuple[str, str | None]) -> MlflowRunRepository:
    return MlflowRunRepository(store[0], artifact_root=store[1])


@pytest.fixture
def registry(store: tuple[str, str | None]) -> MlflowModelRegistry:
    return MlflowModelRegistry(store[0])


@pytest.fixture
def name() -> str:
    return f"model-{uuid.uuid4().hex[:8]}"


def _model_dir(root: Path) -> Path:
    model = root / "model"
    (model / "nested").mkdir(parents=True)
    (model / "MLmodel").write_text("flavors: {}\n")
    (model / "nested" / "weights.bin").write_bytes(b"\x00\x01")
    return model


def _tracked_run(runs: MlflowRunRepository, name: str, tmp_path: Path) -> str:
    run_id = runs.start(name, {"mlship.model_name": name})
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    runs.log(
        run_id,
        params={"seed": "42"},
        metrics={"rmse": 0.5},
        tags={"mlship.code_sha256": "c0de"},
        artifacts={"model": _model_dir(tmp_path), "manifest.json": manifest},
    )
    return run_id


def test_available_store_passes_the_check(runs: MlflowRunRepository) -> None:
    runs.check_available()


def test_unreachable_server_fails_fast_with_a_hint() -> None:
    repository = MlflowRunRepository("http://127.0.0.1:9")  # nothing listens on port 9
    with pytest.raises(TrackingUnavailableError) as info:
        repository.check_available()
    assert "docker compose up" in info.value.hint


def test_run_keeps_params_metrics_tags_artifacts_and_status(
    runs: MlflowRunRepository, name: str, tmp_path: Path
) -> None:
    run_id = _tracked_run(runs, name, tmp_path)
    runs.end(run_id, RunStatus.FINISHED, {"mlship.model_version": "1"})

    run = runs._mlflow().get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params == {"seed": "42"} and run.data.metrics == {"rmse": 0.5}
    assert run.data.tags["mlship.code_sha256"] == "c0de"
    listed = {a.path for a in runs._mlflow().list_artifacts(run_id)}
    assert listed == {"model", "manifest.json"}
    nested = {a.path for a in runs._mlflow().list_artifacts(run_id, "model/nested")}
    assert nested == {"model/nested/weights.bin"}


def test_each_model_name_is_its_own_experiment_newest_first(
    runs: MlflowRunRepository, name: str, tmp_path: Path
) -> None:
    first = runs.start(name, {})
    runs.end(first, RunStatus.FAILED, {})
    second = _tracked_run(runs, name, tmp_path)
    runs.end(second, RunStatus.FINISHED, {"mlship.model_version": "3"})
    runs.start(f"{name}-other", {})

    records = runs.list_recent(name, limit=10)
    assert [r.run_id for r in records] == [second, first]
    assert records[0].model_version == 3 and records[0].metrics == {"rmse": 0.5}
    assert records[1].status is RunStatus.FAILED and records[1].model_version is None


def test_unknown_model_has_no_runs(runs: MlflowRunRepository) -> None:
    assert runs.list_recent("never-trained", limit=5) == []


def test_versions_are_sequential_and_carry_the_model_hash(
    runs: MlflowRunRepository, registry: MlflowModelRegistry, name: str, tmp_path: Path
) -> None:
    run_a = _tracked_run(runs, name, tmp_path / "a")
    run_b = _tracked_run(runs, name, tmp_path / "b")
    v1 = registry.register(name, run_a, "aaa")
    v2 = registry.register(name, run_b, "bbb")
    assert (v1.version, v2.version) == (1, 2)
    assert registry.get_version(name, 2).model_sha256 == "bbb"
    assert registry.get_version(name, 1).run_id == run_a


def test_alias_points_at_one_version_and_moves(
    runs: MlflowRunRepository, registry: MlflowModelRegistry, name: str, tmp_path: Path
) -> None:
    for folder in ("a", "b"):
        registry.register(name, _tracked_run(runs, name, tmp_path / folder), folder)
    assert registry.get_alias(name, "champion") is None
    registry.set_alias(name, "champion", 1, {"mlship.champion.promoted_by": "laura"})
    registry.set_alias(name, "champion", 2, {})
    champion = registry.get_alias(name, "champion")
    assert champion is not None and champion.version == 2 and champion.aliases == ("champion",)
    tags = registry._mlflow().get_model_version(name, "1").tags
    assert tags["mlship.champion.promoted_by"] == "laura"


def test_missing_version_or_model_is_not_found(
    runs: MlflowRunRepository, registry: MlflowModelRegistry, name: str, tmp_path: Path
) -> None:
    registry.register(name, _tracked_run(runs, name, tmp_path), "aaa")
    with pytest.raises(ModelNotFoundError):
        registry.get_version(name, 9)
    with pytest.raises(ModelNotFoundError):
        registry.get_version("no-such-model", 1)
    assert registry.get_alias("no-such-model", "champion") is None
