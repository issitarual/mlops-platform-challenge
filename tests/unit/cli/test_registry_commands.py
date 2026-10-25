"""``mlship runs`` and ``mlship promote`` end to end through Typer, with in-memory tracking."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeModelRegistry, FakeRunRepository
from typer.testing import CliRunner

from mlship.controllers.app import app
from mlship.domain.runs import RunStatus
from mlship.services.registry_service import RegistryService

runner = CliRunner()


@pytest.fixture
def tracking(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeRunRepository, FakeModelRegistry]:
    runs, registry = FakeRunRepository(), FakeModelRegistry()
    services = SimpleNamespace(registry=RegistryService(runs, registry, actor=lambda: "laura"))
    for module in ("runs", "promote"):
        monkeypatch.setattr(f"mlship.controllers.{module}.load_services", lambda: services)
    return runs, registry


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "california-housing"
    project.mkdir()
    return project


def test_runs_lists_the_project_model(
    tmp_path: Path, tracking: tuple[FakeRunRepository, FakeModelRegistry]
) -> None:
    runs, _ = tracking
    run_id = runs.start("california-housing", {})
    runs.log(run_id, params={}, metrics={"rmse": 0.47}, tags={}, artifacts={})
    runs.end(run_id, RunStatus.FINISHED, {})
    runs.start("another-model", {})

    result = runner.invoke(app, ["runs", str(_project(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "california-housing" in result.output and "0.47" in result.output
    assert run_id[:12] in result.output


def test_runs_without_history_says_how_to_start(
    tmp_path: Path, tracking: tuple[FakeRunRepository, FakeModelRegistry]
) -> None:
    result = runner.invoke(app, ["runs", str(_project(tmp_path))])
    assert result.exit_code == 0 and "mlship train" in result.output


def test_promote_moves_the_alias_and_shows_what_it_replaced(
    tmp_path: Path, tracking: tuple[FakeRunRepository, FakeModelRegistry]
) -> None:
    _, registry = tracking
    registry.register("california-housing", "run-a", "aaa")
    registry.register("california-housing", "run-b", "bbb")
    project = str(_project(tmp_path))
    runner.invoke(app, ["promote", "1", "--project", project])
    result = runner.invoke(app, ["promote", "2", "--project", project])
    assert result.exit_code == 0, result.output
    assert "v1" in result.output and "v2" in result.output
    assert registry.aliases[("california-housing", "champion")] == 2


def test_promote_of_a_missing_version_fails_clearly(
    tmp_path: Path, tracking: tuple[FakeRunRepository, FakeModelRegistry]
) -> None:
    result = runner.invoke(app, ["promote", "3", "--project", str(_project(tmp_path))])
    assert result.exit_code == 1
    assert "no version 3" in result.output


def test_promote_refuses_latest(
    tmp_path: Path, tracking: tuple[FakeRunRepository, FakeModelRegistry]
) -> None:
    _, registry = tracking
    registry.register("california-housing", "run-a", "aaa")
    result = runner.invoke(
        app, ["promote", "1", "--alias", "latest", "--project", str(_project(tmp_path))]
    )
    assert result.exit_code == 1 and "reserved" in result.output
