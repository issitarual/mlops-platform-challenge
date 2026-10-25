"""``mlship serve`` through Typer, with the host-side service on in-memory fakes."""

import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_serving_service import World
from typer.testing import CliRunner

from mlship.controllers.app import app
from mlship.domain.serving import PreparedServing
from mlship.services.serving_service import ServingService

runner = CliRunner()


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    world = World(tmp_path)
    services = SimpleNamespace(serving=world.serving())
    monkeypatch.setattr("mlship.controllers.serve.load_services", lambda: services)
    return world


def test_serve_prints_what_it_verified_and_where_it_listens(world: World) -> None:
    result = runner.invoke(
        app,
        ["serve", "--project", str(world.project), "--model-name", "demo", "--port", "9001"],
    )
    assert result.exit_code == 0, result.output
    assert "demo v1" in result.output and "verified" in result.output
    assert "http://127.0.0.1:9001/v1/predict" in result.output


def test_api_key_is_forwarded_only_when_set(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLSHIP_SERVING_API_KEY", "k")
    runner.invoke(app, ["serve", "--project", str(world.project), "--model-name", "demo"])
    (job,) = world.runtime.serving_jobs
    assert job.environment == {"MLSHIP_SERVING_API_KEY": "k"}


def test_serving_an_alias_nobody_has_fails_with_a_hint(world: World) -> None:
    result = runner.invoke(
        app,
        ["serve", "--alias", "challenger", "--project", str(world.project), "--model-name", "demo"],
    )
    assert result.exit_code == 1 and "mlship promote" in result.output


class _TerminatedWhileServing:
    """Prepares like the real service, then receives SIGTERM while the API runs."""

    def __init__(self, inner: ServingService) -> None:
        self._inner = inner

    def prepare(self, project_dir: Path, model_name: str, alias: str) -> PreparedServing:
        return self._inner.prepare(project_dir, model_name, alias)

    def run(self, *args: object, **kwargs: object) -> None:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill(SIGTERM) ends the process on Windows")
def test_sigterm_stops_like_ctrl_c_and_restores_the_handler(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = signal.getsignal(signal.SIGTERM)
    services = SimpleNamespace(serving=_TerminatedWhileServing(world.serving()))
    monkeypatch.setattr("mlship.controllers.serve.load_services", lambda: services)
    result = runner.invoke(app, ["serve", "--project", str(world.project), "--model-name", "demo"])
    assert result.exit_code == 0, result.output
    assert "Stopped" in result.output
    assert signal.getsignal(signal.SIGTERM) is before
