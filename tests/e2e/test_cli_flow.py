"""The scientist's path through the real CLI: init -> validate -> train -> runs ->
promote -> serve -> predict -> stop. Each step is the installed ``mlship`` command,
so exit codes, output and the commands working together are what is tested.

Needs Docker and network (lock and image build). Run with:
    uv run pytest -m e2e tests/e2e -v
"""

import csv
import itertools
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

import docker
import pytest
from tracking_cleanup import remove_models

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(shutil.which("docker") is None, reason="Docker not installed"),
]

MLSHIP = shutil.which("mlship", path=str(Path(sys.executable).parent))
STEP_TIMEOUT = 900
READY_TIMEOUT = 300


def _environment(tmp_path: Path, model_name: str) -> dict[str, str]:
    """CI points MLSHIP_TRACKING_URI at the compose server; otherwise a local SQLite store."""
    env = {**os.environ, "MLSHIP_MODEL_NAME": model_name, "NO_COLOR": "1", "COLUMNS": "200"}
    if "MLSHIP_TRACKING_URI" not in os.environ:
        env["MLSHIP_TRACKING_URI"] = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
        env["MLSHIP_ARTIFACT_ROOT"] = (tmp_path / "artifacts").as_uri()
    return env


def _mlship(env: dict[str, str], *args: str, expect: int = 0) -> str:
    assert MLSHIP is not None, "the mlship command is not installed in this environment"
    result = subprocess.run(  # noqa: S603 - fixed executable, arguments from the test
        [MLSHIP, *args],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=STEP_TIMEOUT,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == expect, f"mlship {' '.join(args)}:\n{output}"
    return output


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _get(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - localhost
            return int(response.status)
    except (urllib.error.URLError, OSError):
        return 0


def _post(url: str, payload: object) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(  # noqa: S310 - localhost
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _start_serving(
    env: dict[str, str], port: int, project: Path, log: IO[str]
) -> subprocess.Popen[str]:
    assert MLSHIP is not None
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP  # so Ctrl+Break reaches only serve
    else:
        flags = 0
    return subprocess.Popen(  # noqa: S603 - fixed executable, arguments from the test
        [MLSHIP, "serve", "--project", str(project), "--port", str(port)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=flags,
    )


def _stop(server: subprocess.Popen[str]) -> int:
    """The way a supervisor stops it: SIGTERM, or Ctrl+Break on Windows."""
    if sys.platform == "win32":
        server.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        server.send_signal(signal.SIGTERM)
    return server.wait(timeout=60)


def _tail(path: Path, lines: int = 40) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _containers_of(model_name: str) -> list[Any]:
    client = docker.from_env()
    label = f"org.mlship.model-name={model_name}"
    return list(client.containers.list(all=True, filters={"label": label}))


@pytest.fixture
def name() -> Iterator[str]:
    model_name = f"e2e-{uuid.uuid4().hex[:8]}"
    yield model_name
    if "MLSHIP_TRACKING_URI" in os.environ:
        remove_models(os.environ["MLSHIP_TRACKING_URI"], [model_name])


def test_a_new_project_goes_from_init_to_predictions(tmp_path: Path, name: str) -> None:
    env = _environment(tmp_path, name)
    project = tmp_path / "project"

    output = _mlship(env, "init", str(project))
    assert "Wrote mlship.lock" in output
    _mlship(env, "validate", str(project), "--data", str(project / "data"), "--strict")

    output = _mlship(env, "train", str(project), "--data", str(project / "data"))
    match = re.search(rf"Registered {name} version (\d+)", output)
    assert match, output
    version = match.group(1)
    assert name in _mlship(env, "runs", str(project))
    _mlship(env, "promote", version, "--project", str(project))

    port = _free_port()
    log_path = tmp_path / "serve.log"
    log = log_path.open("w", encoding="utf-8")
    server = _start_serving(env, port, project, log)
    try:
        deadline = time.monotonic() + READY_TIMEOUT
        while _get(f"http://127.0.0.1:{port}/health/ready") != 200:
            assert server.poll() is None, f"serve exited early:\n{_tail(log_path)}"
            assert time.monotonic() < deadline, f"never ready:\n{_tail(log_path)}"
            time.sleep(1)

        with (project / "data" / "example.csv").open(newline="") as handle:
            rows = [
                {k: float(v) for k, v in row.items() if k != "target"}
                for row in itertools.islice(csv.DictReader(handle), 3)
            ]
        status, body = _post(f"http://127.0.0.1:{port}/v1/predict", {"rows": rows})
        assert status == 200, body
        assert body["model"] == {"name": name, "version": version}
        assert len(body["predictions"]) == 3

        status, body = _post(f"http://127.0.0.1:{port}/v1/predict", {"rows": [{"x": "y"}]})
        assert status == 422 and len(body["details"]) >= 2
    finally:
        exit_code = _stop(server)
        log.close()
    assert exit_code == 0, f"stopping serve is a normal exit:\n{_tail(log_path)}"
    assert _containers_of(name) == [], "the API container is removed when serve stops"
