"""Real training in the locked-down container (needs Docker and network for the build).

Run with: uv run pytest -m integration tests/integration -v
"""

import json
import os
import shutil
import socket
import threading
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import docker
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from tracking_cleanup import remove_models

from mlship.bootstrap import build_services
from mlship.config import Settings
from mlship.domain.errors import ContainerRunError
from mlship.domain.training import TrainingOutcome

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "california-housing"
COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("docker") is None, reason="Docker not installed"),
]

ISOLATION_PROBE = """
import os
import socket
from pathlib import Path

from sklearn.dummy import DummyRegressor
import pandas as pd

from mlship_sdk import TrainOutput


def blocked(action) -> float:
    try:
        action()
    except OSError:
        return 1.0
    return 0.0


def train(data_dir: Path, params: dict) -> TrainOutput:
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    model = DummyRegressor().fit(x, [1.0, 2.0, 3.0])
    return TrainOutput(
        model=model,
        metrics={
            "network_blocked": blocked(lambda: socket.create_connection(("1.1.1.1", 53), 3)),
            "root_fs_read_only": blocked(lambda: Path("/escape").write_text("x")),
            "project_read_only": blocked(lambda: (Path(__file__).parent / "x").write_text("x")),
            "data_read_only": blocked(lambda: (data_dir / "x").write_text("x")),
            "non_root": float(os.getuid() != 0),
            "tmp_writable": 1.0 - blocked(lambda: Path("/tmp/x").write_text("x")),
        },
        input_example=x,
    )
"""


def _project(root: Path, train_py: str) -> tuple[Path, Path]:
    project, data = root / "project", root / "data"
    project.mkdir()
    data.mkdir()
    # Same requirements as the example, so every test reuses one environment image.
    shutil.copyfile(EXAMPLE / "requirements.txt", project / "requirements.txt")
    (project / "train.py").write_text(train_py)
    rows = "\n".join(",".join(str(i + j) for j in range(9)) for i in range(200))
    (data / "california_housing.csv").write_text(",".join([*COLUMNS, "MedHouseVal"]) + "\n" + rows)
    return project, data


def _settings(root: Path, **overrides: object) -> Settings:
    """CI points MLSHIP_TRACKING_URI at the compose server; otherwise a local SQLite store."""
    if "MLSHIP_TRACKING_URI" in os.environ:
        return Settings(**overrides)  # type: ignore[arg-type]
    store = {
        "tracking_uri": f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        "artifact_root": (root / "artifacts").as_uri(),
    }
    return Settings(**store, **overrides)  # type: ignore[arg-type]


def _train(project: Path, data: Path, **overrides: object) -> TrainingOutcome:
    services = build_services(_settings(project.parent, **overrides))
    return services.training.train(project, data, {}, model_name=_model_name())


_CREATED_MODELS: list[str] = []


def _model_name() -> str:
    name = f"it-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    _CREATED_MODELS.append(name)
    return name


@pytest.fixture(scope="module", autouse=True)
def _remove_test_models() -> Iterator[None]:
    """Keep a shared MLflow server (CI, or the one used for the demo) free of test runs."""
    yield
    if "MLSHIP_TRACKING_URI" in os.environ:
        remove_models(os.environ["MLSHIP_TRACKING_URI"], _CREATED_MODELS)


def test_example_trains_registers_and_promotes(tmp_path: Path) -> None:
    project, data = _project(tmp_path, (EXAMPLE / "train.py").read_text())
    services = build_services(_settings(tmp_path))
    name = _model_name()
    outcome = services.training.train(project, data, {}, model_name=name)

    assert outcome.succeeded, outcome.issues
    assert set(outcome.metrics) == {"rmse", "mae", "r2"}
    assert outcome.serialization == "skops"
    manifest = json.loads((outcome.run_dir / "manifest.json").read_text())
    assert manifest["model"]["files"]["MLmodel"]
    version = outcome.model_version
    assert version is not None and version.model_sha256 == manifest["model"]["sha256"]

    promotion = services.registry.promote(name, version.version, "champion")
    assert promotion.current.version == version.version
    (record,) = services.registry.list_runs(name, limit=5)
    assert record.run_id == outcome.run_id and record.model_version == version.version


def test_the_same_inputs_reproduce_the_same_model_behaviour(tmp_path: Path) -> None:
    """ADR-011: same inputs -> same hashes, metrics and predictions on a fixed input.

    Not compared on purpose: the whole manifest (run id, timestamps) and the bytes of
    the saved model (serialization is not byte-for-byte deterministic).
    """
    project, data = _project(tmp_path, (EXAMPLE / "train.py").read_text())
    services = build_services(_settings(tmp_path))
    name = _model_name()
    runs = [
        services.training.train(project, data, {"seed": "7"}, model_name=name) for _ in range(2)
    ]
    assert all(run.succeeded for run in runs), [run.issues for run in runs]

    first, second = (json.loads((r.run_dir / "manifest.json").read_text()) for r in runs)
    assert first["code"]["sha256"] == second["code"]["sha256"]
    assert first["data"]["sha256"] == second["data"]["sha256"]
    assert first["dependencies"]["lock_sha256"] == second["dependencies"]["lock_sha256"]
    assert first["environment"]["environment_key"] == second["environment"]["environment_key"]
    assert first["seed"] == second["seed"] == 7
    assert runs[0].metrics == runs[1].metrics

    fixed = pd.read_csv(data / "california_housing.csv").drop(columns=["MedHouseVal"]).head(20)
    predictions = [
        mlflow.pyfunc.load_model(str(run.run_dir / "output" / "model")).predict(fixed)
        for run in runs
    ]
    assert np.array_equal(np.asarray(predictions[0]), np.asarray(predictions[1]))


def test_the_container_is_isolated(tmp_path: Path) -> None:
    project, data = _project(tmp_path, ISOLATION_PROBE)
    outcome = _train(project, data)
    assert outcome.succeeded, outcome.issues
    assert outcome.metrics == {
        "network_blocked": 1.0,
        "root_fs_read_only": 1.0,
        "project_read_only": 1.0,
        "data_read_only": 1.0,
        "non_root": 1.0,
        "tmp_writable": 1.0,
    }
    assert not (project / "x").exists() and not (data / "x").exists()


def test_timeout_stops_training(tmp_path: Path) -> None:
    slow = "import time\n\ndef train(data_dir, params):\n    time.sleep(120)\n"
    project, data = _project(tmp_path, slow)
    outcome = _train(project, data, train_timeout_seconds=5)
    assert not outcome.succeeded
    assert [i.code for i in outcome.issues] == ["TRAIN_TIMEOUT"]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _post(url: str, payload: object) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(  # noqa: S310 - local test server
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_promoted_model_is_served_end_to_end(tmp_path: Path) -> None:
    """train -> promote -> serve -> predict, with the real images and container."""
    project, data = _project(tmp_path, (EXAMPLE / "train.py").read_text())
    services = build_services(_settings(tmp_path))
    name = _model_name()
    trained = services.training.train(project, data, {}, model_name=name)
    assert trained.succeeded and trained.model_version is not None, trained.issues
    services.registry.promote(name, trained.model_version.version, "champion")

    prepared = services.serving.prepare(project, name, "champion")
    assert prepared.environment_image == trained.environment.image, "same image as training"
    port, ready = _free_port(), threading.Event()
    urls: list[str] = []

    def on_ready(url: str) -> None:
        urls.append(url)
        ready.set()

    stopped: list[ContainerRunError] = []

    def serve() -> None:
        try:
            services.serving.run(
                prepared,
                port=port,
                workers=2,
                environment={},
                on_log=lambda _: None,
                on_ready=on_ready,
            )
        except ContainerRunError as exc:
            stopped.append(exc)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    try:
        assert ready.wait(timeout=300), "the API never became ready"
        row = dict.fromkeys(COLUMNS, 1)  # the fixture CSV has integers: long columns
        status, body = _post(f"{urls[0]}/v1/predict", {"rows": [row, row]})
        assert status == 200, body
        assert body["model"] == {"name": name, "version": str(trained.model_version.version)}
        assert len(body["predictions"]) == 2
        status, body = _post(f"{urls[0]}/v1/predict", {"rows": [{"MedInc": "x"}]})
        assert status == 422 and len(body["details"]) >= 2
        with urllib.request.urlopen(f"{urls[0]}/metrics", timeout=10) as response:  # noqa: S310
            assert f'mlship_model_info{{name="{name}"' in response.read().decode()
    finally:
        client = docker.from_env()
        for container in client.containers.list(filters={"label": f"org.mlship.model-name={name}"}):
            container.stop(timeout=10)
        server.join(timeout=60)
    assert len(stopped) == 1 and "stopped while serving" in stopped[0].message
