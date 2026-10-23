"""Real training in the locked-down container (needs Docker and network for the build).

Run with: uv run pytest -m integration tests/integration -v
"""

import shutil
from pathlib import Path

import pytest

from mlship.bootstrap import build_services
from mlship.config import Settings
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


def _train(project: Path, data: Path, **settings: object) -> TrainingOutcome:
    return build_services(Settings(**settings)).training.train(project, data, {})  # type: ignore[arg-type]


def test_example_trains_in_the_container(tmp_path: Path) -> None:
    project, data = _project(tmp_path, (EXAMPLE / "train.py").read_text())
    outcome = _train(project, data)
    assert outcome.succeeded, outcome.issues
    assert set(outcome.metrics) == {"rmse", "mae", "r2"}
    assert outcome.serialization == "skops"
    assert (outcome.run_dir / "output" / "model" / "MLmodel").is_file()


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
