import json
from pathlib import Path
from typing import Any

import pytest
from runner_fixtures import GOOD_TRAIN, write_config

from mlship_runtime.runner.__main__ import main


def _run(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch, train_py: str
) -> tuple[int, dict[str, Any]]:
    (workspace["project"] / "train.py").write_text(train_py)
    config = write_config(workspace["project"].parent / "run.json", workspace)
    monkeypatch.setenv("MLSHIP_RUN_CONFIG", str(config))
    code = main()
    return code, json.loads((workspace["output"] / "result.json").read_text())


def test_successful_run_writes_model_metrics_and_result(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    code, result = _run(workspace, monkeypatch, GOOD_TRAIN)
    assert code == 0
    assert result["status"] == "succeeded"
    assert result["errors"] == []
    assert result["model"]["serialization"] == "skops"
    assert set(result["metrics"]) == {"rmse"}
    assert (workspace["output"] / "model" / "MLmodel").is_file()
    assert json.loads((workspace["output"] / "metrics.json").read_text()) == result["metrics"]


def test_exception_in_train_is_reported(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    code, result = _run(
        workspace, monkeypatch, "def train(data_dir, params):\n    raise KeyError('price')\n"
    )
    assert code == 1
    assert [e["code"] for e in result["errors"]] == ["TRAIN_RAISED"]
    assert "KeyError" in result["errors"][0]["message"]


def test_missing_train_function_is_reported(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    code, result = _run(workspace, monkeypatch, "def fit(data_dir, params):\n    pass\n")
    assert code == 1
    assert [e["code"] for e in result["errors"]] == ["TRAIN_FUNCTION_MISSING"]


def test_invalid_output_reports_every_issue(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    train_py = (
        "from mlship_sdk import TrainOutput\n"
        "def train(data_dir, params):\n"
        "    return TrainOutput(\n"
        "        model=object(), metrics={'rmse': float('nan')}, input_example=None\n"
        "    )\n"
    )
    code, result = _run(workspace, monkeypatch, train_py)
    assert code == 1
    assert {e["code"] for e in result["errors"]} == {
        "MODEL_NO_PREDICT",
        "METRIC_NOT_FINITE",
        "INPUT_EXAMPLE_MISSING",
    }
    assert not (workspace["output"] / "model").exists()


def test_model_that_cannot_be_saved_is_a_user_error_not_an_mlship_bug(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_: object) -> None:
        raise TypeError("unsupported input_example type")

    monkeypatch.setattr("mlship_runtime.runner.__main__.save_model", refuse)
    code, result = _run(workspace, monkeypatch, GOOD_TRAIN)
    assert code == 1
    assert [e["code"] for e in result["errors"]] == ["MODEL_SAVE_FAILED"]
    assert "unsupported input_example type" in result["errors"][0]["message"]
