"""The drift baseline: decile bins per numeric feature, saved with the model."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from runner_fixtures import GOOD_TRAIN, write_config

from mlship_runtime import reference
from mlship_runtime.runner.__main__ import main

FRAME = pd.DataFrame(
    {
        "rooms": np.arange(1000, dtype=float),
        "age": np.arange(1000) % 50,
        "city": ["a", "b"] * 500,
        "flag": [True, False] * 500,
    }
)


def test_deciles_split_the_training_rows_evenly() -> None:
    baseline = reference.compute(FRAME, ["rooms", "age", "city", "flag"])
    assert set(baseline.features) == {"rooms", "age"}, "text and booleans have no deciles"
    rooms = baseline.features["rooms"]
    assert len(rooms.edges) == 9 and len(rooms.shares) == 10
    assert sum(rooms.shares) == pytest.approx(1.0)
    assert all(share == pytest.approx(0.1, abs=0.01) for share in rooms.shares)
    assert rooms.bin_of(-1.0) == 0 and rooms.bin_of(10_000.0) == 9


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([[1.0]], "pandas DataFrame"),
        (FRAME[["age"]], "lacks the model input columns"),
        (FRAME.head(10), "at least 100"),
    ],
)
def test_unusable_reference_data_says_what_to_pass(data: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        reference.compute(data, ["rooms", "age"])


def test_round_trip_and_malformed_files(tmp_path: Path) -> None:
    baseline = reference.compute(FRAME, ["rooms", "age"])
    reference.save(baseline, tmp_path)
    assert reference.load(tmp_path) == baseline
    (tmp_path / reference.REFERENCE_FILE).write_text('{"schema_version": 1, "features": 3}')
    assert reference.load(tmp_path) is None
    (tmp_path / reference.REFERENCE_FILE).unlink()
    assert reference.load(tmp_path) is None


def _run(workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch, train_py: str) -> int:
    (workspace["project"] / "train.py").write_text(train_py)
    config = write_config(workspace["project"].parent / "run.json", workspace)
    monkeypatch.setenv("MLSHIP_RUN_CONFIG", str(config))
    return main()


WITH_REFERENCE = GOOD_TRAIN.replace(
    "input_example=x.head(3))",
    "input_example=x.head(3), reference_data=REFERENCE)",
) + ("\nREFERENCE = pd.DataFrame({'a': np.arange(200.0), 'b': np.arange(200.0) * 2})\n")


def test_the_baseline_is_saved_inside_the_model_folder(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(workspace, monkeypatch, WITH_REFERENCE) == 0
    saved = reference.load(workspace["output"] / "model")
    assert saved is not None and set(saved.features) == {"a", "b"} and saved.rows == 200
    assert "drift baseline for 2 feature(s)" in capsys.readouterr().err


def test_without_reference_data_training_succeeds_and_says_drift_is_off(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(workspace, monkeypatch, GOOD_TRAIN) == 0
    assert not (workspace["output"] / "model" / reference.REFERENCE_FILE).exists()
    assert "drift monitoring will be off" in capsys.readouterr().err


def test_invalid_reference_data_fails_with_a_hint(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = GOOD_TRAIN.replace(
        "input_example=x.head(3))", "input_example=x.head(3), reference_data=[1, 2])"
    )
    assert _run(workspace, monkeypatch, broken) == 1
    result = json.loads((workspace["output"] / "result.json").read_text())
    assert [e["code"] for e in result["errors"]] == ["REFERENCE_DATA_INVALID"]
    assert "optional" in result["errors"][0]["hint"]
