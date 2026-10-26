import dataclasses
import tomllib
from pathlib import Path

import pytest

from mlship_sdk import TrainOutput

ROOT = Path(__file__).resolve().parents[3]


def test_train_output_holds_fields() -> None:
    output = TrainOutput(model="m", metrics={"rmse": 0.5}, input_example=[[1.0]])
    assert output.model == "m"
    assert output.metrics == {"rmse": 0.5}
    assert output.input_example == [[1.0]]


def test_reference_data_is_optional() -> None:
    assert TrainOutput(model="m", metrics={}, input_example=None).reference_data is None


def test_train_output_is_immutable() -> None:
    output = TrainOutput(model="m", metrics={}, input_example=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        output.model = "other"  # type: ignore[misc]


def test_sdk_declares_no_dependencies() -> None:
    pyproject = tomllib.loads((ROOT / "packages/sdk/pyproject.toml").read_text())
    assert pyproject["project"]["dependencies"] == []
