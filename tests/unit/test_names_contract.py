"""The runner and the CLI must agree on which metric and param names are valid."""

import pytest

from mlship.domain import names as cli_names
from mlship_runtime.runner import names as runner_names

SAMPLES = [
    "rmse",
    "r2",
    "val/rmse",
    "f1 macro",
    "métrica",
    "a:b",
    "rmse (test)",
    "f1@0.5",
    "../x",
    "/abs",
    ".",
    "",
    "x" * 250,
    "x" * 251,
]


@pytest.mark.parametrize("name", SAMPLES)
def test_both_sides_apply_the_same_rule(name: str) -> None:
    assert cli_names.is_valid_name(name) == runner_names.is_valid_name(name)


def test_the_rule_is_the_strictest_one_mlflow_has() -> None:
    """Colon is valid for MLflow on Linux but not on Windows clients: refused everywhere."""
    assert not cli_names.is_valid_name("a:b")
    assert cli_names.is_valid_name("val/rmse") and cli_names.is_valid_name("métrica")
