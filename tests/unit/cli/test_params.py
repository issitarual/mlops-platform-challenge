import pytest

from mlship.domain.params import MAX_PARAMS, param_issues


def _codes(params: dict[str, str]) -> list[str]:
    return [i.code for i in param_issues(params)]


def test_ordinary_hyperparameters_pass() -> None:
    assert _codes({"learning_rate": "0.05", "max_iter": "200", "model/depth": "3"}) == []


@pytest.mark.parametrize("key", ["lr (fast)", "f1@0.5", "../x", "a:b", ""])
def test_names_mlflow_would_refuse_on_some_os_are_rejected(key: str) -> None:
    assert "PARAM_NAME_INVALID" in _codes({key: "1"})


@pytest.mark.parametrize("key", ["password", "DB_PASSWD", "api_key", "aws-access-key", "token"])
def test_secret_looking_names_are_rejected(key: str) -> None:
    assert "PARAM_LOOKS_SECRET" in _codes({key: "x"})


def test_too_many_params_and_too_long_values_are_rejected() -> None:
    many = {f"p{i}": "1" for i in range(MAX_PARAMS + 1)}
    assert _codes(many) == ["PARAMS_TOO_MANY"]
    assert _codes({"notes": "n" * 6_001}) == ["PARAM_VALUE_TOO_LONG"]
