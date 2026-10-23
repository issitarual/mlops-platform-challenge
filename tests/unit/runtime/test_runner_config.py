from pathlib import Path
from typing import Any

import pytest

from mlship_runtime.runner.config import parse_config

VALID: dict[str, Any] = {
    "schema_version": 1,
    "project_dir": "/mlship/project",
    "data_dir": "/mlship/data",
    "output_dir": "/output",
    "params": {"seed": "42"},
    "seed": 42,
    "serialization": {"allow_pickle": False, "skops_trusted_types": ["a.B"]},
}


def test_valid_config_is_parsed() -> None:
    config = parse_config(VALID)
    assert config.output_dir == Path("/output")
    assert config.serialization.skops_trusted_types == ("a.B",)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", 2),
        ("params", {"seed": 42}),
        ("seed", "42"),
        ("seed", True),
        ("serialization", {"allow_pickle": "no", "skops_trusted_types": []}),
        ("output_dir", None),
    ],
)
def test_invalid_fields_are_rejected(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=r"invalid run\.json"):
        parse_config({**VALID, key: value})
