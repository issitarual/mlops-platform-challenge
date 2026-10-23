import dataclasses

import pytest

from mlship.domain.environment import EnvironmentInputs

BASE = EnvironmentInputs(
    lock_sha256="a" * 64,
    runtime_source_sha256="b" * 64,
    dockerfile_sha256="c" * 64,
    platform="linux/amd64",
)


def test_key_is_deterministic() -> None:
    assert BASE.key() == dataclasses.replace(BASE).key()


@pytest.mark.parametrize(
    "field", ["lock_sha256", "runtime_source_sha256", "dockerfile_sha256", "platform"]
)
def test_key_changes_when_any_input_changes(field: str) -> None:
    changed = dataclasses.replace(BASE, **{field: "x"})
    assert changed.key() != BASE.key()
