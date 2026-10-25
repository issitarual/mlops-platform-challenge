from pathlib import Path

import pytest

from mlship.domain.errors import MlshipError
from mlship.domain.runs import model_name_from_directory, resolve_model_name


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("california-housing", "california-housing"),
        ("My Project (v2)", "my-project-v2"),
        ("__init__", "init"),
    ],
)
def test_folder_names_become_safe_model_names(folder: str, expected: str) -> None:
    assert model_name_from_directory(folder) == expected


def test_explicit_name_wins_over_the_folder(tmp_path: Path) -> None:
    assert resolve_model_name(tmp_path, "churn") == "churn"


@pytest.mark.parametrize("name", ["Churn", "-x", "a" * 101, "a b"])
def test_invalid_explicit_names_are_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(MlshipError, match="not a valid model name"):
        resolve_model_name(tmp_path, name)
