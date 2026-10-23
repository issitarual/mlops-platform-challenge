from pathlib import Path

import pytest

from mlship.domain.validation import Severity
from mlship.services.validation.data import check_data_dir
from mlship.services.validation.requirements import check_requirements
from mlship.services.validation.train_script import check_train_script


def _train(tmp_path: Path, source: str) -> set[str]:
    (tmp_path / "train.py").write_text(source)
    return {i.code for i in check_train_script(tmp_path)}


def test_valid_train_script(tmp_path: Path) -> None:
    assert _train(tmp_path, "def train(data_dir, params, extra=1):\n    pass\n") == set()


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("def train(:\n", "TRAIN_SYNTAX"),
        ("def fit(data_dir, params):\n    pass\n", "TRAIN_FUNCTION_MISSING"),
        ("def train(data_dir):\n    pass\n", "TRAIN_SIGNATURE"),
        ("def train(a, b, c):\n    pass\n", "TRAIN_SIGNATURE"),
        ("def train(a, b, *, c):\n    pass\n", "TRAIN_SIGNATURE"),
        ("async def train(a, b):\n    pass\n", "TRAIN_SIGNATURE"),
    ],
)
def test_train_script_errors(tmp_path: Path, source: str, code: str) -> None:
    assert code in _train(tmp_path, source)


def test_duplicate_train_is_reported(tmp_path: Path) -> None:
    source = "def train(a, b):\n    pass\n\ndef train(a, b):\n    pass\n"
    assert _train(tmp_path, source) == {"TRAIN_FUNCTION_DUPLICATE"}


def test_missing_train_script(tmp_path: Path) -> None:
    assert {i.code for i in check_train_script(tmp_path)} == {"TRAIN_FILE_MISSING"}


def test_requirements_report_every_problem_with_its_line(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text(
        "# comment\n"
        "pandas==3.1.0\n"
        "--index-url https://mirror.example/simple\n"
        "-e .\n"
        "git+https://github.com/x/y\n"
        "pkg @ https://example.com/pkg.whl\n"
        "./local_pkg\n"
        "scikit-learn\n"
    )
    issues = check_requirements(tmp_path)
    assert [(i.code, i.location) for i in issues] == [
        ("REQUIREMENTS_OPTION_BLOCKED", "requirements.txt:3"),
        ("REQUIREMENTS_OPTION_BLOCKED", "requirements.txt:4"),
        ("REQUIREMENTS_DIRECT_REFERENCE", "requirements.txt:5"),
        ("REQUIREMENTS_DIRECT_REFERENCE", "requirements.txt:6"),
        ("REQUIREMENTS_LOCAL_PATH", "requirements.txt:7"),
        ("REQUIREMENTS_UNPINNED", "requirements.txt:8"),
    ]
    assert issues[-1].severity is Severity.WARNING


def test_empty_requirements_are_valid(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("")
    assert check_requirements(tmp_path) == []


def test_data_dir_must_exist_and_have_files(tmp_path: Path) -> None:
    assert check_data_dir(tmp_path / "nope")[0].code == "DATA_DIR_MISSING"
    (tmp_path / "empty").mkdir()
    assert check_data_dir(tmp_path / "empty")[0].code == "DATA_DIR_EMPTY"
    (tmp_path / "empty" / "x.csv").write_text("a")
    assert check_data_dir(tmp_path / "empty") == []
