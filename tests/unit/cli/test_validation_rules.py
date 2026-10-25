from pathlib import Path

import pytest

from mlship.domain.validation import Severity
from mlship.services.validation.data import check_data_dir
from mlship.services.validation.requirements import check_requirements, runtime_specifiers
from mlship.services.validation.train_script import check_train_script

RUNTIME = runtime_specifiers(["pandas>=3.1.0", "numpy>=2.0; python_version >= '3.10'"])


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


def _requirements(tmp_path: Path, text: str) -> list[tuple[str, str]]:
    (tmp_path / "requirements.txt").write_text(text)
    return [(i.code, i.location) for i in check_requirements(tmp_path, RUNTIME).issues]


def test_requirements_report_every_problem_with_its_line(tmp_path: Path) -> None:
    issues = _requirements(
        tmp_path,
        "# comment\n"
        "pandas==3.1.0\n"
        "--index-url https://mirror.example/simple\n"
        "-e .\n"
        "git+https://github.com/x/y\n"
        "pkg @ https://example.com/pkg.whl\n"
        "./local_pkg\n"
        "scikit-learn  # inline comment\n"
        "numpy==1.26.4\n"
        "pandas=2\n"
        "mlship-sdk\n"
        "requests==2.32.3 \\\n"
        "    --hash=sha256:abc\n"
        "https://example.com/pkg.whl\n",
    )
    assert issues == [
        ("REQUIREMENTS_OPTION_BLOCKED", "requirements.txt:3"),
        ("REQUIREMENTS_OPTION_BLOCKED", "requirements.txt:4"),
        ("REQUIREMENTS_DIRECT_REFERENCE", "requirements.txt:5"),
        ("REQUIREMENTS_DIRECT_REFERENCE", "requirements.txt:6"),
        ("REQUIREMENTS_LOCAL_PATH", "requirements.txt:7"),
        ("REQUIREMENTS_UNPINNED", "requirements.txt:8"),
        ("REQUIREMENTS_RUNTIME_CONFLICT", "requirements.txt:9"),
        ("REQUIREMENTS_SYNTAX", "requirements.txt:10"),
        ("REQUIREMENTS_PLATFORM_PACKAGE", "requirements.txt:11"),
        ("REQUIREMENTS_OPTION_BLOCKED", "requirements.txt:12"),
        ("REQUIREMENTS_DIRECT_REFERENCE", "requirements.txt:14"),
    ]


def test_runtime_conflict_names_what_the_runtime_needs(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("Pandas==2.2.3\n")
    (issue,) = check_requirements(tmp_path, RUNTIME).issues
    assert issue.code == "REQUIREMENTS_RUNTIME_CONFLICT" and issue.severity is Severity.ERROR
    assert "pandas>=3.1.0" in issue.message.lower()


def test_compatible_pins_extras_and_markers_are_valid(tmp_path: Path) -> None:
    assert (
        _requirements(
            tmp_path,
            "pandas[performance]==3.1.0\n"
            "scikit_learn==1.9.1 ; python_version >= '3.10'\n"
            "xgboost===3.1.0\n",
        )
        == []
    )
    report = check_requirements(tmp_path, RUNTIME)
    assert report.declared == {"pandas", "scikit-learn", "xgboost"}


def test_ranges_and_wildcards_are_unpinned_warnings(tmp_path: Path) -> None:
    codes = _requirements(tmp_path, "pandas>=3.1,<4\nnumpy==2.*\n")
    assert [c for c, _ in codes] == ["REQUIREMENTS_UNPINNED", "REQUIREMENTS_UNPINNED"]


def test_empty_requirements_are_valid(tmp_path: Path) -> None:
    assert _requirements(tmp_path, "") == []


def test_missing_requirements(tmp_path: Path) -> None:
    assert [i.code for i in check_requirements(tmp_path, RUNTIME).issues] == [
        "REQUIREMENTS_MISSING"
    ]


def test_data_dir_must_exist_and_have_files(tmp_path: Path) -> None:
    assert check_data_dir(tmp_path / "nope")[0].code == "DATA_DIR_MISSING"
    (tmp_path / "empty").mkdir()
    assert check_data_dir(tmp_path / "empty")[0].code == "DATA_DIR_EMPTY"
    (tmp_path / "empty" / ".DS_Store").write_text("a")
    assert check_data_dir(tmp_path / "empty")[0].code == "DATA_DIR_EMPTY"
    (tmp_path / "empty" / "x.csv").write_text("a")
    assert check_data_dir(tmp_path / "empty") == []


def test_unusual_data_formats_are_one_warning_with_examples(tmp_path: Path) -> None:
    for name in ("a.csv", "b.xlsx", "c.pkl", "d.h5", "e.bin", "nested/f.PARQUET"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("x")
    (issue,) = check_data_dir(tmp_path)
    assert issue.code == "DATA_FORMAT_UNKNOWN" and issue.severity is Severity.WARNING
    assert "4 file(s)" in issue.message and "and 1 more" in issue.message
