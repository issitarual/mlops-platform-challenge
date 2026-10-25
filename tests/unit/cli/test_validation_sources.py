"""Rules over the project's Python files: imports and container-hostile code."""

from pathlib import Path

from packaging.utils import canonicalize_name

from mlship.domain.validation import Severity
from mlship.services.validation.imports import check_imports
from mlship.services.validation.source_rules import check_sources
from mlship.services.validation.sources import read_sources
from mlship.services.validation.validator import ProjectValidator

AVAILABLE = frozenset(
    canonicalize_name(n) for n in ("pandas", "numpy", "mlship-sdk", "mlflow-skinny")
)


def _write(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def _imports(root: Path, files: dict[str, str]) -> list[tuple[str, str]]:
    sources = read_sources(_write(root, files))
    return [(i.message.split("'")[1], i.location) for i in check_imports(sources, AVAILABLE)]


def _codes(root: Path, source: str) -> list[tuple[str, str]]:
    sources = read_sources(_write(root, {"train.py": source}))
    return [(i.code, i.location) for i in check_sources(sources)]


def test_undeclared_imports_are_reported_once_with_the_package_to_add(tmp_path: Path) -> None:
    found = _imports(
        tmp_path,
        {
            "train.py": (
                "import os, json\n"
                "import pandas as pd\n"
                "import sklearn.pipeline\n"
                "from sklearn import linear_model\n"
                "import mlflow\n"
                "from mlship_sdk import TrainOutput\n"
                "from . import sibling\n"
                "import features\n"
                "import utils\n"
            ),
            "utils.py": "import yaml\n",
            "features/__init__.py": "",
        },
    )
    assert found == [("sklearn", "train.py:3"), ("yaml", "utils.py:1")]


def test_the_hint_names_the_real_package(tmp_path: Path) -> None:
    sources = read_sources(_write(tmp_path, {"train.py": "import cv2\nimport PIL.Image\n"}))
    hints = [i.hint for i in check_imports(sources, AVAILABLE)]
    assert "'opencv-python' or 'opencv-python-headless'" in hints[0]
    assert "'pillow'" in hints[1]


def test_optional_and_type_checking_imports_are_not_required(tmp_path: Path) -> None:
    found = _imports(
        tmp_path,
        {
            "train.py": (
                "from typing import TYPE_CHECKING\n"
                "try:\n    import lightgbm\nexcept ImportError:\n    lightgbm = None\n"
                "if TYPE_CHECKING:\n    import torch\n"
                "def train(data_dir, params):\n    import xgboost\n"
            )
        },
    )
    assert found == [("xgboost", "train.py:9")]


def test_only_code_reachable_from_train_py_is_checked(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "train.py": (
                "import features.build\nfrom helpers import clean\nfrom data import loader\n"
            ),
            "features/__init__.py": "",
            "features/build.py": "from . import common\nfrom .common import x\n",
            "features/common.py": "x = 1\n",
            "helpers.py": "def clean(): ...\n",
            "data/loader.py": "import torch\n",
            "tests/test_train.py": "import pytest\nimport requests\n",
            "scripts/download.py": "import requests\nrequests.get('https://x.io')\n",
            "notes.py": "import matplotlib\n",
        },
    )
    sources = read_sources(tmp_path, exclude=(tmp_path / "data",))
    assert sorted(s.relative for s in sources.files) == [
        "features/__init__.py",
        "features/build.py",
        "features/common.py",
        "helpers.py",
        "train.py",
    ]
    assert check_sources(sources) == []
    assert check_imports(sources, AVAILABLE) == []


def test_a_broken_helper_warns_only_when_train_py_imports_it(tmp_path: Path) -> None:
    _write(tmp_path, {"train.py": "def train(:\n", "unused.py": "def x(:\n"})
    assert read_sources(tmp_path).issues == ()
    _write(tmp_path, {"train.py": "import helper\n", "helper.py": "def x(:\n"})
    (issue,) = read_sources(tmp_path).issues
    assert (issue.code, issue.severity, issue.location) == (
        "SOURCE_SYNTAX",
        Severity.WARNING,
        "helper.py:1",
    )


def test_files_are_read_as_python_reads_them(tmp_path: Path) -> None:
    validator = ProjectValidator([])
    (tmp_path / "requirements.txt").write_text("")
    train = tmp_path / "train.py"

    train.write_bytes(b"\xef\xbb\xbfdef train(data_dir, params):\n    pass\n")
    assert validator.validate(tmp_path, None).issues == (), "a BOM is fine for Python"

    train.write_bytes(b"# -*- coding: latin-1 -*-\n# caf\xe9\ndef train(d, p):\n    pass\n")
    assert validator.validate(tmp_path, None).issues == ()

    train.write_bytes("def train(d, p):\n    pass\n".encode("utf-16"))
    (issue,) = validator.validate(tmp_path, None).issues
    assert issue.code == "TRAIN_ENCODING" and "UTF-8" in issue.hint


def test_requirements_with_a_bom_are_read_and_utf16_is_refused(tmp_path: Path) -> None:
    validator = ProjectValidator(["pandas>=2.2.2"])
    (tmp_path / "train.py").write_text("import sklearn\ndef train(d, p):\n    pass\n")
    requirements = tmp_path / "requirements.txt"

    requirements.write_bytes(b"\xef\xbb\xbfscikit-learn==1.9.1\n")
    assert validator.validate(tmp_path, None).issues == ()

    requirements.write_bytes("scikit-learn==1.9.1\r\n".encode("utf-16"))
    codes = [i.code for i in validator.validate(tmp_path, None).issues]
    assert codes == ["REQUIREMENTS_ENCODING"], "imports are not judged against an unread file"


def test_a_missing_project_is_one_clear_error(tmp_path: Path) -> None:
    for target in (tmp_path / "typo", tmp_path / "a-file.txt"):
        if target.suffix:
            target.write_text("x")
        (issue,) = ProjectValidator([]).validate(target, tmp_path / "data").issues
        assert issue.code == "PROJECT_DIR_MISSING" and "mlship init" in issue.hint


def test_work_at_import_time_is_flagged_but_ordinary_module_code_is_not(tmp_path: Path) -> None:
    source = (
        '"""Docstring."""\n'
        "import logging\n"
        "import pandas as pd\n"
        "LOGGER = logging.getLogger(__name__)\n"
        "COLUMNS = ['a', 'b']\n"
        "FRAME = pd.read_csv('data.csv')\n"
        "for i in range(3):\n    pass\n"
        "def train(data_dir, params):\n    return pd.read_csv(data_dir / 'x.csv')\n"
        "if __name__ == '__main__':\n    train(None, {})\n"
    )
    assert _codes(tmp_path, source) == [
        ("SOURCE_MODULE_LEVEL_WORK", "train.py:6"),
        ("SOURCE_MODULE_LEVEL_WORK", "train.py:7"),
    ]


def test_network_absolute_paths_and_secret_variables_are_flagged(tmp_path: Path) -> None:
    source = (
        "import os\n"
        "import requests\n"
        "from urllib.request import urlopen\n"
        "from urllib import parse\n"
        "def train(data_dir, params):\n"
        "    '''Reads /home/laura/data in the docstring: not flagged.'''\n"
        "    a = 'https://example.com/data.csv'\n"
        "    b = 's3://bucket/key'\n"
        "    c = '/home/laura/data.csv'\n"
        "    d = 'C:\\\\Users\\\\laura\\\\data.csv'\n"
        "    e = os.environ['AWS_SECRET_ACCESS_KEY']\n"
        "    f = os.getenv('API_TOKEN')\n"
        "    g = os.environ.get('MLFLOW_TRACKING_PASSWORD', '')\n"
        "    h = os.environ.get('N_JOBS')\n"
        "    i = '/v1/predict'\n"
        "    j = '/dbfs/mnt/raw/x.csv'\n"
    )
    assert _codes(tmp_path, source) == [
        ("SOURCE_NETWORK", "train.py:2"),
        ("SOURCE_NETWORK", "train.py:3"),
        ("SOURCE_NETWORK", "train.py:7"),
        ("SOURCE_NETWORK", "train.py:8"),
        ("SOURCE_ABSOLUTE_PATH", "train.py:9"),
        ("SOURCE_ABSOLUTE_PATH", "train.py:10"),
        ("SOURCE_ABSOLUTE_PATH", "train.py:16"),
        ("SOURCE_SECRET_ENV", "train.py:11"),
        ("SOURCE_SECRET_ENV", "train.py:12"),
        ("SOURCE_SECRET_ENV", "train.py:13"),
    ]


def test_a_project_with_many_problems_reports_every_one_of_them(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "train.py": (
                "import requests\nimport sklearn\n"
                "def train(data_dir):\n    return requests.get('https://x.io')\n"
                "def train(data_dir, params):\n    pass\n"
            ),
            "requirements.txt": "pandas==2.0.0\n--extra-index-url https://x\nnumpy\n",
        },
    )
    (tmp_path / "data").mkdir()
    report = ProjectValidator(["pandas>=3.1.0", "numpy>=2.0"]).validate(tmp_path, tmp_path / "data")
    assert {i.code for i in report.errors} == {
        "TRAIN_FUNCTION_DUPLICATE",
        "REQUIREMENTS_RUNTIME_CONFLICT",
        "REQUIREMENTS_OPTION_BLOCKED",
        "DATA_DIR_EMPTY",
    }
    assert {i.code for i in report.warnings} == {
        "REQUIREMENTS_UNPINNED",
        "REQUIREMENTS_IMPORT_UNDECLARED",
        "SOURCE_NETWORK",
    }
    assert report.blocks(strict=False)
