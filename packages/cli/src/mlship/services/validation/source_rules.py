"""Warnings about code that will misbehave in the training container (ADR-021).

Heuristics over the syntax tree: they point at likely mistakes, never block on their
own, and ``--strict`` turns them into errors for CI.
"""

import ast
import re
from collections.abc import Iterator

from mlship.domain.validation import Severity, ValidationIssue
from mlship.services.validation.sources import ProjectSources

NETWORK_MODULES = frozenset(
    {
        "aiohttp",
        "boto3",
        "botocore",
        "ftplib",
        "http.client",
        "httpx",
        "paramiko",
        "requests",
        "smtplib",
        "socket",
        "urllib.request",
        "urllib3",
        "websocket",
        "websockets",
    }
)
_URL = re.compile(r"^(https?|s3|gs|ftp|abfss?)://", re.IGNORECASE)
_ABSOLUTE_PATH = re.compile(
    r"^(/(home|Users|mnt|data|tmp|opt|var|root|srv|dbfs|Volumes|Workspace|content)/"
    r"|[A-Za-z]:[\\/]|~/)"
)
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)
_HEAVY_CALLS = frozenset(
    {
        "download",
        "fit",
        "fit_transform",
        "from_pretrained",
        "load",
        "load_dataset",
        "open",
        "read_csv",
        "read_excel",
        "read_json",
        "read_parquet",
        "read_sql",
        "read_table",
        "sleep",
        "train",
        "urlopen",
    }
)


def check_sources(sources: ProjectSources) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for source in sources.files:
        issues.extend(_module_level_work(source.relative, source.tree))
        issues.extend(_network(source.relative, source.tree))
        issues.extend(_constants(source.relative, source.tree))
        issues.extend(_secret_environment(source.relative, source.tree))
    return issues


def _module_level_work(relative: str, tree: ast.Module) -> Iterator[ValidationIssue]:
    for statement in tree.body:
        if isinstance(
            statement,
            ast.Import | ast.ImportFrom | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        ) or _is_main_guard(statement):
            continue
        loop = isinstance(statement, ast.For | ast.While | ast.AsyncFor)
        call = next((c for c in _calls(statement) if _call_name(c) in _HEAVY_CALLS), None)
        if loop or call is not None:
            what = "a loop" if loop else f"{_call_name(call)}()"
            yield ValidationIssue(
                "SOURCE_MODULE_LEVEL_WORK",
                Severity.WARNING,
                f"{what} runs when the module is imported, outside train().",
                f"{relative}:{statement.lineno}",
                "Move it into train() or a function it calls: work at import time runs "
                "before mlship sets the seeds, and again whenever the module is imported.",
            )


def _network(relative: str, tree: ast.Module) -> Iterator[ValidationIssue]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
        else:
            continue
        hit = next((n for n in names if _is_network_module(n)), None)
        if hit is not None:
            yield ValidationIssue(
                "SOURCE_NETWORK",
                Severity.WARNING,
                f"{hit!r} is a network client, and the training container has no network.",
                f"{relative}:{node.lineno}",
                "Read the data from data_dir; download it before `mlship train`.",
            )


def _is_network_module(name: str) -> bool:
    return name in NETWORK_MODULES or name.split(".")[0] in NETWORK_MODULES


def _constants(relative: str, tree: ast.Module) -> Iterator[ValidationIssue]:
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        value = node.value.strip()
        if _URL.match(value):
            yield ValidationIssue(
                "SOURCE_NETWORK",
                Severity.WARNING,
                f"{_shorten(value)!r} points outside the container, which has no network.",
                f"{relative}:{node.lineno}",
                "Read the data from data_dir; download it before `mlship train`.",
            )
        elif _ABSOLUTE_PATH.match(value):
            yield ValidationIssue(
                "SOURCE_ABSOLUTE_PATH",
                Severity.WARNING,
                f"{_shorten(value)!r} is an absolute path that will not exist in the container.",
                f"{relative}:{node.lineno}",
                "Build paths from data_dir (for data) or Path(__file__).parent (for "
                "project files).",
            )


def _secret_environment(relative: str, tree: ast.Module) -> Iterator[ValidationIssue]:
    for node in ast.walk(tree):
        name = _environment_read(node)
        if name is not None and _SECRET_NAME.search(name):
            yield ValidationIssue(
                "SOURCE_SECRET_ENV",
                Severity.WARNING,
                f"Reads the environment variable {name!r}, which looks like a secret.",
                f"{relative}:{getattr(node, 'lineno', 0)}",
                "The training container receives no host variables or credentials, so it "
                "will be empty. Training code should not need secrets: read data from "
                "data_dir.",
            )


def _environment_read(node: ast.AST) -> str | None:
    """The literal name in ``os.environ[...]``, ``os.environ.get(...)``, ``os.getenv(...)``."""
    if isinstance(node, ast.Subscript) and _is_environ(node.value):
        key = node.slice
    elif isinstance(node, ast.Call) and node.args:
        function = node.func
        is_get = (
            isinstance(function, ast.Attribute)
            and function.attr == "get"
            and _is_environ(function.value)
        )
        is_getenv = (isinstance(function, ast.Attribute) and function.attr == "getenv") or (
            isinstance(function, ast.Name) and function.id == "getenv"
        )
        if not (is_get or is_getenv):
            return None
        key = node.args[0]
    else:
        return None
    return key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None


def _is_environ(node: ast.expr) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


def _is_main_guard(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.If) or not isinstance(statement.test, ast.Compare):
        return False
    left = statement.test.left
    return isinstance(left, ast.Name) and left.id == "__name__"


def _calls(statement: ast.stmt) -> Iterator[ast.Call]:
    return (n for n in ast.walk(statement) if isinstance(n, ast.Call))


def _call_name(call: ast.Call | None) -> str:
    if call is None:
        return ""
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr
    return function.id if isinstance(function, ast.Name) else ""


def _docstring_nodes(tree: ast.Module) -> set[int]:
    nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                nodes.add(id(body[0].value))
    return nodes


def _shorten(value: str) -> str:
    return value if len(value) <= 60 else f"{value[:57]}..."
