"""Rules for train.py, via ``ast`` only (the file is parsed, never imported)."""

import ast
from pathlib import Path

from mlship.domain.validation import Severity, ValidationIssue
from mlship.services.validation.sources import ENCODING_HINT, is_encoding_error, parse_source

_SIGNATURE_HINT = "Use: def train(data_dir: Path, params: dict) -> TrainOutput"


def check_train_script(project_dir: Path) -> list[ValidationIssue]:
    path = project_dir / "train.py"
    if not path.is_file():
        return [
            ValidationIssue(
                "TRAIN_FILE_MISSING",
                Severity.ERROR,
                "train.py was not found.",
                str(path),
                "Create train.py, or run `mlship init`.",
            )
        ]
    try:
        tree = parse_source(path)
    except SyntaxError as exc:
        if is_encoding_error(exc):
            return [
                ValidationIssue(
                    "TRAIN_ENCODING",
                    Severity.ERROR,
                    f"train.py cannot be read as Python source: {exc.msg}.",
                    "train.py",
                    ENCODING_HINT,
                )
            ]
        return [
            ValidationIssue(
                "TRAIN_SYNTAX",
                Severity.ERROR,
                f"Syntax error: {exc.msg}.",
                f"train.py:{exc.lineno}" if exc.lineno else "train.py",
                "Fix the syntax; the file must run with Python 3.12.",
            )
        ]
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "train"
    ]
    if not functions:
        return [
            ValidationIssue(
                "TRAIN_FUNCTION_MISSING",
                Severity.ERROR,
                "No top-level function named train.",
                "train.py",
                _SIGNATURE_HINT,
            )
        ]
    issues: list[ValidationIssue] = []
    if len(functions) > 1:
        lines = ", ".join(str(f.lineno) for f in functions)
        issues.append(
            ValidationIssue(
                "TRAIN_FUNCTION_DUPLICATE",
                Severity.ERROR,
                f"train is defined more than once (lines {lines}).",
                "train.py",
                "Keep a single train function; the last one would silently win.",
            )
        )
    function = functions[-1]
    args = function.args
    positional = [*args.posonlyargs, *args.args]
    if isinstance(function, ast.AsyncFunctionDef):
        issues.append(
            ValidationIssue(
                "TRAIN_SIGNATURE",
                Severity.ERROR,
                "train must not be async.",
                f"train.py:{function.lineno}",
                _SIGNATURE_HINT,
            )
        )
    required = len(positional) - len(args.defaults)
    required_keyword_only = any(default is None for default in args.kw_defaults)
    if len(positional) < 2 or required > 2 or required_keyword_only:
        issues.append(
            ValidationIssue(
                "TRAIN_SIGNATURE",
                Severity.ERROR,
                "train must accept exactly two arguments: data_dir and params.",
                f"train.py:{function.lineno}",
                _SIGNATURE_HINT,
            )
        )
    return issues
