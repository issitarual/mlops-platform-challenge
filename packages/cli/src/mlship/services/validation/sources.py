"""The Python code that runs in the training container, parsed with ``ast`` (never run).

That code is ``train.py`` plus the project modules it imports, followed transitively:
the runner puts the project folder first on ``sys.path``, so ``import utils`` is
``utils.py`` next to train.py. Tests, scripts and notebooks that train.py never
imports do not run in the container and are not checked.
"""

import ast
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from mlship.domain.validation import Severity, ValidationIssue

MAX_SOURCE_BYTES = 1_000_000
MAX_SOURCE_FILES = 500
ENCODING_HINT = (
    "Save the file as UTF-8. In Windows PowerShell 5, `>` writes UTF-16: use "
    "`Set-Content -Encoding utf8` or an editor instead."
)
_ENTRY_POINT = "train.py"


@dataclass(frozen=True, slots=True)
class Source:
    relative: str
    tree: ast.Module


@dataclass(frozen=True, slots=True)
class ProjectSources:
    files: tuple[Source, ...]
    local_modules: frozenset[str]
    issues: tuple[ValidationIssue, ...]


def parse_source(path: Path) -> ast.Module:
    """From bytes, as Python imports it: a BOM and a ``# coding:`` line are honoured."""
    return ast.parse(path.read_bytes(), filename=path.name)


def is_encoding_error(exc: SyntaxError) -> bool:
    message = (exc.msg or "").lower()
    return "null bytes" in message or "codec" in message or "utf-8" in message


def read_sources(project_dir: Path, exclude: tuple[Path, ...] = ()) -> ProjectSources:
    """train.py and every project module reachable from it through imports.

    train.py's own problems are reported by its rule; a broken helper that train.py
    imports is a warning here, because Python reports it only when the import runs.
    """
    root = project_dir.resolve()
    excluded = tuple(p.resolve() for p in exclude)
    local = _local_modules(project_dir)
    files: list[Source] = []
    issues: list[ValidationIssue] = []
    seen: set[Path] = set()
    pending = deque([project_dir / _ENTRY_POINT])
    while pending:
        path = pending.popleft()
        resolved = path.resolve()
        if resolved in seen or not _inside(resolved, root, excluded):
            continue
        seen.add(resolved)
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_SOURCE_BYTES:
            continue
        relative = resolved.relative_to(root).as_posix()
        if len(files) >= MAX_SOURCE_FILES:
            issues.append(
                ValidationIssue(
                    "SOURCE_TOO_MANY_FILES",
                    Severity.WARNING,
                    f"Only the first {MAX_SOURCE_FILES} modules train.py imports were checked.",
                    relative,
                    "Keep vendored code out of the project folder.",
                )
            )
            break
        try:
            tree = parse_source(path)
        except SyntaxError as exc:
            if relative != _ENTRY_POINT:
                issues.append(_unparsable(relative, exc))
            continue
        files.append(Source(relative, tree))
        pending.extend(_imported_files(tree, path, project_dir, local))
    return ProjectSources(tuple(files), local, tuple(issues))


def _unparsable(relative: str, exc: SyntaxError) -> ValidationIssue:
    encoding = is_encoding_error(exc)
    return ValidationIssue(
        "SOURCE_ENCODING" if encoding else "SOURCE_SYNTAX",
        Severity.WARNING,
        f"{relative}, imported by train.py, cannot be parsed: {exc.msg}.",
        f"{relative}:{exc.lineno}" if exc.lineno else relative,
        ENCODING_HINT if encoding else "Fix it: the import fails when train.py runs.",
    )


def _imported_files(
    tree: ast.Module, path: Path, project_dir: Path, local: frozenset[str]
) -> Iterator[Path]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] in local:
                    yield from _module_files(project_dir, parts)
        elif isinstance(node, ast.ImportFrom):
            parts = node.module.split(".") if node.module else []
            if node.level == 0:
                if not parts or parts[0] not in local:
                    continue
                base = project_dir
            else:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
            yield from _module_files(base, parts)
            for alias in node.names:
                yield from _module_files(base, [*parts, alias.name])


def _module_files(base: Path, parts: list[str]) -> list[Path]:
    """The files that importing ``parts`` from ``base`` executes, packages included."""
    found: list[Path] = []
    current = base
    for part in parts:
        package = current / part
        if package.is_dir():
            if (package / "__init__.py").is_file():
                found.append(package / "__init__.py")
            current = package
            continue
        if (current / f"{part}.py").is_file():
            found.append(current / f"{part}.py")
        break
    return found


def _local_modules(project_dir: Path) -> frozenset[str]:
    """Top-level names the project itself provides: ``utils.py`` and ``features/``."""
    if not project_dir.is_dir():
        return frozenset()
    names = set()
    for entry in project_dir.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_file() and entry.suffix == ".py":
            names.add(entry.stem)
        elif entry.is_dir() and any(entry.glob("*.py")):
            names.add(entry.name)
    return frozenset(names)


def _inside(path: Path, root: Path, excluded: tuple[Path, ...]) -> bool:
    if not path.is_relative_to(root):
        return False
    return not any(path.is_relative_to(e) for e in excluded) and not any(
        part.startswith(".") for part in path.relative_to(root).parts
    )
