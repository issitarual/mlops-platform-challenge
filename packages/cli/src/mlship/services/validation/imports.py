"""Imports that no declared package provides (ADR-021).

The import name is mapped to a package name: the curated map covers the common names
that differ (``sklearn`` is ``scikit-learn``); otherwise both are assumed equal.
Imports guarded by ``try/except ImportError`` and ``if TYPE_CHECKING`` are optional
and not reported.
"""

import ast
import sys
from collections.abc import Iterator

from packaging.utils import NormalizedName, canonicalize_name

from mlship.domain.validation import Severity, ValidationIssue
from mlship.services.validation.sources import ProjectSources

IMPORT_TO_PACKAGES: dict[str, tuple[str, ...]] = {
    "attr": ("attrs",),
    "bs4": ("beautifulsoup4",),
    "crypto": ("pycryptodome",),
    "cv2": ("opencv-python", "opencv-python-headless"),
    "dateutil": ("python-dateutil",),
    "docx": ("python-docx",),
    "dotenv": ("python-dotenv",),
    "fitz": ("pymupdf",),
    "jwt": ("pyjwt",),
    "mlflow": ("mlflow", "mlflow-skinny"),
    "openssl": ("pyopenssl",),
    "pil": ("pillow",),
    "pkg_resources": ("setuptools",),
    "serial": ("pyserial",),
    "skimage": ("scikit-image",),
    "sklearn": ("scikit-learn",),
    "umap": ("umap-learn",),
    "yaml": ("pyyaml",),
    "zmq": ("pyzmq",),
}
_OPTIONAL_ERRORS = frozenset({"ImportError", "ModuleNotFoundError", "Exception"})


def check_imports(
    sources: ProjectSources, available: frozenset[NormalizedName]
) -> list[ValidationIssue]:
    """``available``: what requirements.txt declares plus what the runtime guarantees."""
    issues: list[ValidationIssue] = []
    reported: set[str] = set()
    for source in sources.files:
        for name, line in _required_imports(source.tree):
            if name in reported or name in sys.stdlib_module_names:
                continue
            if name == "__future__" or name in sources.local_modules:
                continue
            packages = IMPORT_TO_PACKAGES.get(name.lower(), (name,))
            if any(canonicalize_name(p) in available for p in packages):
                continue
            reported.add(name)
            issues.append(
                ValidationIssue(
                    "REQUIREMENTS_IMPORT_UNDECLARED",
                    Severity.WARNING,
                    f"{name!r} is imported, but requirements.txt does not declare a "
                    "package that provides it.",
                    f"{source.relative}:{line}",
                    f"Add {' or '.join(repr(p) for p in packages)} to requirements.txt. "
                    "If it is installed today only as a dependency of another package, "
                    "it can disappear when that package changes.",
                )
            )
    return issues


def _required_imports(tree: ast.Module) -> Iterator[tuple[str, int]]:
    optional = _optional_import_nodes(tree)
    for node in ast.walk(tree):
        if id(node) in optional:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0], node.lineno


def _optional_import_nodes(tree: ast.Module) -> set[int]:
    optional: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try | ast.If) and _guards_optional_imports(node):
            for statement in node.body:
                optional.update(id(n) for n in ast.walk(statement))
    return optional


def _guards_optional_imports(node: ast.Try | ast.If) -> bool:
    if isinstance(node, ast.Try):
        return any(_catches_import_error(h) for h in node.handlers)
    return _is_type_checking(node.test)


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(t, ast.Name) and t.id in _OPTIONAL_ERRORS for t in types)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )
