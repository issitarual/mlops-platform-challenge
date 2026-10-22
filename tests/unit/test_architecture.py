"""Dependency rules between packages and layers.

A uv workspace shares one dev environment, so nothing stops a wrong import at
runtime. This test is the guard. It reports every violation at once.
"""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SDK = ROOT / "packages/sdk/src/mlship_sdk"
RUNTIME = ROOT / "packages/runtime/src/mlship_runtime"
CLI = ROOT / "packages/cli/src/mlship"
STDLIB = frozenset(sys.stdlib_module_names)


@dataclass(frozen=True)
class Rule:
    name: str
    scope: Path
    forbidden: tuple[str, ...] = ()
    stdlib_only_except: tuple[str, ...] | None = None  # None = no stdlib-only rule


RULES = [
    Rule("sdk has no dependencies", SDK, stdlib_only_except=("mlship_sdk",)),
    Rule("domain is pure", CLI / "domain", stdlib_only_except=("mlship.domain",)),
    Rule(
        "cli services depend only on domain and ports",
        CLI / "services",
        forbidden=(
            "docker",
            "mlflow",
            "typer",
            "rich",
            "fastapi",
            "mlship.adapters",
            "mlship.repositories",
            "mlship.controllers",
            "mlship.bootstrap",
        ),
    ),
    Rule(
        "controllers get services from bootstrap only",
        CLI / "controllers",
        forbidden=("docker", "mlflow", "mlship.adapters", "mlship.repositories"),
    ),
    Rule(
        "adapters do not reach up the layers",
        CLI / "adapters",
        forbidden=("typer", "mlship.controllers", "mlship.bootstrap"),
    ),
    Rule(
        "repositories do not reach up the layers",
        CLI / "repositories",
        forbidden=("typer", "mlship.controllers", "mlship.bootstrap"),
    ),
    Rule("cli never imports the runtime", CLI, forbidden=("mlship_runtime",)),
    Rule("runtime never imports the cli", RUNTIME, forbidden=("mlship", "docker", "typer")),
    Rule(
        "serving services depend only on ports",
        RUNTIME / "serving/services",
        forbidden=(
            "fastapi",
            "mlflow",
            "mlship_runtime.serving.api",
            "mlship_runtime.serving.engines",
        ),
    ),
]


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative imports are banned by ruff; flag them here too
                found.append((node.lineno, "." * node.level + (node.module or "")))
            elif node.module:
                found.append((node.lineno, node.module))
    return found


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _violations(rule: Rule) -> list[str]:
    problems: list[str] = []
    for path in sorted(rule.scope.rglob("*.py")):
        for line, module in _imports(path):
            where = f"{path.relative_to(ROOT).as_posix()}:{line}"
            if module.startswith("."):
                problems.append(f"{where} relative import '{module}'")
                continue
            if any(_matches(module, p) for p in rule.forbidden):
                problems.append(f"{where} imports '{module}'")
            if rule.stdlib_only_except is not None:
                allowed = module.split(".")[0] in STDLIB or any(
                    _matches(module, p) for p in rule.stdlib_only_except
                )
                if not allowed:
                    problems.append(f"{where} imports non-stdlib '{module}'")
    return problems


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.name)
def test_dependency_rule(rule: Rule) -> None:
    assert list(rule.scope.rglob("*.py")), f"scope is empty or moved: {rule.scope}"
    problems = _violations(rule)
    assert not problems, f"{rule.name}:\n" + "\n".join(problems)
