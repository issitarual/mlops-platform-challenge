"""``RuntimePackager`` for the mlship workspace (packages/runtime and packages/sdk).

Reads package metadata only; the CLI never imports runtime code (see the
architecture test). In production the runtime is published to a private index
instead (ADR-005).
"""

import hashlib
import json
import subprocess
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from mlship.adapters.hashing import sha256_tree
from mlship.adapters.process import Runner, find_executable, run, tail
from mlship.domain.errors import EnvironmentBuildError, MlshipError

RUNTIME_DIST = "mlship-runtime"
SDK_DIST = "mlship-sdk"
_UV_HINT = "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
_CACHE_DIRS = frozenset({"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"})


def _source_dir(distribution: str) -> Path:
    """Directory of a package installed from local source (editable workspace member)."""
    try:
        raw = metadata.distribution(distribution).read_text("direct_url.json")
    except metadata.PackageNotFoundError as exc:
        raise MlshipError(
            f"Package {distribution} is not installed.",
            hint="Run `uv sync --all-packages` at the repository root.",
        ) from exc
    url = json.loads(raw or "{}").get("url", "")
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise MlshipError(
            f"{distribution} was not installed from local source.",
            hint="Run mlship from the repository with `uv sync --all-packages`.",
        )
    return Path(url2pathname(unquote(parsed.path)))


class WorkspaceRuntimePackager:
    def __init__(self, runner: Runner = subprocess.run) -> None:
        self._runner = runner

    def requirements(self) -> list[str]:
        """Runtime dependencies, minus the sdk (installed from a local wheel)."""
        requires = metadata.requires(RUNTIME_DIST) or []
        return sorted(r for r in requires if not r.replace("_", "-").startswith(SDK_DIST))

    def source_sha256(self) -> str:
        """One hash over the source of runtime and sdk (caches excluded)."""
        trees = {
            dist: sha256_tree(_source_dir(dist), exclude_dirs=_CACHE_DIRS)
            for dist in (RUNTIME_DIST, SDK_DIST)
        }
        canonical = json.dumps(trees, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def build_wheels(self, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        uv = find_executable("uv", _UV_HINT)
        for dist in (SDK_DIST, RUNTIME_DIST):
            source = _source_dir(dist)
            command = [uv, "build", "--wheel", "--quiet", "--out-dir", str(output_dir), str(source)]
            result = run(command, source, self._runner)
            if result.returncode != 0:
                raise EnvironmentBuildError(
                    f"Could not build the {dist} wheel.", details=tail(result.stderr)
                )
        return sorted(output_dir.glob("*.whl"))
