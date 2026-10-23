"""``LockResolver`` that co-resolves user requirements with the runtime using ``uv``."""

import subprocess
from collections.abc import Sequence
from pathlib import Path

from mlship.adapters.process import Runner, find_executable, run, tail
from mlship.domain.errors import LockResolutionError

_UV_HINT = "Install uv: https://docs.astral.sh/uv/getting-started/installation/"


class UvLockResolver:
    def __init__(
        self,
        python_version: str,
        python_platform: str,
        runner: Runner = subprocess.run,
    ) -> None:
        self._python_version = python_version
        self._python_platform = python_platform
        self._runner = runner

    def command(self, uv: str, requirement_files: Sequence[Path], output: Path) -> list[str]:
        return [
            uv,
            "pip",
            "compile",
            *(f.name for f in requirement_files),
            "--python-version",
            self._python_version,
            "--python-platform",
            self._python_platform,
            "--generate-hashes",
            "--no-build",
            "--no-header",
            "--quiet",
            "--output-file",
            output.name,
        ]

    def resolve(self, requirement_files: Sequence[Path], output: Path) -> None:
        workdir = output.parent
        if any(f.parent != workdir for f in requirement_files):
            raise ValueError("requirement files and output must share one directory")
        uv = find_executable("uv", _UV_HINT)
        result = run(self.command(uv, requirement_files, output), workdir, self._runner)
        if result.returncode != 0:
            raise LockResolutionError(
                "Could not resolve your requirements together with the mlship runtime.",
                hint=(
                    "Check package names and version pins in requirements.txt. A conflict "
                    "with the runtime means a pin is incompatible with what mlship needs."
                ),
                details=tail(result.stderr),
            )
