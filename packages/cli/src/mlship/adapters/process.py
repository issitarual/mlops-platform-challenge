"""Running external tools: argument lists only, never a shell."""

import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from mlship.domain.errors import ToolNotFoundError

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def find_executable(name: str, hint: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ToolNotFoundError(f"`{name}` was not found on PATH.", hint=hint)
    return path


def run(command: Sequence[str], cwd: Path, runner: Runner) -> "subprocess.CompletedProcess[str]":
    # Fixed executable resolved with shutil.which and an argument list: no shell involved.
    return runner(list(command), cwd=cwd, capture_output=True, text=True, check=False)


def tail(text: str, lines: int = 30) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])
