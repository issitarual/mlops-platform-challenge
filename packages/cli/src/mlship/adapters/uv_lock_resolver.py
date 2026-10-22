"""``LockResolver`` that co-resolves user requirements with the runtime using ``uv``."""

from pathlib import Path

from mlship.domain.environment import ResolvedLock


class UvLockResolver:
    def __init__(self, python_version: str, platform: str) -> None:
        self._python_version = python_version
        self._platform = platform

    def resolve(self, requirements_file: Path, output_dir: Path) -> ResolvedLock:
        """E1: run ``uv`` (argument list, never ``shell=True``) and hash the lock."""
        raise NotImplementedError("E1")
