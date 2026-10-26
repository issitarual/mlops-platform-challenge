"""What a run records about itself: the manifest and the tags derived from it (ADR-019)."""

import json
from collections.abc import Mapping
from pathlib import Path

from mlship.domain.runs import Manifest
from mlship.domain.validation import ValidationIssue

CODE_EXCLUDED_DIRS = frozenset(
    {
        ".mlship",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)
_TAG_PREFIX = "mlship."


CODE_MAX_FILE_BYTES = 1024 * 1024
_HASHED_ELSEWHERE = frozenset({"mlship.lock"})  # recorded under "dependencies"


class CodeSelector:
    """Which project files count as code (ADR-019): every file, except

    - hidden files and folders (``.env``, ``.git``...): never recorded, not even by name;
    - the data folder, when it lives inside the project (hashed as data);
    - files above ``max_file_bytes``: listed in ``skipped`` with their size, not hashed.

    Config files read by ``train.py`` (YAML, JSON...) change the model, so they are code.
    """

    def __init__(
        self, project_dir: Path, data_dir: Path, max_file_bytes: int = CODE_MAX_FILE_BYTES
    ) -> None:
        self._project_dir = project_dir
        self._max_file_bytes = max_file_bytes
        self._data_prefix: str | None = None
        if data_dir.resolve().is_relative_to(project_dir.resolve()):
            self._data_prefix = data_dir.resolve().relative_to(project_dir.resolve()).as_posix()
        self.skipped: dict[str, int] = {}

    def __call__(self, relative_path: str) -> bool:
        parts = relative_path.split("/")
        if any(part.startswith(".") for part in parts) or relative_path in _HASHED_ELSEWHERE:
            return False
        prefix = self._data_prefix
        if prefix is not None and (prefix == "." or relative_path.startswith(f"{prefix}/")):
            return False
        size = (self._project_dir / relative_path).stat().st_size
        if size > self._max_file_bytes:
            self.skipped[relative_path] = size
            return False
        return True


def data_location(project_dir: Path, data_dir: Path) -> str:
    """Relative to the project when inside it, so no local user path is recorded."""
    resolved = data_dir.resolve()
    if resolved.is_relative_to(project_dir.resolve()):
        return resolved.relative_to(project_dir.resolve()).as_posix()
    return str(resolved)


def issue_dicts(issues: tuple[ValidationIssue, ...]) -> tuple[Mapping[str, str], ...]:
    return tuple(
        {"code": i.code, "location": i.location, "message": i.message, "hint": i.hint}
        for i in issues
    )


def write_manifest(manifest: Manifest, path: Path) -> None:
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def manifest_tags(manifest: Manifest) -> dict[str, str]:
    """The identifiers worth searching by in the MLflow UI; the full record is the artifact."""
    values: dict[str, str | None] = {
        "status": manifest.status,
        "model_name": manifest.model_name,
        "code_sha256": manifest.code_sha256,
        "data_sha256": manifest.data_sha256,
        "lock_sha256": manifest.lock_sha256,
        "environment_key": str(manifest.environment.get("environment_key", "")),
        "image_tag": str(manifest.environment.get("image_tag", "")),
        "tool_version": manifest.tool.version,
        "tool_git_sha": manifest.tool.git_sha,
        "tool_git_dirty": "unknown"
        if manifest.tool.git_dirty is None
        else str(manifest.tool.git_dirty).lower(),
        "model_sha256": manifest.model_sha256,
        "serialization": manifest.model_serialization,
        "allow_pickle": str(manifest.allow_pickle).lower(),
        "errors": ",".join(e["code"] for e in manifest.errors) or None,
    }
    return {f"{_TAG_PREFIX}{k}": v for k, v in values.items() if v}
