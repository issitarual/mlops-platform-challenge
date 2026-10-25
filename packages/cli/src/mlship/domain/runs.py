"""Lineage: what was run, with what, and what came out (ADR-015, ADR-019)."""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from mlship.domain.errors import MlshipError

MANIFEST_SCHEMA_VERSION = 1
_MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_ALIAS = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
RESERVED_ALIASES = frozenset({"latest"})
MODEL_VERSION_TAG = "mlship.model_version"  # on the run: which version it produced


class RunStatus(StrEnum):
    """Same names as MLflow run statuses."""

    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


def tree_digest(files: Mapping[str, str]) -> str:
    """One SHA-256 for a set of files: hash of the sorted ``{path: sha256}`` map."""
    canonical = json.dumps(dict(sorted(files.items())), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_name_from_directory(directory_name: str) -> str:
    """Registry name derived from a project folder: lowercase, safe characters only."""
    name = re.sub(r"[^a-z0-9._-]+", "-", directory_name.lower()).strip("-._")
    return name[:100]


def is_valid_model_name(name: str) -> bool:
    return bool(_MODEL_NAME.match(name))


def resolve_model_name(project_dir: Path, override: str | None) -> str:
    """``--model-name`` if given, else the project folder name (one experiment per model)."""
    name = override if override else model_name_from_directory(project_dir.resolve().name)
    if not is_valid_model_name(name):
        raise MlshipError(
            f"{name!r} is not a valid model name.",
            hint="Use --model-name with lowercase letters, digits, '.', '-' or '_' "
            "(up to 100 characters), e.g. --model-name california-housing.",
        )
    return name


def is_valid_alias(alias: str) -> bool:
    return bool(_ALIAS.match(alias)) and alias not in RESERVED_ALIASES


@dataclass(frozen=True, slots=True)
class ToolInfo:
    version: str
    git_sha: str  # "unknown" when not installed from a git checkout
    git_dirty: bool | None  # None when unknown


@dataclass(frozen=True, slots=True)
class Manifest:
    """Every identifier that went into a run and came out of it. Serialized to JSON."""

    run_id: str
    model_name: str
    status: str  # "succeeded" or "failed"
    created_at: str  # ISO 8601, UTC
    tool: ToolInfo
    code_files: Mapping[str, str]
    code_skipped: Mapping[str, int]  # too large to hash: path -> size in bytes
    lock_sha256: str
    lock_inputs_sha256: str
    environment: Mapping[str, Any]
    data_location: str
    data_files: Mapping[str, str]
    params: Mapping[str, str]
    seed: int
    limits: Mapping[str, Any]
    metrics: Mapping[str, float]
    model_serialization: str | None
    model_files: Mapping[str, str]
    errors: tuple[Mapping[str, str], ...] = ()
    warnings: tuple[Mapping[str, str], ...] = ()

    @property
    def code_sha256(self) -> str:
        return tree_digest(self.code_files)

    @property
    def data_sha256(self) -> str:
        return tree_digest(self.data_files)

    @property
    def model_sha256(self) -> str | None:
        return tree_digest(self.model_files) if self.model_files else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "model_name": self.model_name,
            "status": self.status,
            "created_at": self.created_at,
            "tool": {
                "version": self.tool.version,
                "git_sha": self.tool.git_sha,
                "git_dirty": self.tool.git_dirty,
            },
            "code": {
                "sha256": self.code_sha256,
                "files": dict(self.code_files),
                "skipped_large_files": dict(self.code_skipped),
            },
            "dependencies": {
                "lock_sha256": self.lock_sha256,
                "lock_inputs_sha256": self.lock_inputs_sha256,
            },
            "environment": dict(self.environment),
            "data": {
                "location": self.data_location,
                "sha256": self.data_sha256,
                "files": dict(self.data_files),
            },
            "params": dict(self.params),
            "seed": self.seed,
            "limits": dict(self.limits),
            "metrics": dict(self.metrics),
            "model": None
            if not self.model_files
            else {
                "sha256": self.model_sha256,
                "serialization": self.model_serialization,
                "files": dict(self.model_files),
            },
            "validation": {"errors": list(self.errors), "warnings": list(self.warnings)},
        }


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    created_at: datetime
    status: RunStatus
    metrics: Mapping[str, float]
    model_version: int | None


@dataclass(frozen=True, slots=True)
class ModelVersion:
    name: str
    version: int
    run_id: str
    model_sha256: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Promotion:
    alias: str
    current: ModelVersion
    previous: ModelVersion | None  # who held the alias before; None if nobody

    @property
    def changed(self) -> bool:
        return self.previous is None or self.previous.version != self.current.version
