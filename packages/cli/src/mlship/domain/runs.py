"""Lineage: what was run, with what, and what came out."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Manifest:
    """Everything needed to trace and re-run a training run (fields refined in E3)."""

    tool_version: str
    tool_git_sha: str
    code_sha256: str
    lock_sha256: str
    image_digest: str
    data_sha256: Mapping[str, str]  # file -> hash
    params: Mapping[str, str]
    metrics: Mapping[str, float]
    model_sha256: str
    validation_warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    created_at: datetime
    metrics: Mapping[str, float]
    model_version: int | None


@dataclass(frozen=True, slots=True)
class ModelVersion:
    name: str
    version: int
    run_id: str
    model_sha256: str
