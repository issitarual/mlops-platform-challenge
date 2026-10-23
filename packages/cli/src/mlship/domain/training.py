"""Training job description and result, independent of the container technology."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from mlship.domain.environment import PreparedEnvironment
from mlship.domain.validation import ValidationIssue

PROJECT_MOUNT = "/mlship/project"
DATA_MOUNT = "/mlship/data"
CONFIG_MOUNT = "/mlship/run.json"
OUTPUT_MOUNT = "/output"
CONTAINER_TMP = "/tmp"
RUN_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    cpus: float
    memory: str  # Docker format, e.g. "4g"; swap is disabled (same value)
    pids: int
    timeout_seconds: int
    tmpfs_size: str  # size of the writable /tmp, e.g. "1g"


@dataclass(frozen=True, slots=True)
class TrainingJobSpec:
    image_tag: str
    project_dir: Path  # mounted read-only
    data_dir: Path  # mounted read-only
    config_file: Path  # run.json, mounted read-only
    output_dir: Path  # the only writable mount
    environment: Mapping[str, str]
    limits: ResourceLimits
    run_id: str


@dataclass(frozen=True, slots=True)
class TrainingJobResult:
    exit_code: int
    timed_out: bool
    oom_killed: bool
    logs_tail: str


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    run_id: str
    run_dir: Path
    succeeded: bool
    issues: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]
    metrics: Mapping[str, float]
    serialization: str | None
    environment: PreparedEnvironment
    job: TrainingJobResult
    params: Mapping[str, str] = field(default_factory=dict)
