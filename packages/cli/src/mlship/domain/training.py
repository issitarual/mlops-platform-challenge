"""Training job description and result, independent of the container technology."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mlship.domain.environment import ImageRef


@dataclass(frozen=True, slots=True)
class TrainingJobSpec:
    image: ImageRef
    project_dir: Path  # mounted read-only
    data_dir: Path  # mounted read-only
    output_dir: Path  # the only writable mount
    params: Mapping[str, str]
    timeout_seconds: int
    memory_limit: str
    cpu_limit: float


@dataclass(frozen=True, slots=True)
class TrainingJobResult:
    exit_code: int
    timed_out: bool
    logs_tail: str
