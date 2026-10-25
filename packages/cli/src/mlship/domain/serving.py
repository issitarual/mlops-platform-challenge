"""Serving one model version: what goes into its image and how its container runs."""

from collections.abc import Mapping
from dataclasses import dataclass

from mlship.domain.environment import ImageRef
from mlship.domain.runs import ModelVersion

SERVING_PORT = 8080


@dataclass(frozen=True, slots=True)
class ServingImageSpec:
    """Everything baked into the serving image (ADR-007): base, model, load policy."""

    base: ImageRef
    model: ModelVersion
    allow_pickle: bool
    skops_trusted_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ServingLimits:
    cpus: float
    memory: str
    pids: int


@dataclass(frozen=True, slots=True)
class ServingJobSpec:
    image: ImageRef
    model: ModelVersion
    host_port: int
    workers: int
    environment: Mapping[str, str]
    limits: ServingLimits


@dataclass(frozen=True, slots=True)
class PreparedServing:
    model: ModelVersion
    alias: str
    environment_image: ImageRef
    environment_reused: bool
    serving_image: ImageRef
