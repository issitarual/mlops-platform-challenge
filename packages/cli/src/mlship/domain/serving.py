"""Serving one model version: what goes into its image and how its container runs."""

from collections.abc import Mapping
from dataclasses import dataclass

from mlship.domain.environment import ImageRef
from mlship.domain.runs import ModelVersion

SERVING_PORT = 8080
SERVING_DEFAULT_PORT = 8080
# Created by `docker compose --profile monitoring`; shared only by the API and Prometheus.
MONITORING_NETWORK = "mlship-serving"
MONITORING_ALIAS = "mlship-api"


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


THREAD_LIMIT_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# Native libraries (OpenMP, BLAS) size their pools by the host's CPUs, not by the container's
# quota, and concurrent predictions each open a pool: the container is throttled and latency
# goes from milliseconds to seconds. One thread per prediction; throughput scales with
# workers and replicas, as online serving usually does (ADR-020).
SINGLE_THREAD_ENVIRONMENT: Mapping[str, str] = dict.fromkeys(THREAD_LIMIT_VARIABLES, "1")


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
