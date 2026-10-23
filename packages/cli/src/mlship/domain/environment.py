"""Reproducible environment: one set of build inputs -> one image."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResolvedLock:
    path: Path
    sha256: str  # hash of the whole lock file
    inputs_sha256: str  # hash of what was resolved (requirements, runtime, target)
    reused: bool  # True when an up-to-date lock already existed
    discarded_modified: bool = False  # an existing lock failed its integrity check


@dataclass(frozen=True, slots=True)
class ImageRef:
    tag: str
    digest: str


@dataclass(frozen=True, slots=True)
class EnvironmentInputs:
    """Every input that changes the image. Same inputs -> same key -> same image."""

    lock_sha256: str
    runtime_source_sha256: str
    dockerfile_sha256: str
    platform: str

    def key(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedEnvironment:
    image: ImageRef
    lock: ResolvedLock
    inputs: EnvironmentInputs
    cache_hit: bool
