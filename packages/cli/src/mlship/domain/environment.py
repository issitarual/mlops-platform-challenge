"""Reproducible environment: one lock -> one image."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResolvedLock:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class ImageRef:
    tag: str
    digest: str
