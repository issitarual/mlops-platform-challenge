"""``run.json``: everything the host decided for this run (ADR-018)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SerializationPolicy:
    allow_pickle: bool
    skops_trusted_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunConfig:
    project_dir: Path
    data_dir: Path
    output_dir: Path
    params: dict[str, str]
    seed: int
    serialization: SerializationPolicy


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid run.json: {message}")


def parse_config(raw: dict[str, Any]) -> RunConfig:
    _require(raw.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    params = raw.get("params")
    if not isinstance(params, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in params.items()
    ):
        raise ValueError("invalid run.json: params must map strings to strings")
    seed = raw.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("invalid run.json: seed must be an integer")
    serialization = raw.get("serialization")
    if not isinstance(serialization, dict):
        raise ValueError("invalid run.json: serialization must be an object")
    allow_pickle = serialization.get("allow_pickle")
    trusted = serialization.get("skops_trusted_types")
    if not isinstance(allow_pickle, bool):
        raise ValueError("invalid run.json: allow_pickle must be a boolean")
    if not isinstance(trusted, list) or not all(isinstance(t, str) for t in trusted):
        raise ValueError("invalid run.json: skops_trusted_types must be a list of strings")
    directories: dict[str, Path] = {}
    for key in ("project_dir", "data_dir", "output_dir"):
        value = raw.get(key)
        if not isinstance(value, str):
            raise ValueError(f"invalid run.json: {key} must be a string")
        directories[key] = Path(value)
    return RunConfig(
        project_dir=directories["project_dir"],
        data_dir=directories["data_dir"],
        output_dir=directories["output_dir"],
        params=dict(params),
        seed=seed,
        serialization=SerializationPolicy(allow_pickle, tuple(trusted)),
    )


def load_config(path: Path) -> RunConfig:
    return parse_config(json.loads(path.read_text(encoding="utf-8")))
