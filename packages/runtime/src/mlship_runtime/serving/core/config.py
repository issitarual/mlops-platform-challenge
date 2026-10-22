"""Serving configuration, read from environment variables only."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingSettings(BaseSettings):
    """Settings for one serving instance (prefix ``MLSHIP_SERVING_``)."""

    model_config = SettingsConfigDict(env_prefix="MLSHIP_SERVING_")

    model_dir: Path = Path("/opt/mlship/model")
    max_batch_rows: int = 1_000
    max_payload_bytes: int = 1_000_000
