"""Serving configuration, read from environment variables only (prefix ``MLSHIP_SERVING_``).

The host sets every value when it starts the container (``mlship serve``); nothing in
the model directory can change them.
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLSHIP_SERVING_")

    model_dir: Path = Path("/opt/mlship/model")
    model_name: str = "model"
    model_version: str = "0"
    model_sha256: str = ""
    allow_pickle: bool = False
    skops_trusted_types: list[str] = []

    max_batch_rows: int = Field(default=1_000, gt=0)
    max_payload_bytes: int = Field(default=1_000_000, gt=0)
    request_timeout_seconds: float = Field(default=10.0, gt=0)
    api_key: SecretStr | None = None
    prediction_log_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    prediction_log_exclude_columns: list[str] = []
