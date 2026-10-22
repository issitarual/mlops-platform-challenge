"""CLI settings, from environment variables (prefix ``MLSHIP_``) only."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLSHIP_")

    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "mlship"
    model_name: str = "mlship-model"
    work_dir: Path = Path(".mlship")
    python_version: str = "3.12"
    platform: str = "linux/amd64"
    train_timeout_seconds: int = 3_600
    train_memory_limit: str = "4g"
    train_cpu_limit: float = 2.0
