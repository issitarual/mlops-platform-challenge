"""CLI settings, from environment variables (prefix ``MLSHIP_``) only."""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BASE_IMAGE = (
    "python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704"
)

_CPU_BY_DOCKER_PLATFORM = {"linux/amd64": "x86_64", "linux/arm64": "aarch64"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLSHIP_")

    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "mlship"
    model_name: str = "mlship-model"
    work_dir: Path = Path(".mlship")
    python_version: str = "3.12"
    platform: str = "linux/amd64"
    python_platform: str = "x86_64-manylinux_2_28"
    base_image: str = DEFAULT_BASE_IMAGE
    lock_filename: str = "mlship.lock"
    image_repository: str = "mlship-env"
    train_timeout_seconds: int = Field(default=3_600, gt=0)
    train_memory_limit: str = "4g"
    train_cpu_limit: float = Field(default=2.0, gt=0)
    train_pids_limit: int = Field(default=256, gt=0)
    train_tmpfs_size: str = "1g"
    skops_trusted_types: list[str] = [
        "sklearn.tree._tree.Tree",
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    ]
    allow_pickle_enabled: bool = True

    @model_validator(mode="after")
    def _target_is_consistent(self) -> Self:
        """Fail at startup, not in the middle of a build, when the target is incoherent."""
        problems: list[str] = []
        cpu = _CPU_BY_DOCKER_PLATFORM.get(self.platform)
        if cpu is None:
            supported = ", ".join(sorted(_CPU_BY_DOCKER_PLATFORM))
            problems.append(f"MLSHIP_PLATFORM={self.platform!r} is not supported ({supported}).")
        elif not self.python_platform.startswith(f"{cpu}-"):
            problems.append(
                f"MLSHIP_PYTHON_PLATFORM={self.python_platform!r} does not match "
                f"MLSHIP_PLATFORM={self.platform!r} (expected a {cpu}-... platform)."
            )
        image, _, digest = self.base_image.partition("@sha256:")
        if len(digest) != 64:
            problems.append(
                f"MLSHIP_BASE_IMAGE={self.base_image!r} must be pinned by digest "
                "(...@sha256:<64 hex>)."
            )
        tag = image.rsplit("/", 1)[-1].partition(":")[2]
        if not tag.startswith(self.python_version):
            problems.append(
                f"MLSHIP_BASE_IMAGE tag {tag!r} does not match "
                f"MLSHIP_PYTHON_VERSION={self.python_version!r}."
            )
        if problems:
            raise ValueError(" ".join(problems))
        return self
