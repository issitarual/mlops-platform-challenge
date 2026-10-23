import pytest
from pydantic import ValidationError

from mlship.config import DEFAULT_BASE_IMAGE, Settings

PINNED = "python:3.12-slim@sha256:" + "0" * 64


def test_defaults_are_consistent() -> None:
    settings = Settings()
    assert settings.base_image == DEFAULT_BASE_IMAGE


def test_arm_image_with_x86_lock_target_is_rejected() -> None:
    with pytest.raises(ValidationError, match="does not match MLSHIP_PLATFORM"):
        Settings(platform="linux/arm64", python_platform="x86_64-manylinux_2_28")


def test_matching_arm_target_is_accepted() -> None:
    Settings(platform="linux/arm64", python_platform="aarch64-manylinux_2_28")


def test_unsupported_platform_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not supported"):
        Settings(platform="windows/amd64")


def test_base_image_must_be_pinned_by_digest() -> None:
    with pytest.raises(ValidationError, match="pinned by digest"):
        Settings(base_image="python:3.12-slim")


def test_base_image_python_must_match_python_version() -> None:
    with pytest.raises(ValidationError, match="does not match MLSHIP_PYTHON_VERSION"):
        Settings(base_image=PINNED.replace("3.12", "3.13"))


def test_registry_prefixed_base_image_is_accepted() -> None:
    Settings(base_image="public.ecr.aws/docker/library/" + PINNED)


@pytest.mark.parametrize("field", ["train_timeout_seconds", "train_cpu_limit", "train_pids_limit"])
def test_resource_limits_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(**{field: 0})
