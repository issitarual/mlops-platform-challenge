"""Errors shown to the user: what went wrong, how to fix it, and the raw details."""


class MlshipError(Exception):
    """Base error. ``hint`` says how to fix it; ``details`` carries tool output, if any."""

    def __init__(self, message: str, *, hint: str = "", details: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details


class ToolNotFoundError(MlshipError):
    """A required external tool (uv, Docker) is missing or not running."""


class LockResolutionError(MlshipError):
    """Dependencies could not be resolved into a lock."""


class EnvironmentBuildError(MlshipError):
    """The environment image could not be built."""
