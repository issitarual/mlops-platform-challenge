"""Errors shown to the user: what went wrong, how to fix it, and the raw details."""

from mlship.domain.validation import ValidationReport


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


class ContainerRunError(MlshipError):
    """Docker refused to create or start a container."""


class TrackingUnavailableError(MlshipError):
    """The MLflow tracking server cannot be reached."""


class TrackingError(MlshipError):
    """MLflow rejected or failed an operation."""


class ModelNotFoundError(MlshipError):
    """A model, version or alias does not exist in the registry."""


class UntrustedArtifactError(MlshipError):
    """Training output contains something other than regular files and directories."""


class ArtifactTooLargeError(UntrustedArtifactError):
    """Training output is larger, or has more files, than the platform allows."""


class ProjectInvalidError(MlshipError):
    """The project breaks the contract; ``report`` lists every issue at once."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__(f"The project has {len(report.errors)} error(s).")
        self.report = report
