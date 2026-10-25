"""``mlship validate``: the static rules, the lock, then the real resolution (ADR-021)."""

from pathlib import Path

from mlship.domain.environment import LockState, ResolvedLock
from mlship.domain.errors import ProjectInvalidError
from mlship.domain.validation import ResolutionCheck, Severity, ValidationIssue, ValidationReport
from mlship.services.environment_service import EnvironmentService
from mlship.services.validation.validator import ProjectValidator

_LOCK_HINT = (
    "Run `mlship lock` and commit mlship.lock. Without a current lock, the next train "
    "resolves again and may pick newer versions than the ones you tested."
)
_LOCK_ISSUES = {
    LockState.MISSING: ("LOCK_MISSING", "mlship.lock does not exist."),
    LockState.STALE: (
        "LOCK_STALE",
        "mlship.lock does not match requirements.txt or the mlship runtime.",
    ),
    LockState.MODIFIED: ("LOCK_MODIFIED", "mlship.lock was edited by hand."),
}


class ValidationService:
    def __init__(self, validator: ProjectValidator, environment: EnvironmentService) -> None:
        self._validator = validator
        self._environment = environment

    def validate(
        self, project_dir: Path, data_dir: Path | None, *, offline: bool
    ) -> tuple[ValidationReport, ResolutionCheck]:
        """A current mlship.lock already proves the resolution, so it needs no network.

        Otherwise the resolution runs unless ``offline``. Both are skipped when
        requirements.txt is missing or has errors, which uv would only repeat.
        """
        report = self._validator.validate(project_dir, data_dir)
        blocked = any(i.code.startswith(("REQUIREMENTS_", "PROJECT_")) for i in report.errors)
        if blocked or not (project_dir / "requirements.txt").is_file():
            return report, ResolutionCheck.SKIPPED
        state = self._environment.lock_state(project_dir)
        issues = list(report.issues)
        if state in _LOCK_ISSUES:
            code, message = _LOCK_ISSUES[state]
            issues.append(
                ValidationIssue(code, Severity.WARNING, message, "mlship.lock", _LOCK_HINT)
            )
        if state is LockState.CURRENT:
            return ValidationReport(tuple(issues)), ResolutionCheck.PASSED
        if offline:
            return ValidationReport(tuple(issues)), ResolutionCheck.OFFLINE
        problem = self._environment.check_resolution(project_dir)
        if problem is None:
            return ValidationReport(tuple(issues)), ResolutionCheck.PASSED
        return ValidationReport((*issues, problem)), ResolutionCheck.FAILED

    def lock(self, project_dir: Path) -> ResolvedLock:
        """Write (or keep) mlship.lock; requirements.txt is validated first.

        Only errors that concern the dependencies block: the lock does not depend on
        train.py, and the resolution must never see a blocked option.
        """
        report = self._validator.validate(project_dir, None)
        blocking = tuple(
            i for i in report.errors if i.code.startswith(("REQUIREMENTS_", "PROJECT_"))
        )
        if blocking:
            raise ProjectInvalidError(ValidationReport(blocking))
        return self._environment.lock(project_dir)
