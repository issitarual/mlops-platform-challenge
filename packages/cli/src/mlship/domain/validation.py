"""Validation results. Every check reports all issues; nothing fails fast."""

from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    ERROR = "error"  # violates the contract: blocks training
    WARNING = "warning"  # bad practice: blocks only with --strict


class ResolutionCheck(StrEnum):
    """Whether requirements.txt was resolved together with the runtime."""

    PASSED = "passed"
    FAILED = "failed"
    OFFLINE = "offline"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str  # stable identifier, e.g. "TRAIN_SIGNATURE"
    severity: Severity
    message: str  # what is wrong
    location: str  # where: "train.py:12" or "requirements.txt:3"
    hint: str  # how to fix it


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.WARNING)

    def blocks(self, *, strict: bool) -> bool:
        """True when training must not start."""
        return bool(self.errors) or (strict and bool(self.warnings))
