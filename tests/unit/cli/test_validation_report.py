from mlship.domain.validation import Severity, ValidationIssue, ValidationReport


def _issue(severity: Severity) -> ValidationIssue:
    return ValidationIssue("CODE", severity, "message", "train.py:1", "hint")


def test_errors_always_block() -> None:
    report = ValidationReport((_issue(Severity.ERROR),))
    assert report.blocks(strict=False)


def test_warnings_block_only_in_strict_mode() -> None:
    report = ValidationReport((_issue(Severity.WARNING),))
    assert not report.blocks(strict=False)
    assert report.blocks(strict=True)


def test_report_keeps_every_issue() -> None:
    report = ValidationReport((_issue(Severity.ERROR), _issue(Severity.WARNING)))
    assert len(report.errors) == 1
    assert len(report.warnings) == 1
