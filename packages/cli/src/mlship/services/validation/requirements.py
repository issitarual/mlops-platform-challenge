"""Rules for requirements.txt: format and supply-chain options (ADR-005)."""

from pathlib import Path

from mlship.domain.validation import Severity, ValidationIssue

_PINS = ("==", "===")


def check_requirements(project_dir: Path) -> list[ValidationIssue]:
    path = project_dir / "requirements.txt"
    if not path.is_file():
        return [
            ValidationIssue(
                "REQUIREMENTS_MISSING",
                Severity.ERROR,
                "requirements.txt was not found.",
                str(path),
                "Create it; an empty file is valid.",
            )
        ]
    issues: list[ValidationIssue] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        location = f"requirements.txt:{number}"
        if line.startswith("-"):
            issues.append(
                ValidationIssue(
                    "REQUIREMENTS_OPTION_BLOCKED",
                    Severity.ERROR,
                    f"Option {line.split()[0]!r} is not allowed.",
                    location,
                    "Indexes, editable installs and nested files bypass the platform's "
                    "package controls. List packages by name only.",
                )
            )
        elif "://" in line or line.startswith(("git+", "hg+", "svn+", "bzr+")) or " @ " in line:
            issues.append(
                ValidationIssue(
                    "REQUIREMENTS_DIRECT_REFERENCE",
                    Severity.ERROR,
                    f"Direct reference {line!r} is not allowed.",
                    location,
                    "Use a released package from the index, e.g. name==1.2.3.",
                )
            )
        elif line.startswith((".", "/", "~")) or "\\" in line:
            issues.append(
                ValidationIssue(
                    "REQUIREMENTS_LOCAL_PATH",
                    Severity.ERROR,
                    f"Local path {line!r} is not allowed.",
                    location,
                    "Publish the package, or move the code into the project next to train.py.",
                )
            )
        elif not any(pin in line for pin in _PINS):
            issues.append(
                ValidationIssue(
                    "REQUIREMENTS_UNPINNED",
                    Severity.WARNING,
                    f"{line!r} has no exact version.",
                    location,
                    "mlship pins it in mlship.lock; pin it here too to make upgrades explicit.",
                )
            )
    return issues
