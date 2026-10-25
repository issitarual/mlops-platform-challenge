"""Rules for requirements.txt: PEP 508 format, supply chain and the runtime (ADR-005, ADR-021)."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import NormalizedName, canonicalize_name

from mlship.domain.validation import Severity, ValidationIssue
from mlship.services.validation.sources import ENCODING_HINT

PLATFORM_PACKAGES = frozenset({"mlship", "mlship-sdk", "mlship-runtime"})
_VCS_PREFIXES = ("git+", "hg+", "svn+", "bzr+")


@dataclass(frozen=True, slots=True)
class RequirementsReport:
    issues: tuple[ValidationIssue, ...]
    declared: frozenset[NormalizedName]
    readable: bool = True


def runtime_specifiers(runtime_requirements: Sequence[str]) -> dict[NormalizedName, SpecifierSet]:
    specifiers: dict[NormalizedName, SpecifierSet] = {}
    for raw in runtime_requirements:
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        specifiers[canonicalize_name(requirement.name)] = requirement.specifier
    return specifiers


def check_requirements(
    project_dir: Path, runtime: Mapping[NormalizedName, SpecifierSet]
) -> RequirementsReport:
    path = project_dir / "requirements.txt"
    if not path.is_file():
        missing = ValidationIssue(
            "REQUIREMENTS_MISSING",
            Severity.ERROR,
            "requirements.txt was not found.",
            str(path),
            "Create it; an empty file is valid.",
        )
        return RequirementsReport((missing,), frozenset(), readable=False)
    text = _decode(path.read_bytes())
    if text is None:
        unreadable = ValidationIssue(
            "REQUIREMENTS_ENCODING",
            Severity.ERROR,
            "requirements.txt is not UTF-8 text.",
            "requirements.txt",
            ENCODING_HINT,
        )
        return RequirementsReport((unreadable,), frozenset(), readable=False)
    issues: list[ValidationIssue] = []
    declared: set[NormalizedName] = set()
    for number, line in _logical_lines(text):
        location = f"requirements.txt:{number}"
        problem = _supply_chain_issue(line, location)
        if problem is not None:
            issues.append(problem)
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            issues.append(
                ValidationIssue(
                    "REQUIREMENTS_SYNTAX",
                    Severity.ERROR,
                    f"{line!r} is not a valid requirement: {str(exc).splitlines()[0]}.",
                    location,
                    "Use the PEP 508 form, e.g. scikit-learn==1.9.1 or pandas>=2.2,<3.",
                )
            )
            continue
        if requirement.url:
            issues.append(_direct_reference(line, location))
            continue
        name = canonicalize_name(requirement.name)
        declared.add(name)
        issues.extend(_package_issues(requirement, name, line, location, runtime))
    return RequirementsReport(tuple(issues), frozenset(declared))


def _decode(content: bytes) -> str | None:
    """UTF-8, with or without a BOM (uv accepts both); anything else is refused."""
    if content.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in content:
        return None
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def _logical_lines(text: str) -> Iterator[tuple[int, str]]:
    """Lines without comments, with ``\\`` continuations joined, numbered by their start."""
    pending: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        if not pending:
            start = number
        stripped = raw.strip()
        pending.append(stripped.removesuffix("\\").strip())
        if stripped.endswith("\\"):
            continue
        line = _without_comment(" ".join(pending))
        pending = []
        if line:
            yield start, line
    if pending and (line := _without_comment(" ".join(pending))):
        yield start, line


def _without_comment(line: str) -> str:
    line = line.strip()
    return "" if line.startswith("#") else line.split(" #", 1)[0].strip()


def _supply_chain_issue(line: str, location: str) -> ValidationIssue | None:
    if line.startswith("-"):
        return ValidationIssue(
            "REQUIREMENTS_OPTION_BLOCKED",
            Severity.ERROR,
            f"Option {line.split()[0]!r} is not allowed.",
            location,
            "Indexes, editable installs and nested files bypass the platform's package "
            "controls. List packages by name only.",
        )
    if " --" in line:
        option = line[line.index(" --") + 1 :].split()[0].split("=")[0]
        return ValidationIssue(
            "REQUIREMENTS_OPTION_BLOCKED",
            Severity.ERROR,
            f"Inline option {option!r} is not allowed.",
            location,
            "List the package and version only; mlship generates the hashes in mlship.lock.",
        )
    if line.startswith(_VCS_PREFIXES) or ("://" in line and "@" not in line):
        return _direct_reference(line, location)
    if line.startswith((".", "/", "~")) or "\\" in line:
        return ValidationIssue(
            "REQUIREMENTS_LOCAL_PATH",
            Severity.ERROR,
            f"Local path {line!r} is not allowed.",
            location,
            "Publish the package, or move the code into the project next to train.py.",
        )
    return None


def _direct_reference(line: str, location: str) -> ValidationIssue:
    return ValidationIssue(
        "REQUIREMENTS_DIRECT_REFERENCE",
        Severity.ERROR,
        f"Direct reference {line!r} is not allowed.",
        location,
        "Use a released package from the index, e.g. name==1.2.3.",
    )


def _package_issues(
    requirement: Requirement,
    name: NormalizedName,
    line: str,
    location: str,
    runtime: Mapping[NormalizedName, SpecifierSet],
) -> list[ValidationIssue]:
    if name in PLATFORM_PACKAGES:
        return [
            ValidationIssue(
                "REQUIREMENTS_PLATFORM_PACKAGE",
                Severity.ERROR,
                f"{requirement.name!r} is provided by mlship and must not be declared.",
                location,
                "Remove the line; train.py imports TrainOutput from mlship_sdk as it is.",
            )
        ]
    pinned = _exact_version(requirement.specifier)
    needed = runtime.get(name)
    if pinned is not None and needed is not None and not needed.contains(pinned, True):
        return [
            ValidationIssue(
                "REQUIREMENTS_RUNTIME_CONFLICT",
                Severity.ERROR,
                f"{line!r} conflicts with the mlship runtime, which needs "
                f"{requirement.name}{needed}.",
                location,
                f"Pin a version that satisfies {requirement.name}{needed}, or remove the "
                "pin and let mlship choose one in mlship.lock.",
            )
        ]
    if pinned is None:
        return [
            ValidationIssue(
                "REQUIREMENTS_UNPINNED",
                Severity.WARNING,
                f"{requirement.name!r} has no exact version.",
                location,
                "mlship pins it in mlship.lock; pin it here too to make upgrades explicit.",
            )
        ]
    return []


def _exact_version(specifier: SpecifierSet) -> str | None:
    exact = [s for s in specifier if s.operator in ("==", "===") and "*" not in s.version]
    return exact[0].version if len(exact) == 1 else None
