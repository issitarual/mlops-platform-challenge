"""Rules for requirements.txt (parsing and supply-chain options)."""

from pathlib import Path

from mlship.domain.validation import ValidationIssue


def check_requirements(project_dir: Path) -> list[ValidationIssue]:
    raise NotImplementedError("E5")
