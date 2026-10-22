"""Rules for train.py, via ``ast`` only (the file is parsed, never imported)."""

from pathlib import Path

from mlship.domain.validation import ValidationIssue


def check_train_script(project_dir: Path) -> list[ValidationIssue]:
    raise NotImplementedError("E5")
