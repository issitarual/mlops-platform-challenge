"""Rules for the data directory."""

from pathlib import Path

from mlship.domain.validation import ValidationIssue


def check_data_dir(data_dir: Path) -> list[ValidationIssue]:
    raise NotImplementedError("E5")
