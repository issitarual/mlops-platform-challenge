"""Rules for the data directory."""

from pathlib import Path

from mlship.domain.validation import Severity, ValidationIssue


def check_data_dir(data_dir: Path) -> list[ValidationIssue]:
    if not data_dir.is_dir():
        return [
            ValidationIssue(
                "DATA_DIR_MISSING",
                Severity.ERROR,
                f"{data_dir} is not a directory.",
                str(data_dir),
                "Pass the folder that holds the training data with --data.",
            )
        ]
    if not any(p.is_file() for p in data_dir.rglob("*")):
        return [
            ValidationIssue(
                "DATA_DIR_EMPTY",
                Severity.ERROR,
                f"{data_dir} has no files.",
                str(data_dir),
                "Download or copy the data first (see the example's README).",
            )
        ]
    return []
