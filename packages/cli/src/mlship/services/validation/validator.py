"""Runs every rule and merges all issues into one report."""

from pathlib import Path

from mlship.domain.validation import ValidationReport


class ProjectValidator:
    def validate(self, project_dir: Path, data_dir: Path | None) -> ValidationReport:
        """Run train.py, requirements.txt and data rules; never stop at the first issue."""
        raise NotImplementedError("E5")
