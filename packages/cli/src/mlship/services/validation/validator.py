"""Runs every rule and merges all issues into one report."""

from pathlib import Path

from mlship.domain.validation import ValidationReport
from mlship.services.validation.data import check_data_dir
from mlship.services.validation.requirements import check_requirements
from mlship.services.validation.train_script import check_train_script


class ProjectValidator:
    def validate(self, project_dir: Path, data_dir: Path | None) -> ValidationReport:
        """Run train.py, requirements.txt and data rules; never stop at the first issue."""
        issues = [*check_train_script(project_dir), *check_requirements(project_dir)]
        if data_dir is not None:
            issues.extend(check_data_dir(data_dir))
        return ValidationReport(tuple(issues))
