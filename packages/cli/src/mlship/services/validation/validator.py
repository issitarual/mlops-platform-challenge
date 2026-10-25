"""Runs every static rule and merges all issues into one report."""

from collections.abc import Sequence
from pathlib import Path

from packaging.utils import canonicalize_name

from mlship.domain.validation import Severity, ValidationIssue, ValidationReport
from mlship.services.validation.data import check_data_dir
from mlship.services.validation.imports import check_imports
from mlship.services.validation.requirements import check_requirements, runtime_specifiers
from mlship.services.validation.source_rules import check_sources
from mlship.services.validation.sources import read_sources
from mlship.services.validation.train_script import check_train_script

SDK_PACKAGE = "mlship-sdk"


class ProjectValidator:
    def __init__(self, runtime_requirements: Sequence[str]) -> None:
        self._runtime = runtime_specifiers(runtime_requirements)
        self._provided = frozenset({*self._runtime, canonicalize_name(SDK_PACKAGE)})

    def validate(self, project_dir: Path, data_dir: Path | None) -> ValidationReport:
        """Parse, never import or run: every rule runs and every issue is reported."""
        if not project_dir.is_dir():
            missing = ValidationIssue(
                "PROJECT_DIR_MISSING",
                Severity.ERROR,
                f"{project_dir} is not a folder.",
                str(project_dir),
                "Pass the folder that holds train.py and requirements.txt, or create one "
                "with `mlship init`.",
            )
            return ValidationReport((missing,))
        requirements = check_requirements(project_dir, self._runtime)
        sources = read_sources(project_dir, exclude=(data_dir,) if data_dir else ())
        issues = [
            *check_train_script(project_dir),
            *requirements.issues,
            *sources.issues,
            *check_sources(sources),
        ]
        if requirements.readable:
            issues.extend(check_imports(sources, requirements.declared | self._provided))
        if data_dir is not None:
            issues.extend(check_data_dir(data_dir))
        return ValidationReport(tuple(issues))
