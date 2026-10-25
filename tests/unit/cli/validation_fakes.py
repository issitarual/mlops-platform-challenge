"""Validation and lock services on in-memory fakes (no uv, no network)."""

from collections.abc import Sequence
from pathlib import Path

from fakes import FakeContainerRuntime, FakeDockerfile, FakeLockResolver, FakeRuntimePackager

from mlship.domain.errors import LockResolutionError
from mlship.services.environment_service import EnvironmentService
from mlship.services.validation.validator import ProjectValidator
from mlship.services.validation_service import ValidationService


class FailingResolver(FakeLockResolver):
    def resolve(self, requirement_files: Sequence[Path], output: Path) -> None:
        self.calls.append([f.read_text() for f in requirement_files])
        raise LockResolutionError(
            "Could not resolve your requirements together with the mlship runtime.",
            hint="Check package names and version pins in requirements.txt.",
            details="  \u00d7 No solution found when resolving dependencies:\n"
            "  ╰─▶ Because there is no version of nopkg==9.9 and you require nopkg==9.9, "
            "we can conclude that your requirements are unsatisfiable.\n",
        )


def validation_environment(resolver: FakeLockResolver) -> EnvironmentService:
    return EnvironmentService(
        resolver,
        FakeRuntimePackager(),
        FakeDockerfile(),
        FakeContainerRuntime(),
        python_version="3.12",
        python_platform="x86_64-manylinux_2_28",
        docker_platform="linux/amd64",
    )


def validation_service(resolver: FakeLockResolver) -> ValidationService:
    validator = ProjectValidator(FakeRuntimePackager().requirements())
    return ValidationService(validator, validation_environment(resolver))
