"""``mlship validate`` and ``train --strict``: every issue, resolution, exit codes, JSON."""

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeLockResolver
from training_fakes import RecordingTrainingRuntime, make_project, make_service
from typer.testing import CliRunner
from validation_fakes import FailingResolver, validation_environment, validation_service

from mlship.controllers.app import app
from mlship.domain.errors import LockResolutionError, ProjectInvalidError
from mlship.domain.validation import ResolutionCheck
from mlship.services.validation_service import ValidationService

runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, Path]:
    return make_project(tmp_path)


def _invoke(
    monkeypatch: pytest.MonkeyPatch, service: ValidationService, *args: str
) -> tuple[int, str]:
    services = SimpleNamespace(validation=service)
    monkeypatch.setattr("mlship.controllers.validate.load_services", lambda: services)
    result = runner.invoke(app, ["validate", *args])
    return result.exit_code, " ".join(result.output.split())


def test_a_valid_project_resolves_without_touching_the_project(
    project: tuple[Path, Path],
) -> None:
    resolver = FakeLockResolver()
    report, resolution = validation_service(resolver).validate(
        project[0], project[1], offline=False
    )
    assert [i.code for i in report.issues] == ["LOCK_MISSING"]
    assert resolution is ResolutionCheck.PASSED and len(resolver.calls) == 1
    assert not (project[0] / "mlship.lock").exists(), "validate is read-only"


def test_an_intact_current_lock_answers_without_resolving_or_network(
    project: tuple[Path, Path],
) -> None:
    validation_environment(FakeLockResolver()).lock(project[0])
    resolver = FakeLockResolver()
    for offline in (False, True):
        report, resolution = validation_service(resolver).validate(
            project[0], project[1], offline=offline
        )
        assert report.issues == () and resolution is ResolutionCheck.PASSED
    assert resolver.calls == []


def test_a_stale_or_edited_lock_is_a_warning(project: tuple[Path, Path]) -> None:
    validation_environment(FakeLockResolver()).lock(project[0])
    lock = project[0] / "mlship.lock"
    lock.write_text(lock.read_text() + "evil==1.0\n")
    report, _ = validation_service(FakeLockResolver()).validate(project[0], None, offline=True)
    assert [i.code for i in report.warnings] == ["LOCK_MODIFIED"]
    (project[0] / "requirements.txt").write_text("scikit-learn==1.9.2\n")
    report, _ = validation_service(FakeLockResolver()).validate(project[0], None, offline=True)
    assert [i.code for i in report.warnings] == ["LOCK_STALE"]
    assert "mlship lock" in report.warnings[0].hint


def test_lock_writes_only_after_the_requirements_pass(
    project: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    services = SimpleNamespace(validation=validation_service(FakeLockResolver()))
    monkeypatch.setattr("mlship.controllers.lock.load_services", lambda: services)
    (project[0] / "requirements.txt").write_text("--index-url https://evil\nnumpy==2.0.0\n")
    (project[0] / "train.py").write_text("def broken(:\n")
    refused = runner.invoke(app, ["lock", str(project[0])])
    assert refused.exit_code == 1 and "REQUIREMENTS_OPTION_BLOCKED" in refused.output
    assert "TRAIN_SYNTAX" not in refused.output, "the lock does not depend on train.py"
    assert not (project[0] / "mlship.lock").exists()

    (project[0] / "requirements.txt").write_text("numpy==2.0.0\n")
    written = runner.invoke(app, ["lock", str(project[0])])
    assert written.exit_code == 0 and "Wrote mlship.lock" in written.output
    again = runner.invoke(app, ["lock", str(project[0])])
    assert "nothing changed" in again.output


def test_unresolvable_requirements_become_an_issue_with_uvs_reason(
    project: tuple[Path, Path],
) -> None:
    report, resolution = validation_service(FailingResolver()).validate(
        project[0], project[1], offline=False
    )
    (issue,) = report.errors
    assert resolution is ResolutionCheck.FAILED and issue.code == "REQUIREMENTS_UNRESOLVABLE"
    assert "no version of nopkg==9.9" in issue.message and "╰" not in issue.message


def test_an_unreachable_index_is_not_blamed_on_the_requirements(
    project: tuple[Path, Path],
) -> None:
    class OfflineResolver(FakeLockResolver):
        def resolve(self, requirement_files: Sequence[Path], output: Path) -> None:
            raise LockResolutionError(
                "Could not resolve.",
                details="error: Failed to fetch: `https://pypi.org/simple/numpy/`\n"
                "  Caused by: dns error: failed to lookup address information\n",
            )

    report, _ = validation_service(OfflineResolver()).validate(project[0], None, offline=False)
    (issue,) = report.errors
    assert issue.code == "REQUIREMENTS_INDEX_UNREACHABLE" and "--offline" in issue.hint


def test_resolution_is_skipped_offline_or_when_requirements_are_already_wrong(
    project: tuple[Path, Path],
) -> None:
    resolver = FakeLockResolver()
    service = validation_service(resolver)
    assert service.validate(project[0], None, offline=True)[1] is ResolutionCheck.OFFLINE
    (project[0] / "requirements.txt").write_text("-e .\n")
    assert service.validate(project[0], None, offline=False)[1] is ResolutionCheck.SKIPPED
    assert resolver.calls == []


def test_warnings_pass_unless_strict(
    project: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    (project[0] / "requirements.txt").write_text("scikit-learn\n")
    service = validation_service(FakeLockResolver())
    code, output = _invoke(monkeypatch, service, str(project[0]), "--offline")
    assert code == 0 and "REQUIREMENTS_UNPINNED" in output and "Project is valid" in output
    code, output = _invoke(monkeypatch, service, str(project[0]), "--offline", "--strict")
    assert code == 1 and "warnings count as errors with --strict" in output


def test_json_output_lists_every_issue_for_ci(
    project: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    (project[0] / "requirements.txt").write_text("pandas[perf]\n--index-url https://x\n")
    code, output = _invoke(
        monkeypatch, validation_service(FakeLockResolver()), str(project[0]), "--format", "json"
    )
    document = json.loads(output)
    assert document["schema_version"] == 1
    assert code == 1 and document["valid"] is False
    assert document["resolution"] == "skipped"
    assert [i["code"] for i in document["issues"]] == [
        "REQUIREMENTS_UNPINNED",
        "REQUIREMENTS_OPTION_BLOCKED",
    ]
    assert document["issues"][0]["message"] == "'pandas' has no exact version."


def test_text_output_shows_user_text_literally(
    project: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    (project[0] / "requirements.txt").write_text("pandas[performance]=2\n")
    _, output = _invoke(
        monkeypatch, validation_service(FakeLockResolver()), str(project[0]), "--offline"
    )
    assert "pandas[performance]=2" in output


def test_the_next_command_names_the_project_it_validated(
    project: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_environment(FakeLockResolver()).lock(project[0])
    args = (str(project[0]), "--data", str(project[1]))
    code, output = _invoke(monkeypatch, validation_service(FakeLockResolver()), *args)
    assert code == 0, output
    expected = f"Next: mlship train {project[0]} --data {project[1]}"
    assert expected.replace(" ", "") in output.replace(" ", ""), "long paths wrap in the terminal"


def test_train_strict_refuses_warnings_before_any_work(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    (project / "requirements.txt").write_text("scikit-learn\n")
    runtime = RecordingTrainingRuntime()
    with pytest.raises(ProjectInvalidError):
        make_service(runtime).train(project, data, {}, model_name="demo", strict=True)
    assert runtime.specs == [] and not (project / "mlship.lock").exists()
