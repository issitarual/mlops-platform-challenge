from pathlib import Path

import pytest
from training_fakes import RecordingTrainingRuntime, make_project, make_service

from mlship.domain.errors import MlshipError, ProjectInvalidError


def test_successful_run(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runtime = RecordingTrainingRuntime()
    logs: list[str] = []
    outcome = make_service(runtime).train(project, data, {}, on_log=logs.append)

    assert outcome.succeeded
    assert outcome.metrics == {"rmse": 0.5}
    assert outcome.serialization == "skops"
    assert outcome.run_dir == project / ".mlship" / "runs" / outcome.run_id
    assert logs == ["training...\n"]


def test_invalid_project_stops_before_building_anything(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    (project / "requirements.txt").write_text("--extra-index-url https://evil.example\n")
    (project / "train.py").write_text("def fit(:\n")
    runtime = RecordingTrainingRuntime()
    with pytest.raises(ProjectInvalidError) as info:
        make_service(runtime).train(project, data, {})
    assert {i.code for i in info.value.report.errors} == {
        "TRAIN_SYNTAX",
        "REQUIREMENTS_OPTION_BLOCKED",
    }
    assert runtime.builds == [] and runtime.specs == []


def test_container_environment_carries_seed_and_thread_limits_only(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runtime = RecordingTrainingRuntime()
    make_service(runtime).train(project, data, {"seed": "7"})
    assert dict(runtime.specs[0].environment) == {
        "MLSHIP_RUN_CONFIG": "/mlship/run.json",
        "PYTHONHASHSEED": "7",
        "HOME": "/tmp",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }


def test_seed_defaults_to_42_and_is_visible_to_train(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    outcome = make_service(RecordingTrainingRuntime()).train(project, data, {})
    assert outcome.params["seed"] == "42"


def test_non_integer_seed_is_rejected(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    with pytest.raises(MlshipError, match="not an integer"):
        make_service(RecordingTrainingRuntime()).train(project, data, {"seed": "abc"})


def test_allow_pickle_can_be_disabled_by_the_platform(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    service = make_service(RecordingTrainingRuntime(), allow_pickle_enabled=False)
    with pytest.raises(MlshipError, match="disabled"):
        service.train(project, data, {}, allow_pickle=True)


def test_failed_run_returns_the_runner_issues(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    result = {
        "schema_version": 1,
        "status": "failed",
        "errors": [
            {"code": "TRAIN_RAISED", "message": "boom", "hint": "h", "location": "train.py"}
        ],
        "metrics": {},
        "model": None,
    }
    outcome = make_service(RecordingTrainingRuntime(result, exit_code=1)).train(project, data, {})
    assert not outcome.succeeded
    assert [i.code for i in outcome.issues] == ["TRAIN_RAISED"]
