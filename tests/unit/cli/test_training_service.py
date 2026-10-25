import json
import sys
from pathlib import Path

import pytest
from fakes import FakeModelRegistry, FakeRunRepository
from training_fakes import GOOD_RESULT, RecordingTrainingRuntime, make_project, make_service

from mlship.adapters.hashing import sha256_file
from mlship.domain.errors import (
    EnvironmentBuildError,
    MlshipError,
    ProjectInvalidError,
    TrackingError,
    TrackingUnavailableError,
)
from mlship.domain.runs import RunStatus, tree_digest
from mlship.services.lineage import CODE_MAX_FILE_BYTES


def test_successful_run(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runtime = RecordingTrainingRuntime()
    logs: list[str] = []
    outcome = make_service(runtime).train(project, data, {}, model_name="demo", on_log=logs.append)

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
        make_service(runtime).train(project, data, {}, model_name="demo")
    assert {i.code for i in info.value.report.errors} == {
        "TRAIN_SYNTAX",
        "REQUIREMENTS_OPTION_BLOCKED",
    }
    assert runtime.builds == [] and runtime.specs == []


def test_container_environment_carries_seed_and_thread_limits_only(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runtime = RecordingTrainingRuntime()
    make_service(runtime).train(project, data, {"seed": "7"}, model_name="demo")
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
    outcome = make_service(RecordingTrainingRuntime()).train(project, data, {}, model_name="demo")
    assert outcome.params["seed"] == "42"


def test_non_integer_seed_is_rejected(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    with pytest.raises(MlshipError, match="not an integer"):
        make_service(RecordingTrainingRuntime()).train(
            project, data, {"seed": "abc"}, model_name="demo"
        )


def test_allow_pickle_can_be_disabled_by_the_platform(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    service = make_service(RecordingTrainingRuntime(), allow_pickle_enabled=False)
    with pytest.raises(MlshipError, match="disabled"):
        service.train(project, data, {}, model_name="demo", allow_pickle=True)


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
    outcome = make_service(RecordingTrainingRuntime(result, exit_code=1)).train(
        project, data, {}, model_name="demo"
    )
    assert not outcome.succeeded
    assert [i.code for i in outcome.issues] == ["TRAIN_RAISED"]


# --- E3: tracking, manifest and registration ------------------------------------------


def test_success_is_tracked_registered_and_described_by_the_manifest(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runs, registry = FakeRunRepository(), FakeModelRegistry()
    outcome = make_service(RecordingTrainingRuntime(), runs=runs, registry=registry).train(
        project, data, {"alpha": "1"}, model_name="demo"
    )
    run = runs.runs[outcome.run_id]
    manifest = json.loads((outcome.run_dir / "manifest.json").read_text())

    assert run["experiment"] == "demo" and run["status"] is RunStatus.FINISHED
    assert run["params"] == {"alpha": "1", "seed": "42"} and run["metrics"] == {"rmse": 0.5}
    assert set(run["artifacts"]) == {"model", "manifest.json", "run.json", "result.json"}
    assert outcome.model_version is not None and outcome.model_version.version == 1
    assert run["tags"]["mlship.model_version"] == "1"

    model_files = {"MLmodel": sha256_file(outcome.run_dir / "output" / "model" / "MLmodel")}
    assert manifest["model"]["files"] == model_files
    assert manifest["model"]["sha256"] == tree_digest(model_files)
    assert outcome.model_version.model_sha256 == manifest["model"]["sha256"]
    assert set(manifest["code"]["files"]) == {"train.py", "requirements.txt"}
    assert manifest["data"]["files"] == {"data.csv": sha256_file(data / "data.csv")}
    assert manifest["tool"] == {"version": "0.1.0", "git_sha": "abc123", "git_dirty": False}
    assert manifest["environment"]["image_tag"].startswith("mlship-env:")
    assert manifest["seed"] == 42 and manifest["status"] == "succeeded"


def test_code_is_every_project_file_except_outputs_caches_hidden_and_data(
    tmp_path: Path,
) -> None:
    project, _ = make_project(tmp_path)
    (project / "helpers.py").write_text("X = 1\n")
    (project / "config.yaml").write_text("depth: 3\n")  # read by train.py: changes the model
    (project / "__pycache__").mkdir()
    (project / "__pycache__" / "helpers.cpython-312.pyc").write_bytes(b"compiled")
    (project / ".env").write_text("TOKEN=never-recorded")
    (project / "data").mkdir()
    (project / "data" / "rows.csv").write_text("a\n1\n")
    service = make_service(RecordingTrainingRuntime())
    first = service.train(project, project / "data", {}, model_name="demo")
    second = service.train(project, project / "data", {}, model_name="demo")
    manifests = [json.loads((o.run_dir / "manifest.json").read_text()) for o in (first, second)]

    assert set(manifests[0]["code"]["files"]) == {
        "train.py",
        "requirements.txt",
        "helpers.py",
        "config.yaml",
    }
    assert manifests[0]["code"]["sha256"] == manifests[1]["code"]["sha256"], (
        "earlier runs under .mlship/ must not count as code"
    )
    assert manifests[0]["data"]["location"] == "data", "no local user path is recorded"
    assert set(manifests[0]["data"]["files"]) == {"rows.csv"}


def test_large_project_files_are_listed_not_hashed(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    (project / "weights.bin").write_bytes(b"0" * (CODE_MAX_FILE_BYTES + 1))
    outcome = make_service(RecordingTrainingRuntime()).train(project, data, {}, model_name="demo")
    code = json.loads((outcome.run_dir / "manifest.json").read_text())["code"]
    assert "weights.bin" not in code["files"]
    assert code["skipped_large_files"] == {"weights.bin": CODE_MAX_FILE_BYTES + 1}


def test_changing_a_config_file_changes_the_code_hash(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    service = make_service(RecordingTrainingRuntime())
    (project / "config.yaml").write_text("depth: 3\n")
    before = service.train(project, data, {}, model_name="demo")
    (project / "config.yaml").write_text("depth: 4\n")
    after = service.train(project, data, {}, model_name="demo")
    hashes = [
        json.loads((o.run_dir / "manifest.json").read_text())["code"]["sha256"]
        for o in (before, after)
    ]
    assert hashes[0] != hashes[1]


def test_each_successful_run_is_a_new_version(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    registry = FakeModelRegistry()
    service = make_service(RecordingTrainingRuntime(), registry=registry)
    versions = [service.train(project, data, {}, model_name="demo").model_version for _ in "ab"]
    assert [v.version for v in versions if v] == [1, 2]


def test_failed_training_is_tracked_as_failed_and_not_registered(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runs, registry = FakeRunRepository(), FakeModelRegistry()
    failed = {**GOOD_RESULT, "status": "failed", "model": None}
    outcome = make_service(
        RecordingTrainingRuntime(failed, exit_code=1), runs=runs, registry=registry
    ).train(project, data, {}, model_name="demo")
    run = runs.runs[outcome.run_id]
    assert run["status"] is RunStatus.FAILED
    assert "model" not in run["artifacts"] and registry.versions == {}
    assert json.loads((outcome.run_dir / "manifest.json").read_text())["model"] is None


def test_unreachable_tracking_fails_before_lock_and_build(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runtime = RecordingTrainingRuntime()
    with pytest.raises(TrackingUnavailableError):
        make_service(runtime, runs=FakeRunRepository(available=False)).train(
            project, data, {}, model_name="demo"
        )
    assert runtime.builds == [] and not (project / "mlship.lock").exists()


def test_environment_failure_closes_the_run_as_failed(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runs = FakeRunRepository()
    runtime = RecordingTrainingRuntime()
    runtime.selfcheck_failure = "broken image"
    with pytest.raises(EnvironmentBuildError):
        make_service(runtime, runs=runs).train(project, data, {}, model_name="demo")
    (run,) = runs.runs.values()
    assert run["status"] is RunStatus.FAILED
    assert "self-check" in run["tags"]["mlship.error"]


def test_interrupted_training_closes_the_run_as_killed(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runs = FakeRunRepository()

    class Interrupted(RecordingTrainingRuntime):
        def run_training(self, spec, on_log):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        make_service(Interrupted(), runs=runs).train(project, data, {}, model_name="demo")
    (run,) = runs.runs.values()
    assert run["status"] is RunStatus.KILLED


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra rights on Windows")
def test_model_with_a_symlink_is_refused_and_nothing_is_uploaded(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    secret = tmp_path / "host-secret.txt"
    secret.write_text("aws_secret_access_key = ...")

    class Exfiltrating(RecordingTrainingRuntime):
        def run_training(self, spec, on_log):  # type: ignore[no-untyped-def]
            job = super().run_training(spec, on_log)
            (spec.output_dir / "model" / "credentials").symlink_to(secret)
            return job

    runs, registry = FakeRunRepository(), FakeModelRegistry()
    outcome = make_service(Exfiltrating(), runs=runs, registry=registry).train(
        project, data, {}, model_name="demo"
    )
    assert not outcome.succeeded
    assert [i.code for i in outcome.issues] == ["MODEL_UNTRUSTED_FILES"]
    assert "model" not in runs.runs[outcome.run_id]["artifacts"] and registry.versions == {}


def test_oversized_model_is_refused_and_nothing_is_uploaded(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runs, registry = FakeRunRepository(), FakeModelRegistry()

    class Huge(RecordingTrainingRuntime):
        def run_training(self, spec, on_log):  # type: ignore[no-untyped-def]
            job = super().run_training(spec, on_log)
            (spec.output_dir / "model" / "blob.bin").write_bytes(b"0" * 2_000)
            return job

    outcome = make_service(Huge(), runs=runs, registry=registry, model_max_bytes=1_000).train(
        project, data, {}, model_name="demo"
    )
    assert [i.code for i in outcome.issues] == ["MODEL_TOO_LARGE"]
    assert "model" not in runs.runs[outcome.run_id]["artifacts"] and registry.versions == {}


def test_param_problems_are_reported_together_before_anything_runs(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runtime, runs = RecordingTrainingRuntime(), FakeRunRepository()
    params = {"lr (fast)": "1", "db_password": "x", "notes": "n" * 6_001}
    with pytest.raises(ProjectInvalidError) as info:
        make_service(runtime, runs=runs).train(project, data, params, model_name="demo")
    assert {i.code for i in info.value.report.errors} == {
        "PARAM_NAME_INVALID",
        "PARAM_LOOKS_SECRET",
        "PARAM_VALUE_TOO_LONG",
    }
    assert runs.runs == {} and runtime.builds == []


class _BrokenTracking(FakeRunRepository):
    """Accepts the run, then fails while uploading (server down, disk full...)."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def log(self, run_id, **kwargs):  # type: ignore[no-untyped-def]
        raise self.error


def test_tracking_failure_after_training_keeps_outputs_and_closes_the_run(
    tmp_path: Path,
) -> None:
    project, data = make_project(tmp_path)
    runs = _BrokenTracking(TrackingError("MLflow rejected the run data."))
    with pytest.raises(MlshipError, match="recording it in MLflow failed") as info:
        make_service(RecordingTrainingRuntime(), runs=runs).train(
            project, data, {}, model_name="demo"
        )
    ((run_id, run),) = runs.runs.items()
    assert run["status"] is RunStatus.FAILED
    assert str(project / ".mlship" / "runs" / run_id) in info.value.hint
    assert (project / ".mlship" / "runs" / run_id / "manifest.json").is_file()


def test_ctrl_c_during_the_upload_never_leaves_the_run_running(tmp_path: Path) -> None:
    project, data = make_project(tmp_path)
    runs = _BrokenTracking(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        make_service(RecordingTrainingRuntime(), runs=runs).train(
            project, data, {}, model_name="demo"
        )
    (run,) = runs.runs.values()
    assert run["status"] is RunStatus.KILLED
